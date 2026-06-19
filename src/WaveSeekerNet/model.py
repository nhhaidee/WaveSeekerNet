"""
WaveSeekerNet model module.

Contains the core neural network architecture (:class:`WaveSeekerNet`),
the per-block attention-like encoder (:class:`WaveSeekerBlock`), and
the scikit-learn-compatible classifier wrapper
(:class:`WaveSeekerClassifier`).
"""
from __future__ import annotations

import logging
import sys
from time import time
from typing import Optional, Sequence, Type

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.cuda import is_available as is_gpu_available
from torch.nn import RMSNorm
from pytorch_optimizer import create_optimizer

from WaveSeekerNet.sub_modules.wavelet_transform import WaveNETHead
from WaveSeekerNet.sub_modules.fourier_transform import FNETHead
from WaveSeekerNet.sub_modules.gmlp import gMLPBlock
from WaveSeekerNet.sub_modules.smoe import SparseMoE, WaveExpert
from WaveSeekerNet.sub_modules.classification_head import ClassificationHead

from WaveSeekerNet.sub_modules.lib.star_layer import StarLayer
from WaveSeekerNet.sub_modules.lib.make_patches import MakePatches
from WaveSeekerNet.sub_modules.lib.pos_encoding import PositionalEncoding
from WaveSeekerNet.sub_modules.lib.global_pooling import GlobalExpectationPooling
from WaveSeekerNet.sub_modules.lib.activation import ErMish
from WaveSeekerNet.sub_modules.lib.kan_layer import KANLinear

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)


class WaveSeekerBlock(nn.Module):
    """A single WaveSeekerNet encoder block.

    Applies parallel token-mixing heads (Wavelet, FFT, gMLP) in parallel,
    merges their outputs via a :class:`StarLayer`, then processes along
    the hidden dimension using either a Sparse Mixture-of-Experts
    (:class:`SparseMoE`) or a single :class:`WaveExpert`.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the patch embeddings.
    n_patches : int
        Number of patches (sequence length after patch extraction).
    wavelet_names : Sequence[str]
        Names of wavelet filters to use (one WaveNETHead per wavelet).
    ffn_dropout : float
        Dropout probability applied inside feed-forward sub-layers.
    use_fft : bool
        Whether to include the Fourier (FNet) head.
    use_wavelet : bool
        Whether to include wavelet heads.
    device : torch.device
        Compute device.
    use_smoe : bool
        Use Sparse Mixture-of-Experts; otherwise use a single WaveExpert.
    activation : type
        Activation function class (instantiated internally).
    use_gmlp : bool
        Whether to include the gMLP head.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_patches: int,
        wavelet_names: Sequence[str],
        ffn_dropout: float,
        use_fft: bool,
        use_wavelet: bool,
        device: torch.device,
        use_smoe: bool,
        activation: type,
        use_gmlp: bool,
    ) -> None:
        super().__init__()

        self.device = device
        self.use_fft = use_fft
        self.use_wavelet = use_wavelet
        self.use_gmlp = use_gmlp
        self.use_smoe = use_smoe

        out_dim = 0
        self.norm_1 = RMSNorm(embedding_dim, eps=1e-8)

        # Wavelet heads
        if self.use_wavelet:
            self.n_wavelets = len(wavelet_names)
            self.wave_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(embedding_dim, embedding_dim),
                        WaveNETHead(wavelet_name, embedding_dim, n_patches, activation),
                    )
                    for wavelet_name in wavelet_names
                ]
            )
            out_dim += embedding_dim * len(wavelet_names)

        # FFT head
        if self.use_fft:
            self.fft_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                FNETHead(embedding_dim, activation),
            )
            out_dim += embedding_dim

        # gMLP head
        if self.use_gmlp:
            self.gmlp_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                gMLPBlock(
                    embedding_dim=embedding_dim,
                    ffn_dropout=ffn_dropout,
                    n_patches=n_patches,
                    activation=activation,
                ),
            )
            out_dim += embedding_dim

        # Merge heads
        self.dropout = nn.Dropout1d(ffn_dropout)

        if use_smoe:
            self.star = StarLayer(out_dim, embedding_dim, n_patches, activation)
            self.proj_concat = SparseMoE(
                dim=embedding_dim,
                top_k=3,
                activation=activation,
                ffn_dropout=0.25,
                n_patches=n_patches,
            )
        else:
            self.star = StarLayer(
                emb_in=out_dim,
                emb_out=embedding_dim,
                n_patches=n_patches,
                activation=activation,
            )
            self.proj_concat = WaveExpert(
                in_embed=embedding_dim,
                ffn_dropout=ffn_dropout,
                activation=activation,
            )

        self.norm_2 = RMSNorm(embedding_dim, eps=1e-8)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through one WaveSeekerBlock.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(B, n_patches, embedding_dim)``.

        Returns
        -------
        output : torch.Tensor
            Same shape as *inputs*, after token mixing and normalisation.
        z_loss : torch.Tensor
            Auxiliary routing loss (scalar zero when SMoE is disabled).
        """
        x_c: list[torch.Tensor] = []
        x_n = self.norm_1(inputs)

        if self.use_wavelet:
            for wave_head in self.wave_heads:
                x_c.append(wave_head(x_n))

        if self.use_fft:
            x_c.append(self.fft_head(x_n))

        if self.use_gmlp:
            x_c.append(self.gmlp_head(x_n))

        x_merged = torch.concat(x_c, dim=-1)
        x_merged = self.star(x_merged) + inputs
        x_merged = self.norm_2(x_merged)

        if self.use_smoe:
            x_out, z_loss = self.proj_concat(x_merged)
        else:
            z_loss = torch.tensor(0.0, device=inputs.device)
            x_out = self.proj_concat(x_merged)

        return self.dropout(x_out + x_merged), z_loss


class WaveSeekerNet(nn.Module):
    """Core WaveSeekerNet neural network (PyTorch nn.Module).

    Encodes 2-D genomic representations (FCGR images or protein matrices)
    as non-overlapping patches, processes them through ``n_blocks`` stacked
    :class:`WaveSeekerBlock` encoders, pools the result with
    :class:`GlobalExpectationPooling`, and outputs class logits (and
    optionally softmax probabilities).

    Parameters
    ----------
    seq_L : int
        Sequence-length dimension of the input (height of the 2-D matrix).
    res_L : int
        Residue/feature-length dimension (width of the 2-D matrix).
    n_channels : int
        Number of input channels.
    patch_size : tuple[int, int]
        ``(height, width)`` of each patch.
    n_out : int
        Number of output classes.
    device : torch.device
        Compute device.
    emb_dim : int
        Patch embedding dimension.
    wavelet_names : Sequence[str]
        Wavelet filter names passed to each :class:`WaveNETHead`.
    wave_dropout : float
        Dropout rate inside :class:`WaveSeekerBlock` sub-layers.
    use_fft : bool
        Include the Fourier (FNet) token-mixing head.
    use_wavelets : bool
        Include wavelet token-mixing head(s).
    n_blocks : int
        Number of stacked encoder blocks.
    final_dropout : float
        Dropout rate inside the classification head.
    final_hidden_size : int
        Hidden size of the classification head.
    return_probs : bool
        If ``True``, the model also returns softmax class probabilities.
    use_kan : bool
        Use KAN layers inside the classification head.
    use_smoe : bool
        Use Sparse MoE inside encoder blocks.
    patch_mode : str
        One of ``"patch"``, ``"compress"``, or ``"full"``.
    activation : type
        Activation function class (instantiated internally by sub-modules).
    use_gmlp : bool
        Include the gMLP token-mixing head.
    """

    def __init__(
        self,
        seq_L: int,
        res_L: int,
        n_channels: int,
        patch_size: tuple[int, int],
        n_out: int,
        device: torch.device,
        emb_dim: int,
        wavelet_names: Sequence[str],
        wave_dropout: float,
        use_fft: bool,
        use_wavelets: bool,
        n_blocks: int,
        final_dropout: float,
        final_hidden_size: int,
        return_probs: bool,
        use_kan: bool,
        use_smoe: bool,
        patch_mode: str,
        activation: type,
        use_gmlp: bool,
    ) -> None:
        super().__init__()

        self.use_kan = use_kan
        self.use_smoe = use_smoe
        self.patch_mode = patch_mode
        self.return_probs = return_probs
        self.seq_L = seq_L
        self.res_L = res_L
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.n_out = n_out
        self.emb_dim = emb_dim
        self.wavelet_names = list(wavelet_names)
        self.wave_dropout = wave_dropout
        self.use_fft = use_fft
        self.use_wavelets = use_wavelets
        self.n_blocks = n_blocks
        self.final_dropout = final_dropout
        self.final_hidden_size = final_hidden_size
        self.device = device
        self.activation = activation
        self.use_gmlp = use_gmlp

        self.patch_dropout = nn.Dropout1d(0.5)
        self.make_patches = nn.ModuleList(
            [
                MakePatches(
                    patch_width=patch_size[1],
                    patch_height=patch_size[0],
                    emb_dim=self.emb_dim,
                    patch_mode=patch_mode,
                )
                for _ in range(self.n_channels)
            ]
        )

        if self.patch_mode == "full":
            self.n_patches = 160
            self.a_pool = nn.AdaptiveAvgPool1d(self.n_patches)
            self.pool_pos = PositionalEncoding(self.emb_dim)
        elif self.patch_mode == "compress":
            seq_area = self.seq_L * self.res_L
            self.n_patches = (
                seq_area // (self.patch_size[0] * self.patch_size[1])
            ) // 2
            self.a_pool = nn.AdaptiveAvgPool1d(self.n_patches)
            self.pool_pos = PositionalEncoding(self.emb_dim)
        elif self.patch_mode == "patch":
            seq_area = self.seq_L * self.res_L
            self.n_patches = seq_area // (self.patch_size[0] * self.patch_size[1])

        self.self_attention_enc = nn.ModuleList(
            [
                WaveSeekerBlock(
                    embedding_dim=self.emb_dim,
                    n_patches=self.n_patches,
                    wavelet_names=self.wavelet_names,
                    ffn_dropout=self.wave_dropout,
                    use_fft=self.use_fft,
                    use_wavelet=self.use_wavelets,
                    device=device,
                    use_smoe=self.use_smoe,
                    activation=self.activation,
                    use_gmlp=self.use_gmlp,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.create_tokens = GlobalExpectationPooling()
        self.classifier = ClassificationHead(
            self.emb_dim,
            self.n_out,
            self.activation,
            self.return_probs,
            self.use_kan,
            self.final_dropout,
            self.final_hidden_size,
            grid_size=32,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the full WaveSeekerNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, n_channels, res_L, seq_L)`` for
            multi-channel inputs, or ``(B, res_L, seq_L)`` for single-channel.

        Returns
        -------
        When ``return_probs=True``:
            ``(logits, probs, z_loss)`` — shapes
            ``(B, n_out), (B, n_out), scalar``.
        When ``return_probs=False``:
            ``(logits, z_loss)`` — shapes ``(B, n_out), scalar``.
        """
        z_loss_total = torch.tensor(0.0, device=x.device)

        # Patch extraction
        if x.dim() == 3:
            x_patch = self.make_patches[0](x.unsqueeze(1))
        else:
            channel_patches = [
                self.make_patches[i](x[:, i : i + 1, :, :])
                for i in range(self.n_channels)
            ]
            x_patch = torch.concat(channel_patches, dim=-2)

        x_patch = self.patch_dropout(x_patch)

        if self.patch_mode in ("full", "compress"):
            x_patch = self.a_pool(x_patch.transpose(-1, -2)).transpose(-1, -2)
            x_patch = self.pool_pos(x_patch)

        # Encoder blocks
        for block in self.self_attention_enc:
            x_patch, z_loss = block(x_patch)
            z_loss_total = z_loss_total + z_loss

        cls_tokens = self.create_tokens(x_patch)

        if self.return_probs:
            x_logit, x_probs = self.classifier(cls_tokens)
            return x_logit, x_probs, z_loss_total

        return self.classifier(cls_tokens), z_loss_total


class WaveSeekerClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn-compatible classifier wrapping :class:`WaveSeekerNet`.

    Follows the scikit-learn estimator API: ``fit``, ``predict``,
    ``predict_proba``, and ``score`` (inherited from
    :class:`~sklearn.base.ClassifierMixin`).
    Compatible with :class:`~sklearn.pipeline.Pipeline`,
    :class:`~sklearn.model_selection.GridSearchCV`, and
    :func:`~sklearn.model_selection.cross_val_score`.

    Parameters
    ----------
    seq_L : int
        Sequence-length dimension of the input.
    res_L : int
        Residue/feature-length dimension of the input.
    n_channels : int
        Number of input channels.
    patch_size : tuple[int, int]
        ``(height, width)`` of each patch.
    n_out : int
        Number of output classes.
    emb_dim : int
        Patch embedding dimension. Default ``196``.
    wavelet_names : list[str] or None
        Wavelet filter names. Default ``["bior3.3", "sym4"]``.
    wave_dropout : float
        Dropout in WaveSeekerBlocks. Default ``0.5``.
    use_fft : bool
        Include Fourier head. Default ``True``.
    use_wavelets : bool
        Include wavelet heads. Default ``True``.
    n_blocks : int
        Number of encoder blocks. Default ``2``.
    final_dropout : float
        Dropout in the classification head. Default ``0.5``.
    final_hidden_size : int
        Hidden size of the classification head. Default ``32``.
    batch_size : int
        Mini-batch size. Default ``64``.
    epochs : int
        Number of training epochs. Default ``30``.
    lr : float
        Initial learning rate. Default ``1e-3``.
    wd : float
        Weight decay. Default ``0.0``.
    optimizer_name : str
        Optimizer name (from pytorch-optimizer). Default ``"Adan"``.
    use_gc : bool
        Use gradient centralisation. Default ``True``.
    use_lookahead : bool
        Wrap the optimizer with Lookahead. Default ``True``.
    use_kan : bool
        Use KAN layers in the classification head. Default ``True``.
    use_smoe : bool
        Use Sparse MoE in encoder blocks. Default ``True``.
    patch_mode : str
        One of ``"patch"``, ``"compress"``, ``"full"``. Default ``"compress"``.
    activation : type
        Activation function class. Default :class:`~WaveSeekerNet.sub_modules.lib.activation.ErMish`.
    use_gmlp : bool
        Include gMLP head. Default ``True``.
    return_probs : bool
        Return softmax probabilities from the model. Default ``True``.

    Attributes
    ----------
    model_ : WaveSeekerNet
        The fitted underlying PyTorch model. Set by :meth:`fit`.
    device_ : torch.device
        Device used during training. Set by :meth:`fit`.
    loss_history_train_ : list[tuple[float, float, float]]
        Per-epoch ``(bce_loss, kan_loss, smoe_loss)`` tuples.
    loss_history_valid_ : list[float]
        Per-epoch validation losses.
    score_history_ : list[float]
        Per-epoch balanced accuracy scores on the validation set.
    """

    def __init__(
        self,
        seq_L: int,
        res_L: int,
        n_channels: int,
        patch_size: tuple[int, int],
        n_out: int,
        emb_dim: int = 196,
        wavelet_names: Optional[list[str]] = None,
        wave_dropout: float = 0.5,
        use_fft: bool = True,
        use_wavelets: bool = True,
        n_blocks: int = 2,
        final_dropout: float = 0.5,
        final_hidden_size: int = 32,
        batch_size: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        wd: float = 0.0,
        optimizer_name: str = "Adan",
        use_gc: bool = True,
        use_lookahead: bool = True,
        use_kan: bool = True,
        use_smoe: bool = True,
        patch_mode: str = "compress",
        activation: Type[nn.Module] = ErMish,
        use_gmlp: bool = True,
        return_probs: bool = True,
    ) -> None:
        self.seq_L = seq_L
        self.res_L = res_L
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.n_out = n_out
        self.emb_dim = emb_dim
        self.wavelet_names = wavelet_names
        self.wave_dropout = wave_dropout
        self.use_fft = use_fft
        self.use_wavelets = use_wavelets
        self.n_blocks = n_blocks
        self.final_dropout = final_dropout
        self.final_hidden_size = final_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.optimizer_name = optimizer_name
        self.use_gc = use_gc
        self.use_lookahead = use_lookahead
        self.use_kan = use_kan
        self.use_smoe = use_smoe
        self.patch_mode = patch_mode
        self.activation = activation
        self.use_gmlp = use_gmlp
        self.return_probs = return_probs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_wavelet_names(self) -> list[str]:
        """Return the wavelet names list, applying the default if not set."""
        return self.wavelet_names if self.wavelet_names is not None else ["bior3.3", "sym4"]

    @staticmethod
    def _get_device() -> torch.device:
        """Return the best available compute device."""
        return torch.device("cpu")
        #return torch.device("cuda:0" if is_gpu_available() else "cpu")

    def _check_is_fitted(self) -> None:
        """Raise :exc:`~sklearn.exceptions.NotFittedError` if not yet trained."""
        if not hasattr(self, "model_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def _make_dataloader(
        self,
        *tensors: torch.Tensor,
        shuffle: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Wrap tensors in a :class:`~torch.utils.data.TensorDataset` DataLoader."""
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors),
            shuffle=shuffle,
            batch_size=self.batch_size,
        )

    def _init_model(self) -> None:
        """
        Centralized internal helper to initialize the WaveSeekerNet architecture.
        Sets self.device_ and self.model_.
        """
        self.device_ = self._get_device()
        logger.info("Using device: %s", self.device_ )
        wavelet_names = self._resolve_wavelet_names()

        self.model_ = WaveSeekerNet(
            seq_L=self.seq_L,
            res_L=self.res_L,
            n_channels=self.n_channels,
            device=self.device_,
            patch_size=self.patch_size,
            n_out=self.n_out,
            emb_dim=self.emb_dim,
            wavelet_names=wavelet_names,
            wave_dropout=self.wave_dropout,
            use_fft=self.use_fft,
            use_wavelets=self.use_wavelets,
            n_blocks=self.n_blocks,
            final_dropout=self.final_dropout,
            final_hidden_size=self.final_hidden_size,
            use_smoe=self.use_smoe,
            return_probs=self.return_probs,
            patch_mode=self.patch_mode,
            use_kan=self.use_kan,
            activation=self.activation,
            use_gmlp=self.use_gmlp,
        ).to(self.device_)
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_va: Optional[np.ndarray] = None,
        y_va: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> "WaveSeekerClassifier":
        """Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels (integer-encoded).
        X_va : np.ndarray, optional
            Validation features. If ``None``, 10 % of training data is held
            out via stratified split.
        y_va : np.ndarray, optional
            Validation labels (required when *X_va* is provided).
        save_path : str, optional
            If given, model weights are saved to this path after training
            via :func:`torch.save`.

        Returns
        -------
        self : WaveSeekerClassifier
            Fitted estimator (enables method chaining).
        """
        self._init_model()
        use_autocast = is_gpu_available()
        device_type = "cuda" if use_autocast else "cpu"

        self.loss_history_train_: list[tuple[float, float, float]] = []
        self.loss_history_valid_: list[float] = []
        self.score_history_: list[float] = []

        # ---- Data preparation ----
        if X_va is not None:
            X_train_t = torch.tensor(X)
            y_train_t = torch.tensor(y)
            X_valid_t = torch.tensor(X_va)
            y_valid_t = torch.tensor(y_va)
        else:
            X_tr, X_va_s, y_tr, y_va_s = train_test_split(
                X, y, test_size=0.10, stratify=y, random_state=0
            )
            X_train_t = torch.tensor(X_tr)
            y_train_t = torch.tensor(y_tr)
            X_valid_t = torch.tensor(X_va_s)
            y_valid_t = torch.tensor(y_va_s)

        loader_train = self._make_dataloader(X_train_t, y_train_t, shuffle=True)
        loader_valid = self._make_dataloader(X_valid_t, y_valid_t)

        trainable = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model_.parameters())
        logger.info("Trainable parameters: %d / %d total", trainable, total)

        # ---- Optimizer & scheduler ----
        optimizer = create_optimizer(
            self.model_,
            wd_ban_list=[
                name
                for name, _ in self.model_.named_parameters()
                if "m_scaler" in name or "rms_scaler" in name
            ],
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.wd,
            use_lookahead=self.use_lookahead,
            use_gc=self.use_gc,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            steps_per_epoch=len(loader_train),
            epochs=self.epochs,
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device_)
        scaler = torch.amp.GradScaler(device_type)

        # ---- Training loop ----
        for epoch in range(self.epochs):
            train_start = time()

            self.model_.train()
            bce_loss = kan_loss = mh_moe_loss = 0.0

            for x_in, y_in in loader_train:
                x_in, y_in = x_in.to(self.device_), y_in.to(self.device_)

                with torch.amp.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    if self.return_probs:
                        x_logit, _, moe_loss = self.model_(x_in)
                    else:
                        x_logit, moe_loss = self.model_(x_in)

                    loss_1 = loss_fn(x_logit, y_in)
                    total_loss = loss_1

                    if self.use_kan:
                        kan_layers = [
                            layer
                            for layer in self.model_.classifier.logits
                            if isinstance(layer, KANLinear)
                        ]
                        k = sum(
                            layer.regularization_loss(1.0, 1.0)
                            for layer in kan_layers
                        )
                        k = 0.01 * k / max(1, len(kan_layers))
                        total_loss = total_loss + k

                    if self.use_smoe:
                        sa_moe = moe_loss * 0.1
                        total_loss = total_loss + sa_moe

                bce_loss += loss_1.item()
                if self.use_kan:
                    kan_loss += k.item()
                if self.use_smoe:
                    mh_moe_loss += sa_moe.item()

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            train_time = time() - train_start
            n_batches = len(loader_train)
            bce_loss /= n_batches
            kan_loss /= n_batches
            mh_moe_loss /= n_batches

            # ---- Validation ----
            infer_start = time()
            self.model_.eval()
            val_loss = 0.0
            val_pred: list[int] = []
            val_labels: list[int] = []

            with torch.no_grad():
                for x_v, y_v in loader_valid:
                    x_v, y_v = x_v.to(self.device_), y_v.to(self.device_)
                    with torch.amp.autocast(
                        device_type=device_type,
                        dtype=torch.bfloat16,
                        enabled=use_autocast,
                    ):
                        if self.return_probs:
                            outputs, _, _ = self.model_(x_v)
                        else:
                            outputs, _ = self.model_(x_v)
                        val_loss += loss_fn(outputs, y_v).item()

                    _, pred = torch.max(outputs, 1)
                    val_pred.extend(pred.detach().cpu().numpy())
                    val_labels.extend(y_v.detach().cpu().numpy())

            infer_time = time() - infer_start
            val_loss /= len(loader_valid)
            val_bas = balanced_accuracy_score(val_labels, val_pred)

            logger.info(
                "Epoch %d/%d | BCE: %.4f | KAN: %.4f | SMoE: %.4f | "
                "Val Loss: %.4f | Val BA: %.4f | "
                "Train: %.1fs | Infer: %.1fs",
                epoch + 1,
                self.epochs,
                bce_loss,
                kan_loss,
                mh_moe_loss,
                val_loss,
                val_bas,
                train_time,
                infer_time,
            )

            self.loss_history_train_.append((bce_loss, kan_loss, mh_moe_loss))
            self.loss_history_valid_.append(val_loss)
            self.score_history_.append(val_bas)

        if save_path is not None:
            torch.save(self.model_.state_dict(), save_path)
            logger.info("Model weights saved to %s", save_path)

        return self

    def predict(
        self, X: np.ndarray, return_logits: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict class labels for samples in *X*.

        Parameters
        ----------
        X : np.ndarray
            Input features, same shape as used in :meth:`fit`.
        return_logits : bool, default=False

        Returns
        -------
        predictions : np.ndarray
            Integer class labels of shape ``(n_samples,)``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called yet.
        """
        self._check_is_fitted()
        self.model_.eval()

        loader = self._make_dataloader(torch.tensor(X, dtype=torch.float32))
        val_pred: list[int] = []
        val_logits: list[np.ndarray] = []

        with torch.no_grad():
            for (x_v,) in loader:
                x_v = x_v.to(self.device_)
                if self.return_probs:
                    outputs, _, _ = self.model_(x_v)
                else:
                    outputs, _ = self.model_(x_v)
                val_pred.extend(torch.max(outputs, 1)[1].detach().cpu().numpy())
                if return_logits:
                    val_logits.extend(outputs.detach().cpu().float().numpy())
        if return_logits:
            return np.asarray(val_pred), np.asarray(val_logits)
        return np.asarray(val_pred)

    def load_weights(self, path):
        """
        Loads model weights from the specified path.
        """
        if not hasattr(self, "model_"):
            logger.info("Initializing model...")
            self._init_model()

        state_dict = torch.load(path, map_location=self.device_, weights_only=True)
        self.model_.load_state_dict(state_dict)
        self.model_.eval()

    def get_model(self):
        """
        Returns the underlying PyTorch model (nn.Module).
        Ensures the model is initialized first.
        """
        if not hasattr(self, "model_"):
            logger.info ("Initializing model...")
            self._init_model()
        return self.model_

    def summary(self, col_names: Optional[Sequence[str]] = None, depth: int = 10):
        """Print and return the model structure and parameter count using torchinfo.

        Parameters
        ----------
        col_names : Sequence[str], optional
            Columns to display in the summary. Defaults to input size,
            output size, number of parameters, and multiply-accumulates.
        """
        # Ensure the underlying PyTorch model is initialized
        if not hasattr(self, "model_"):
            logger.info("Initializing model...")
            self._init_model()

        if col_names is None:
            col_names = ("input_size", "output_size", "num_params", "mult_adds")

        # Determine the input shape based on the classifier's configuration
        input_size = (self.batch_size, self.n_channels, self.res_L, self.seq_L)

        return summary(
            self.model_,
            input_size=input_size,
            col_names=list(col_names),
            device=self.device_,
            depth=depth,
        )

    def explain(
            self,
            X_explain: np.ndarray,
            background_data: Optional[np.ndarray] = None,
            explainer_type: str = "gradient",
            output_type: str = "logits",
            n_background_samples: int = 50,
            batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute SHAP values to explain WaveSeekerNet predictions.

        Parameters
        ----------
        X_explain : np.ndarray
            Input samples to explain. Shape: (n_samples, res_L, seq_L)
            or (n_samples, n_channels, res_L, seq_L).
        background_data : np.ndarray, optional
            Representative baseline data used to initialize the explainer.
            If None, a random subset of X_explain is used.
        explainer_type : str, default="gradient"
            The type of SHAP explainer to use:
            - "deep": uses DeepExplainer (DeepLIFT). WARNING: May fail with KAN/SMoE/FFT/Wavelets.
            - "gradient": uses GradientExplainer (Integrated Gradients). Recommended.
            - "kernel": model-agnostic KernelExplainer.
        output_type : str, default="logits"
            Explain "logits" (recommended) or "probs" (probabilities).
        n_background_samples : int, default=50
            Number of background samples to choose from if background_data is None.
        batch_size : int, optional
            Batch size used during SHAP evaluation to avoid CUDA Out-Of-Memory (OOM) issues.
            If None, defaults to `self.batch_size`.

        Returns
        -------
        shap_values : np.ndarray
            SHAP values of shape:
            - (n_samples, res_L, seq_L, num_model_output) for single-channel inputs
            - (n_samples, n_channels, res_L, seq_L, num_model_output) for multi-channel inputs
        """
        self._check_is_fitted()
        import shap  # Lazy import to avoid dependency issues if shap is not installed

        # Resolve background data
        if background_data is None:
            indices = np.random.choice(
                X_explain.shape[0],
                min(n_background_samples, X_explain.shape[0]),
                replace=False
            )
            background_data = X_explain[indices]

        # 1. PyTorch-based Explainers (Deep SHAP and Gradient SHAP)
        if explainer_type in ("deep", "gradient"):
            # Internal wrapper to return a single output tensor
            class ShapModelWrapper(torch.nn.Module):
                def __init__(self, model, out_type):
                    super().__init__()
                    self.model = model
                    self.out_type = out_type
                    self.model.eval()

                def forward(self, x):
                    outputs = self.model(x)
                    if self.model.return_probs:
                        logits, probs, _ = outputs
                        return probs if self.out_type == "probs" else logits
                    else:
                        logits, _ = outputs
                        if self.out_type == "probs":
                            return torch.softmax(logits, dim=-1)
                        return logits

            wrapper = ShapModelWrapper(self.model_, output_type)

            # Move background data to the model's device
            background_tensor = torch.tensor(background_data, dtype=torch.float32).to(self.device_)

            if explainer_type == "deep":
                explainer = shap.DeepExplainer(wrapper, background_tensor)
            else:
                # Forward batch_size to control internal combination batching on GPU
                explainer = shap.GradientExplainer(wrapper, background_tensor, batch_size=batch_size)

            # Explain in mini-batches to prevent GPU OOM
            shap_values_batches = []
            num_samples = X_explain.shape[0]

            for i in range(0, num_samples, batch_size):
                batch_data = X_explain[i: i + batch_size]
                # Move only the current batch to the GPU
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(self.device_)
                # Returns list of Tensors/Arrays or a single Tensor/Array
                batch_shap = explainer.shap_values(batch_tensor)
                shap_values_batches.append(batch_shap)

            # Merge batched SHAP outputs
            if isinstance(shap_values_batches[0], list):
                # Handle list of length 1 (e.g. [[batch_values_of_shape_B_R_S_C]])
                if len(shap_values_batches[0]) == 1:
                    inner_shape = (
                        shap_values_batches[0][0].shape
                        if hasattr(shap_values_batches[0][0], "shape")
                        else "unknown"
                    )
                    logger.info(
                        "Merging SHAP batches: Scenario C (list of length 1). "
                        "Inner batch shape: %s",
                        inner_shape
                    )
                    shap_values_batches = [batch[0] for batch in shap_values_batches]
                    if isinstance(shap_values_batches[0], torch.Tensor):
                        shap_vals = torch.cat(shap_values_batches, dim=0).cpu().numpy()
                    else:
                        shap_vals = np.concatenate(shap_values_batches, axis=0)
                else:
                    # Multi-class scenario: list of batch arrays per class (e.g. batch_shap has n_classes items)
                    n_classes = len(shap_values_batches[0])
                    class_shape = (
                        shap_values_batches[0][0].shape
                        if hasattr(shap_values_batches[0][0], "shape")
                        else "unknown"
                    )
                    logger.info(
                        "Merging SHAP batches: Scenario A (list of class arrays). "
                        "Classes detected: %d, Array shape per class: %s",
                        n_classes,
                        class_shape
                    )
                    shap_vals = []
                    for c in range(n_classes):
                        class_batches = [batch[c] for batch in shap_values_batches]
                        if isinstance(class_batches[0], torch.Tensor):
                            merged = torch.cat(class_batches, dim=0).cpu().numpy()
                        else:
                            merged = np.concatenate(class_batches, axis=0)
                        shap_vals.append(merged)
                    # Stack classes along the last dimension -> (n_samples, ..., num_model_output)
                    shap_vals = np.stack(shap_vals, axis=-1)
            else:
                # Single-output or already-joined multi-class scenario of shape [B, res_L, seq_L, n_classes]
                batch_shape = (
                    shap_values_batches[0].shape
                    if hasattr(shap_values_batches[0], "shape")
                    else "unknown"
                )
                logger.info(
                    "Merging SHAP batches: Scenario B (single array). "
                    "Batch shape: %s",
                    batch_shape
                )
                if isinstance(shap_values_batches[0], torch.Tensor):
                    shap_vals = torch.cat(shap_values_batches, dim=0).cpu().numpy()
                else:
                    shap_vals = np.concatenate(shap_values_batches, axis=0)

            return shap_vals

        # 2. Model-Agnostic Kernel Explainer
        elif explainer_type == "kernel":
            original_shape = X_explain.shape[1:]

            # Wrap prediction pipeline using mini-batch evaluation
            def predict_wrapper(X_flat):
                X_reshaped = X_flat.reshape(-1, *original_shape)
                self.model_.eval()

                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.tensor(X_reshaped, dtype=torch.float32)),
                    batch_size=batch_size,
                    shuffle=False,
                )

                preds = []
                with torch.no_grad():
                    for (x_v,) in loader:
                        x_v = x_v.to(self.device_)
                        if self.model_.return_probs:
                            logits, probs, _ = self.model_(x_v)
                            val = probs if output_type == "probs" else logits
                        else:
                            logits, _ = self.model_(x_v)
                            if output_type == "probs":
                                val = torch.softmax(logits, dim=-1)
                            else:
                                val = logits
                        preds.extend(val.detach().cpu().numpy())
                return np.array(preds)

            X_explain_flat = X_explain.reshape(X_explain.shape[0], -1)
            background_flat = background_data.reshape(background_data.shape[0], -1)

            explainer = shap.KernelExplainer(predict_wrapper, background_flat)
            shap_vals = explainer.shap_values(X_explain_flat)

            # Reshape SHAP values back to match the original feature shape
            if isinstance(shap_vals, list):
                if len(shap_vals) == 1:
                    shap_vals = shap_vals[0].reshape(X_explain.shape)
                else:
                    # Each list element has shape (n_samples, res_L, seq_L)
                    shap_vals = [sv.reshape(X_explain.shape) for sv in shap_vals]
                    # Stack along last axis to get (n_samples, res_L, seq_L, num_model_output)
                    shap_vals = np.stack(shap_vals, axis=-1)
            else:
                shap_vals = shap_vals.reshape(X_explain.shape)

            return shap_vals

        else:
            raise ValueError(f"Unknown explainer_type: {explainer_type}. Choose 'deep', 'gradient', or 'kernel'.")
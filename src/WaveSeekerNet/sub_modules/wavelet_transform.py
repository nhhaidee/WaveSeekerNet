"""
Wavelet-based token-mixing head for WaveSeekerNet.

Applies a 1-level Discrete Wavelet Transform (DWT) to each token in the
sequence, processes the approximation and detail coefficients independently,
then reconstructs the sequence via the Inverse DWT.

References
----------
Attention-like DWT: https://arxiv.org/abs/2105.03824
Shrinkage regularisation: https://openreview.net/pdf?id=EXHG-A3jlM
"""
from __future__ import annotations

from math import floor

import torch
import torch.nn as nn
from torch.nn import RMSNorm
from pywt import Wavelet
from torch.cuda import is_available as is_gpu_available
from pytorch_wavelets import DWTForward, DWTInverse

from WaveSeekerNet.sub_modules.lib.star_layer import StarLayer


class WaveNETHead(nn.Module):
    """Wavelet-transform token-mixing head.

    Projects embeddings into multi-head format, applies a 1-level DWT,
    processes the approximation and detail sub-bands separately with a
    :class:`~WaveSeekerNet.sub_modules.lib.star_layer.StarLayer`, then
    reconstructs via the inverse DWT.

    Parameters
    ----------
    wavelet_name : str
        Name of the wavelet filter (e.g. ``"sym4"``, ``"bior3.3"``).
    emb_dim : int
        Embedding dimension (must be divisible by 32).
    n_patches : int
        Sequence length (number of tokens/patches).
    activation : type
        Activation function class (instantiated internally).
    """

    def __init__(
        self,
        wavelet_name: str,
        emb_dim: int,
        n_patches: int,
        activation: type,
    ) -> None:
        super().__init__()

        self.gpu_available = is_gpu_available()
        self.wavelet = wavelet_name
        self.n_heads = emb_dim // 32

        dec_len = Wavelet(wavelet_name).dec_len
        self.w_in_d = floor((32 + dec_len - 1) / 2)
        self.w_in_p = floor((n_patches + dec_len - 1) / 2)

        self.init_project = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            activation(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        self.DWT = DWTForward(J=1, wave=wavelet_name)

        # Approximation-coefficient sub-network
        self.processing_approx = StarLayer(
            emb_in=self.w_in_d,
            emb_out=self.w_in_d,
            n_patches=self.w_in_p,
            activation=activation,
            dropout="2d",
        )
        self.approx_norm = RMSNorm(self.w_in_d, eps=1e-8)
        self.approx_out = nn.Sequential(
            nn.Linear(self.w_in_d, self.w_in_d * 2),
            activation(),
            nn.Linear(self.w_in_d * 2, self.w_in_d),
        )

        # Detail-coefficient sub-network
        self.processing_details = StarLayer(
            emb_in=self.w_in_d,
            emb_out=self.w_in_d,
            n_patches=self.w_in_p,
            activation=activation,
            dropout="2d",
        )
        self.details_norm = RMSNorm(self.w_in_d, eps=1e-8)
        self.details_out = nn.Linear(self.w_in_d, self.w_in_d)

        # Inverse DWT
        self.iDWT = DWTInverse(wave=wavelet_name)

        # Output projection
        self.merge_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            activation(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout1d(0.125),
        )

        self.odd_patch = n_patches % 2 > 0
        if self.odd_patch:
            self.project_patch = nn.Linear(n_patches + 1, n_patches)

    def _shrinkage(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-threshold uninformative wavelet coefficients.

        Applies ``x - arctan(x)`` and zeros values with
        ``|result| <= 0.01``.

        Parameters
        ----------
        x : torch.Tensor
            Wavelet coefficient tensor.

        Returns
        -------
        torch.Tensor
            Sparse coefficient tensor of the same shape.
        """
        x_out = x - torch.arctan(x)
        x_gate = torch.where(torch.abs(x_out) > 0.01, 1, 0)
        return x_out * x_gate

    def _process_approx(self, x_in: torch.Tensor) -> torch.Tensor:
        """Process approximation coefficients.

        Parameters
        ----------
        x_in : torch.Tensor
            Shape ``(B, H, w_in_p, w_in_d)``.
        """
        x = self.processing_approx(x_in) + x_in
        x = self.approx_norm(x)
        return self.approx_out(x) + x

    def _process_details(self, x_in: torch.Tensor) -> list[torch.Tensor]:
        """Process detail coefficients (all orientation sub-bands).

        Parameters
        ----------
        x_in : torch.Tensor
            Detail coefficients from DWT, shape
            ``(B, H, orientations, w_in_p, w_in_d)``.
        """
        x = x_in.view(
            x_in.shape[0],
            self.n_heads * x_in.shape[2],
            x_in.shape[3],
            self.w_in_d,
        )
        x = self.processing_details(x) + x
        x = self.details_norm(x)
        x = self.details_out(x) + x
        x = self._shrinkage(x)

        x = x.view(
            x.shape[0],
            self.n_heads,
            x.shape[1] // self.n_heads,
            x.shape[2],
            self.w_in_d,
        )
        return [x.float()]

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def _dwt_gpu(self, x: torch.Tensor) -> tuple:
        """DWT forward on GPU (cast to float32)."""
        return self.DWT(x)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def _idwt_gpu(
        self, x_a: torch.Tensor, x_d: list[torch.Tensor]
    ) -> torch.Tensor:
        """DWT inverse on GPU (cast to float32)."""
        return self.iDWT([x_a, x_d])

    @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.float32)
    def _dwt_cpu(self, x: torch.Tensor) -> tuple:
        """DWT forward on CPU (cast to float32)."""
        return self.DWT(x)

    @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.float32)
    def _idwt_cpu(
        self, x_a: torch.Tensor, x_d: list[torch.Tensor]
    ) -> torch.Tensor:
        """DWT inverse on CPU (cast to float32)."""
        return self.iDWT([x_a, x_d])

    def forward(self, x_p: torch.Tensor) -> torch.Tensor:
        """Apply the wavelet token-mixing transform.

        Parameters
        ----------
        x_p : torch.Tensor
            Shape ``(B, n_patches, emb_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, n_patches, emb_dim)``.
        """
        x = self.init_project(x_p)

        # Reshape to (B, H, n_patches, 32)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, 32)
        x = x.permute(0, 2, 1, 3).contiguous()

        # DWT
        if self.gpu_available:
            y_approx, y_details = self._dwt_gpu(x)
        else:
            y_approx, y_details = self._dwt_cpu(x)

        y_approx_proc = self._process_approx(y_approx)
        y_detail_proc = self._process_details(y_details[0])

        # Inverse DWT
        if self.gpu_available:
            x = self._idwt_gpu(y_approx_proc, y_detail_proc)
        else:
            x = self._idwt_cpu(y_approx_proc, y_detail_proc)

        # (B, H, n_patches, 32) → (B, n_patches, emb_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_heads * 32)

        x = self.merge_proj(x)

        if self.odd_patch:
            x = self.project_patch(x.transpose(-2, -1)).transpose(-2, -1)

        return x
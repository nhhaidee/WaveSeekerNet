import torch
import torch.nn as nn
from torch.nn import RMSNorm
from torch.cuda import is_available as is_gpu_available
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pytorch_optimizer import create_optimizer
from time import time
from torchinfo import summary

from sub_modules.wavelet_transform import WaveNETHead
from sub_modules.fourier_transform import FNETHead
from sub_modules.gmlp import gMLPBlock
from sub_modules.smoe import SparseMoE, WaveExpert
from sub_modules.classification_head import ClassificationHead

from sub_modules.lib.star_layer import StarLayer
from sub_modules.lib.noisy_linear_layer import NoisyFactorizedLinear
from sub_modules.lib.make_patches import MakePatches
from sub_modules.lib.pos_encoding import PositionalEncoding
from sub_modules.lib.global_pooling import GlobalExpectationPooling
from sub_modules.lib.activation import ErMish


class WaveSeekerBlock(nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_patches,
            wavelet_names,
            ffn_dropout,
            use_fft,
            use_wavelet,
            device,
            use_smoe,
            activation,
            use_gmlp
    ):
        super(WaveSeekerBlock, self).__init__()

        self.device = device

        self.use_fft = use_fft
        self.use_wavelet = use_wavelet
        self.use_gmlp = use_gmlp

        self.use_smoe = use_smoe

        out_dim = 0

        self.norm_1 = RMSNorm(embedding_dim, eps=1e-8)

        # Prepare Wavelet Heads
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

            # Add embedding_dim for every wavelet used
            out_dim += embedding_dim * len(wavelet_names)

        # Prepare FFT Head
        if self.use_fft:
            self.fft_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                FNETHead(embedding_dim, activation),
            )

            out_dim += embedding_dim  # Add for the FNET

        # Prepare gMLP Head
        if self.use_gmlp:
            self.gmlp_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                gMLPBlock(
                    embedding_dim=embedding_dim,
                    ffn_dropout=ffn_dropout,
                    n_patches=n_patches,
                    activation=activation
                ),
            )

            out_dim += embedding_dim  # Add for the gMLP

        # Merging of heads
        self.dropout = nn.Dropout1d(ffn_dropout)

        if use_smoe:
            self.star = StarLayer(out_dim, embedding_dim, n_patches, activation)
            self.proj_concat = SparseMoE(
                dim=embedding_dim,
                top_k=3,
                activation=activation,
                ffn_dropout=0.25,
                n_patches=n_patches
            )

        else:
            self.star = StarLayer(emb_in=out_dim,
                                  emb_out=embedding_dim,
                                  n_patches=n_patches,
                                  activation=activation)
            self.proj_concat = WaveExpert(
                in_embed=embedding_dim,
                ffn_dropout=ffn_dropout,
                activation=activation
            )

        self.norm_2 = RMSNorm(embedding_dim, eps=1e-8)

    def weight_init(self, w):
        if isinstance(w, NoisyFactorizedLinear):
            nn.init.kaiming_uniform_(w.weight)
            nn.init.zeros_(w.bias)

    def forward(self, inputs):

        x_c = []

        # Normalize Inputs
        x_n = self.norm_1(inputs)

        # Wavelet
        if self.use_wavelet:
            [x_c.append(wave_head(x_n)) for wave_head in self.wave_heads]

        # FFT
        if self.use_fft:
            x_c.append(self.fft_head(x_n))

        if self.use_gmlp:
            x_c.append(self.gmlp_head(x_n))

        # Merge heads
        x_c = torch.concat(x_c, dim=-1)
        x_c = self.star(x_c) + inputs
        x_c = self.norm_2(x_c)

        # Process along hidden dimension
        if self.use_smoe:
            x_smoe, z_loss = self.proj_concat(x_c)

        else:
            z_loss = 0.0
            x_smoe = self.proj_concat(x_c)

        x_c = x_smoe + x_c

        return self.dropout(x_c), z_loss


class WaveNetTorch(nn.Module):
    def __init__(
            self,
            seq_L,
            res_L,
            n_channels,
            patch_size,
            n_out,
            device,
            emb_dim,
            wavelet_names,
            wave_dropout,
            use_fft,
            use_wavelets,
            n_blocks,
            final_dropout,
            final_hidden_size,
            return_probs,
            use_kan,
            use_smoe,
            patch_mode,
            activation,
            use_gmlp
    ):
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
        self.wavelet_names = wavelet_names
        self.wave_dropout = wave_dropout
        self.use_fft = use_fft
        self.use_wavelets = use_wavelets
        self.n_blocks = n_blocks
        self.final_dropout = final_dropout
        self.final_hidden_size = final_hidden_size
        self.device = device
        self.activation = activation
        self.use_gmlp = use_gmlp

#        self.make_patches = nn.Sequential(
#            MakePatches(
#                patch_width=patch_size[1],
#                patch_height=patch_size[0],
#                emb_dim=self.emb_dim,
#                n_channel=self.n_channels,
#                patch_mode=patch_mode,
#            ),
#            #PositionalEncoding(self.emb_dim),
#            nn.Dropout1d(0.5),
#        )
        self.patch_dropout = nn.Dropout1d(0.5)
        
        self.make_patches = nn.ModuleList(
            [
                
                MakePatches(
                  patch_width=patch_size[1],
                  patch_height=patch_size[0],
                  emb_dim=self.emb_dim,
                  patch_mode=patch_mode
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

        # "Self-Attention"
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
                    use_gmlp=self.use_gmlp
                )
                for _ in range(self.n_blocks)
            ]
        )

        # Summarize "Self-Attention" for classification
        self.create_tokens = GlobalExpectationPooling()

        # Classification head
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

    def weight_init(self, w):
        if isinstance(w, NoisyFactorizedLinear) or isinstance(w, nn.Conv1d):
            nn.init.kaiming_uniform_(w.weight)
            nn.init.zeros_(w.bias)

    def forward(self, x):
        z_loss_sa = 0.0

        # Project each patch through a non-linearity and add positional encoding
        if len(x.shape) == 3:
            x_patch = self.make_patches[0](x.unsqueeze(1))
        else:
            x_patch_c = []
            for channel_i in range(self.n_channels):
                x_temp = x[:, channel_i:channel_i+1, :, :]
                x_patch_c.append(self.make_patches[channel_i](x_temp))
            x_patch = torch.concat(x_patch_c, dim=-2)
        
        x_patch = self.patch_dropout(x_patch)
        
        if self.patch_mode == "full" or self.patch_mode == "compress":
            x_patch = self.a_pool(x_patch.transpose(-1, -2)).transpose(-1, -2)
            x_patch = self.pool_pos(x_patch)

        # "Self-Attention" Encoder
        for i in range(self.n_blocks):
            x_patch, loss_sa = self.self_attention_enc[i](x_patch)
            z_loss_sa += loss_sa
        #for sa_enconder in self.self_attention_enc:
            #x_sa, loss_sa = sa_enconder(x_patch)
            #z_loss_sa += loss_sa

        # Create classification tokens
        cls_tokens = self.create_tokens(x_patch)

        # Classification output
        if self.return_probs:
            x_logit, x_out = self.classifier(cls_tokens)

            return x_logit, x_out, z_loss_sa

        else:
            return self.classifier(cls_tokens), z_loss_sa


class WaveSeekerClassifier:
    def __init__(
            self,
            seq_L,
            res_L,
            n_channels,
            patch_size,
            n_out,
            emb_dim=196,
            wavelet_names=["bior3.3", "sym4"],
            wave_dropout=0.5,
            use_fft=True,
            use_wavelets=True,
            n_blocks=2,
            final_dropout=0.5,
            final_hidden_size=32,
            batch_size=64,
            epochs=30,
            lr=1e-3,
            wd=0.0,
            optimizer_name="Adan",
            use_gc=True,
            use_lookahead=True,
            use_kan=True,
            use_smoe=True,
            patch_mode="compress",
            activation=ErMish,
            use_gmlp=True,
            return_probs=True
    ):
        self.use_kan = use_kan
        self.use_smoe = use_smoe
        self.patch_mode = patch_mode

        self.activation = activation

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.optimizer_name = optimizer_name
        self.use_gc = use_gc
        self.use_lookahead = use_lookahead

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
        self.use_gmlp = use_gmlp
        self.return_probs = return_probs

    def fit(self, X, y, X_va=None, y_va=None, save_path=None):
        # Get device (CPU or GPU)
        use_autocast = False
        if is_gpu_available():
            use_autocast = True
            device_type = "cuda:0"

        else:
            device_type = "cpu"

        self.device = torch.device("cuda:0" if is_gpu_available() else "cpu")

        print("Using: %s!" % self.device)

        self.loss_history_train = []
        self.loss_history_valid = []
        self.score_history = []

        # Prepare data
        if X_va is not None:
            X_train = torch.tensor(X)
            y_train = torch.tensor(y)

            X_valid = torch.tensor(X_va).to(self.device)
            y_valid = torch.tensor(y_va).to(self.device)

        else:
            X_train, X_va, y_train, y_va = train_test_split(
                X, y, test_size=0.10, stratify=y, random_state=0
            )

            X_train = torch.tensor(X_train)
            y_train = torch.tensor(y_train)

            X_valid = torch.tensor(X_va).to(self.device)
            y_valid = torch.tensor(y_va).to(self.device)

        # Create a dataset loader
        dataset_train = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)),
            shuffle=True,
            batch_size=self.batch_size,
        )
        dataset_valid = torch.utils.data.DataLoader(
            list(zip(X_valid, y_valid)), shuffle=False, batch_size=self.batch_size
        )

        # Put model on the appropriate device
        self.model = WaveNetTorch(
            seq_L=self.seq_L,
            res_L=self.res_L,
            n_channels=self.n_channels,
            device=self.device,
            patch_size=self.patch_size,
            n_out=self.n_out,
            emb_dim=self.emb_dim,
            wavelet_names=self.wavelet_names,
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
            use_gmlp=self.use_gmlp
        )

        self.model.to(self.device)

#        summary(self.model, input_size=( 2,  2, X_train.shape[2], X_train.shape[3]),
#                col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"], depth=10)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params_train = sum([np.prod(p.size()) for p in model_parameters])
        print("Total Trainable Parameters: ", params_train)

        model_parameters = self.model.parameters()
        params_total = sum([np.prod(p.size()) for p in model_parameters])
        print("Total Parameters: ", params_total)

        optimizer_1 = create_optimizer(
            self.model,
            wd_ban_list=[
                str(x[0])
                for x in list(self.model.named_parameters())
                if ("m_scaler" in x[0]) or ("rms_scaler" in x[0])
            ],
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.wd,
            use_lookahead=self.use_lookahead,
            use_gc=self.use_gc,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_1,
            max_lr=0.01,
            steps_per_epoch=len(dataset_train),
            epochs=self.epochs
        )

        loss_fn1 = torch.nn.CrossEntropyLoss().to(self.device)

        scaler_1 = torch.amp.GradScaler(self.device)

        # Training loop
        for epoch in range(self.epochs):
            i = 0
            start = time()

            if torch.cuda.is_available:
                torch.cuda.empty_cache()

            # Training steps
            self.model.train()

            bce_loss = 0.0
            kan_loss = 0.0
            mh_moe_loss = 0.0
            for batch_num, batch in enumerate(dataset_train):
                x_in, y_in = batch
                x_in = x_in.to(self.device)
                y_in = y_in.to(self.device)

                with torch.amp.autocast(
                        device_type=device_type,
                        dtype=torch.bfloat16,
                        enabled=use_autocast
                ):
                    if self.return_probs:
                    # Compute outputs
                      x_logit, sm_out, moe_loss = self.model(x_in)
                    else:
                      x_logit, moe_loss = self.model(x_in)

                    # Calculate loss - KAN
                    if self.use_kan:
                        k = self.model.classifier.logits._modules[
                            "0"
                        ].regularization_loss(1.0, 1.0)
                        k += self.model.classifier.logits._modules[
                            "1"
                        ].regularization_loss(1.0, 1.0)
                        k += self.model.classifier.logits._modules[
                            "2"
                        ].regularization_loss(1.0, 1.0)

                        k = k / 3.0
                        k = 0.01 * k

                    # Calculate loss - BCE
                    loss_1 = loss_fn1(x_logit, y_in)

                    # Combine Losses
                    if self.use_kan:
                        total_loss = loss_1 + k

                    else:
                        total_loss = loss_1

                    if self.use_smoe:
                        sa_moe = moe_loss * 0.1
                        total_loss += sa_moe

                bce_loss += loss_1.item()

                if self.use_kan:
                    kan_loss += k.item()

                if self.use_smoe:
                    mh_moe_loss += sa_moe.item()

                # Backwards pass
                optimizer_1.zero_grad()
                scaler_1.scale(total_loss).backward()

                # Update weights
                scaler_1.step(optimizer_1)
                scaler_1.update()
                scheduler.step()

            end = time()

            total_time = end - start

            bce_loss /= len(dataset_train)
            kan_loss /= len(dataset_train)
            mh_moe_loss /= len(dataset_train)

            # Validation steps
            start = time()

            self.model.eval()

            val_loss = 0.0

            val_pred = []
            val_labels = []

            with torch.no_grad():
                for batch in dataset_valid:
                    x_v, y_v = batch

                    with torch.amp.autocast(
                            device_type=device_type,
                            dtype=torch.bfloat16,
                            enabled=use_autocast
                    ):
                        if self.return_probs:
                          outputs, sm_out, _ = self.model(x_v)
                        else:
                          outputs, _ = self.model(x_v)

                        loss_v = loss_fn1(outputs, y_v)

                    val_loss += loss_v.item()

                    _, pred = torch.max(outputs, 1)

                    val_pred.extend(pred.cpu().numpy())
                    val_labels.extend(y_v.cpu().numpy())

            end = time()

            inference_time = end - start

            val_loss /= len(dataset_valid)
            val_bas = balanced_accuracy_score(val_labels, val_pred)

            print(
                f"Epoch {epoch + 1}: BCE Loss: {bce_loss:.4f}"
            )
            print(
                f"KAN Loss: {kan_loss:.4f}, MH-SMoE (SA) Loss: {mh_moe_loss:.4f}"
            )

            self.loss_history_train.append((bce_loss, kan_loss, mh_moe_loss))

            print(
                f"Val Loss: {val_loss:.4f} Val BA-Score: {val_bas: 4f} Training Time: {total_time: 4f} Inference Time: {inference_time: 4f}"
            )

            self.loss_history_valid.append(val_loss)
            self.score_history.append(val_bas)
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
        return self

    def predict(self, X):
        self.model.eval()

        val_pred = []

        X_unk = torch.tensor(X, device=self.device)

        y_nothing = torch.tensor(np.zeros(shape=(X.shape[0],), dtype=int)).to(
            self.device
        )

        dataset_valid = torch.utils.data.DataLoader(
            list(zip(X_unk, y_nothing)), shuffle=False, batch_size=self.batch_size
        )

        with torch.no_grad():
            for batch in dataset_valid:
                x_v, _ = batch
                if self.return_probs:
                  outputs = self.model(x_v)[-3]
                else:
                  outputs = self.model(x_v)[-2]

                pred = torch.max(outputs, 1)[1]

                val_pred.extend(pred.cpu().numpy())

        return np.asarray(val_pred)

    def predict_proba(self, X):
        self.model.eval()

        val_pred = []

        X_unk = torch.tensor(X, device=self.device)

        y_nothing = torch.tensor(np.zeros(shape=(X.shape[0],), dtype=int)).to(
            self.device
        )

        dataset_valid = torch.utils.data.DataLoader(
            list(zip(X_unk, y_nothing)), shuffle=False, batch_size=self.batch_size
        )

        criterion = torch.nn.Softmax(dim=-1)
        with torch.no_grad():
            for batch in dataset_valid:
                x_v, _ = batch

                outputs = self.model(x_v)[-2]

                val_pred.extend(outputs.cpu().numpy())

        return np.asarray(val_pred)

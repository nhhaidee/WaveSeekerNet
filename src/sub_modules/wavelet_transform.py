import torch
import torch.nn as nn
from torch.nn import RMSNorm
from pywt import Wavelet
from math import floor
from torch.cuda import is_available as is_gpu_available
from pytorch_wavelets import DWTForward, DWTInverse
from sub_modules.lib.star_layer import StarLayer

"""
"Attention"-Like Mechanisms for "Self-Attention"

FNet: https://arxiv.org/abs/2105.03824
FFT Processing: https://openreview.net/pdf?id=EXHG-A3jlM
Inspiriation for shrinkage comes from above.

gMLP: https://arxiv.org/abs/2105.08050
Efficient Attention: https://arxiv.org/abs/1812.01243
"""

class WaveNETHead(nn.Module):
    def __init__(self, wavelet_name, emb_dim, n_patches, activation):
        super(WaveNETHead, self).__init__()

        self.gpu_available = is_gpu_available()

        self.wavelet = wavelet_name

        self.n_heads = emb_dim // 32
        self.w_in_d = floor((32 + Wavelet(wavelet_name).dec_len - 1) / 2)
        self.w_in_p = floor((n_patches + Wavelet(wavelet_name).dec_len - 1) / 2)


        self.init_project = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            activation(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        self.DWT = DWTForward(J=1, wave=wavelet_name)

        # Processing - Approximation
        self.processing_approx = StarLayer(
            emb_in=self.w_in_d,
            emb_out=self.w_in_d,
            n_patches=self.w_in_p,
            activation=activation,
            dropout="2d"
        )

        self.approx_norm = RMSNorm(self.w_in_d, eps=1e-8)

        self.approx_out = nn.Sequential(
            nn.Linear(self.w_in_d, self.w_in_d * 2),
            activation(),
            nn.Linear(self.w_in_d * 2, self.w_in_d),
        )

        # Processing - Detail Coefficients
        self.processing_details = StarLayer(emb_in=self.w_in_d,
                                            emb_out=self.w_in_d,
                                            n_patches=self.w_in_p,
                                            activation=activation,
                                            dropout="2d")

        self.details_norm = RMSNorm(self.w_in_d, eps=1e-8)
        self.details_out = nn.Linear(self.w_in_d, self.w_in_d)

        # Inverse transform
        self.iDWT = DWTInverse(wave=wavelet_name)

        # Merge heads
        self.merge_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            activation(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout1d(0.125)
        )
        self.odd_patch = False
        if n_patches % 2 > 0:
            self.odd_patch = True
            self.project_patch = nn.Linear(n_patches + 1, n_patches)

    def shrinkage(self, x):

        # Shrink uninfomartive frequencies
        x_out = x - torch.arctan(x)

        # Identify all those with value |0.01| or smaller
        x_gate = torch.abs(x_out) - 0.01
        x_gate = torch.where(x_gate > 0, 1, 0)

        return x_out * x_gate

    def process_approx(self, x_in):

        x = self.processing_approx(x_in) + x_in
        x = self.approx_norm(x)
        x = self.approx_out(x) + x

        return x

    def process_details(self, x_in):

        x = x_in.view(
            [
                x_in.shape[0],
                self.n_heads * x_in.shape[2],
                x_in.shape[3],
                self.w_in_d
            ]
        )

        x = self.processing_details(x) + x
        x = self.details_norm(x)
        x = self.details_out(x) + x
        x = self.shrinkage(x)

        x = x_in.view(
            [
                x_in.shape[0],
                self.n_heads,
                x_in.shape[2],
                x_in.shape[3],
                self.w_in_d
            ]
        )

        return [x.float()]

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def dwt_gpu(self, x):
        return self.DWT(x)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def idwt_gpu(self, x_a, x_d):
        return self.iDWT([x_a, x_d])

    @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.float32)
    def dwt_cpu(self, x):
        return self.DWT(x)

    @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.float32)
    def idwt_cpu(self, x_a, x_d):
        return self.iDWT([x_a, x_d])

    def forward(self, x_p):

        x = self.init_project(x_p)

        x = x.view(
            [
                x.shape[0],
                x.shape[1],
                self.n_heads,
                32,
            ]
        )

        x = x.permute(
            0, 2, 1, 3
        ).contiguous()  # (B, L, D) - > (B, L, H, 32) -> (B, H, L, 32)

        # Conduct DWT
        if self.gpu_available:
            y_approx, y_details = self.dwt_gpu(x)

        else:
            y_approx, y_details = self.dwt_cpu(x)

        # Processing of the Approximation Coefficients
        y_approx_proc = self.process_approx(y_approx)

        # Processing of the Detail Coefficients
        y_detail_proc = self.process_details(y_details[0])

        # Reconstruction (Reconstructs the original using the selected coefficients)
        if self.gpu_available:
            x = self.idwt_gpu(y_approx_proc, y_detail_proc)

        else:
            x = self.idwt_cpu(y_approx_proc, y_detail_proc)
        # (B, H, L, 32) - > (B, L, H, 32) -> (B, L, D)
        x = x.permute(
            0, 2, 1, 3
        ).contiguous()

        x = x.view(x.shape[0],
                   x.shape[1],
                   self.n_heads * 32)

        # Merge heads
        x = self.merge_proj(x)

        if self.odd_patch:
            x = torch.transpose(x, -2, -1)
            x = self.project_patch(x)
            x = torch.transpose(x, -2, -1)

        return x
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from einops import rearrange, repeat

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math
from .mamba_custom import MambaCustom as Mamba


class R2Plus1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.2):
        super(R2Plus1DBlock, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.spatial_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.temporal_conv = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            groups=out_channels,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.last = nn.Conv3d(
            out_channels, 2 * out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        out = self.spatial_conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.temporal_conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.last(out)
        return out
    
class SSM(nn.Module):
    def __init__(self,
                 dim=None,
                 dt_rank = None,
                 d_inner = None,
                 d_state = 16,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        bias = False
        # in proj ============================
        self.in_proj = nn.Linear(dim, d_inner*2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias = True, kernel_size=3,padding=1, ** kwargs,
        )
        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        mixer_cls = partial(Mamba,
                        d_state=16,
                        expand=2,
                        layer_idx=None,
                        # **ssm_cfg,
                        # **factory_kwargs
                           )
        self.mixer = mixer_cls(d_inner)

    def forward(self, x: Tensor):
        x = self.in_proj(x) 
        x, z = x.chunk(2, dim=-1)  
        x_shape = x.shape
        z = self.act(z) 
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv3d(x) #[64, 256, 28, 7, 7]
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([x, x.flip(1)], dim=0)  # [2*B, L, D]
        # 一次性处理
        x = self.mixer(x, inference_params=None)  # [2*B, L, D]
        # 分开正向和反向结果
        x_forward, x_reverse = torch.chunk(x, 2, dim=0)  # 各 [B, L, D]
        x = x_forward + x_reverse.flip(1) 
        y = x.view(x_shape)
        y = y * z   
        out = self.dropout(self.out_proj(y))  
        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
class Block(nn.Module):
    def __init__(self,
                 scan_type=None,
                 group_type = None,
                 k_group = None,
                 dim=None,
                 dt_rank = None,
                 d_inner = None,
                 d_state = None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 drop_path_rate=0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.ssm = SSM(dim=dim, d_inner=d_inner, dropout=dropout, **kwargs)
        self.mlp = MLP(in_features=dim, hidden_features=dim*4, act_layer=nn.GELU, drop=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: Tensor):
        x = x + self.drop_path(self.ssm(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VisionMamba(nn.Module):
    def __init__(
            self,
            group_type=None,
            k_group=None,
            depth=None,
            embed_dim=None,
            dt_rank: int = None,
            d_inner: int = None,
            d_state: int = None,
            num_classes: int = None,
            drop_rate=0.,
            drop_path_rate=0.1,
            fused_add_norm=False,
            residual_in_fp32=True,
            bimamba=True,
            # video
            fc_drop_rate=0.,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            Pos_Cls = False,
            scan_type=None,
            pos: str = None,
            cls: str = None,
            conv3D_channel: int = None,
            conv3D_kernel: int = None,
            dim_patch: int = None,
            dim_linear: int = None,
            **kwargs,
        ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.Pos_Cls = Pos_Cls
        self.scan_type = scan_type
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.k_group = k_group
        self.group_type = group_type
        self.conv3d_new = R2Plus1DBlock(in_channels=1, out_channels=conv3D_channel, kernel_size=conv3D_kernel)
        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.layers = nn.ModuleList(
            [
                Block(
                    scan_type=scan_type,
                    group_type=group_type,
                    k_group=k_group,
                    dim=embed_dim,
                    d_state=d_state,
                    d_inner=d_inner,
                    dt_rank=dt_rank,
                    bimamba=bimamba,
                    drop_path_rate=drop_path_rate,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )


    def forward_features(self, x, inference_params=None):
        x = self.conv3d_new(x)
        x = rearrange(x, 'b c t h w -> b t h w c')  # [64, 28, 8, 8, 64]
        for idx, layer in enumerate(self.layers):  
            x = layer(x)
        return self.flatten(self.avgpool(x.permute(0, 4, 1, 2, 3)).mean(dim=2)) 

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x, inference_params)  
        x = self.head(self.head_drop(feature))   
        return x, feature

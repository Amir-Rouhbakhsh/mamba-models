#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Simplified standalone version for custom classification.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp

# ----------------------------------------------------------------------
# Helper functions for window partitioning (needed for transformer blocks)
# ----------------------------------------------------------------------
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x

# ----------------------------------------------------------------------
# Downsample & PatchEmbed (same as original)
# ----------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)

    def forward(self, x):
        return self.reduction(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        return self.conv_down(x)

# ----------------------------------------------------------------------
# ConvBlock (used in first two stages)
# ----------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = False
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        return input + self.drop_path(x)

# ----------------------------------------------------------------------
# MambaVisionMixer (SSM block)
# ----------------------------------------------------------------------
class MambaVisionMixer(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=3, expand=1, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner//2).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(self.d_inner//2, self.d_inner//2, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner//2, **factory_kwargs)
        self.conv1d_z = nn.Conv1d(self.d_inner//2, self.d_inner//2, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner//2, **factory_kwargs)

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None,
                              delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

# ----------------------------------------------------------------------
# Attention block (ViT)
# ----------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

# ----------------------------------------------------------------------
# Single Block (either Mamba or Attention)
# ----------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Mlp_block=Mlp, layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_norm=qk_scale, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

# ----------------------------------------------------------------------
# MambaVision Layer (one stage)
# ----------------------------------------------------------------------
class MambaVisionLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, conv=False, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., layer_scale=None, layer_scale_conv=None, transformer_blocks=None):
        super().__init__()
        self.conv = conv
        self.transformer_block = not conv
        self.window_size = window_size
        if transformer_blocks is None:
            transformer_blocks = []
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layer_scale=layer_scale_conv) for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(dim, num_heads, i, transformer_blocks, mlp_ratio, qkv_bias, qk_scale,
                      drop, attn_drop, drop_path[i] if isinstance(drop_path, list) else drop_path,
                      layer_scale=layer_scale) for i in range(depth)
            ])
        self.downsample = None if not downsample else Downsample(dim)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
                Hp, Wp = x.shape[2], x.shape[3]
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
        for blk in self.blocks:
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)

# ----------------------------------------------------------------------
# Full MambaVision Model
# ----------------------------------------------------------------------
class MambaVision(nn.Module):
    def __init__(self, dim, in_dim, depths, window_size, mlp_ratio, num_heads,
                 drop_path_rate=0.2, in_chans=3, num_classes=1000,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 layer_scale=None, layer_scale_conv=None):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = (i == 0 or i == 1)
            level = MambaVisionLayer(
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                downsample=(i < 3),
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i]))
            )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)

    def load_pretrained(self, url_or_path, strict=False):
        """Load pretrained weights from URL or local file."""
        if not Path(url_or_path).is_file():
            torch.hub.download_url_to_file(url_or_path, dst="/tmp/mambavision_temp.pth")
            checkpoint = torch.load("/tmp/mambavision_temp.pth", map_location="cpu")
        else:
            checkpoint = torch.load(url_or_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        # Remove "module." prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Remove head weights if number of classes differs
        if self.num_classes != 1000:
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
        self.load_state_dict(state_dict, strict=strict)

# ----------------------------------------------------------------------
# Factory function to create model with custom number of classes
# ----------------------------------------------------------------------
def create_mambavision(model_name, num_classes, pretrained=True, **kwargs):
    """
    Create a MambaVision model with a custom number of output classes.

    Args:
        model_name (str): one of 'T', 'T2', 'S', 'B', 'B_21k', 'L', 'L_21k', 'L2', 'L2_512_21k', 'L3_256_21k', 'L3_512_21k'
        num_classes (int): number of target classes
        pretrained (bool): whether to load ImageNet pretrained weights (ignores mismatched head)
        **kwargs: additional arguments to override model config (e.g. drop_path_rate)
    Returns:
        MambaVision model
    """
    # Model configurations (from original code)
    configs = {
        'T':      {'depths': [1,3,8,4],  'num_heads': [2,4,8,16], 'window_size': [8,8,14,7], 'dim':80,  'in_dim':32,  'drop_path_rate':0.2, 'layer_scale':None},
        'T2':     {'depths': [1,3,11,4], 'num_heads': [2,4,8,16], 'window_size': [8,8,14,7], 'dim':80,  'in_dim':32,  'drop_path_rate':0.2, 'layer_scale':None},
        'S':      {'depths': [3,3,7,5],  'num_heads': [2,4,8,16], 'window_size': [8,8,14,7], 'dim':96,  'in_dim':64,  'drop_path_rate':0.2, 'layer_scale':None},
        'B':      {'depths': [3,3,10,5], 'num_heads': [2,4,8,16], 'window_size': [8,8,14,7], 'dim':128, 'in_dim':64,  'drop_path_rate':0.3, 'layer_scale':1e-5},
        'B_21k':  {'depths': [3,3,10,5], 'num_heads': [2,4,8,16], 'window_size': [8,8,14,7], 'dim':128, 'in_dim':64,  'drop_path_rate':0.3, 'layer_scale':1e-5},
        'L':      {'depths': [3,3,10,5], 'num_heads': [4,8,16,32], 'window_size': [8,8,14,7], 'dim':196, 'in_dim':64,  'drop_path_rate':0.3, 'layer_scale':1e-5},
        'L_21k':  {'depths': [3,3,10,5], 'num_heads': [4,8,16,32], 'window_size': [8,8,14,7], 'dim':196, 'in_dim':64,  'drop_path_rate':0.3, 'layer_scale':1e-5},
        'L2':     {'depths': [3,3,12,5], 'num_heads': [4,8,16,32], 'window_size': [8,8,14,7], 'dim':196, 'in_dim':64,  'drop_path_rate':0.3, 'layer_scale':1e-5},
        'L2_512_21k':   {'depths': [3,3,12,5], 'num_heads': [4,8,16,32], 'window_size': [8,8,32,16], 'dim':196, 'in_dim':64, 'drop_path_rate':0.3, 'layer_scale':1e-5},
        'L3_256_21k':   {'depths': [3,3,20,10], 'num_heads': [4,8,16,32], 'window_size': [8,8,16,8],  'dim':256, 'in_dim':64, 'drop_path_rate':0.5, 'layer_scale':1e-5},
        'L3_512_21k':   {'depths': [3,3,20,10], 'num_heads': [4,8,16,32], 'window_size': [8,8,32,16], 'dim':256, 'in_dim':64, 'drop_path_rate':0.5, 'layer_scale':1e-5},
    }
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(configs.keys())}")
    cfg = configs[model_name].copy()
    cfg.update(kwargs)  # allow user overrides
    model = MambaVision(
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        window_size=cfg['window_size'],
        dim=cfg['dim'],
        in_dim=cfg['in_dim'],
        mlp_ratio=4,
        drop_path_rate=cfg['drop_path_rate'],
        num_classes=num_classes,
        layer_scale=cfg.get('layer_scale', None),
        layer_scale_conv=None,
        qkv_bias=True,
        qk_scale=False,
        drop_rate=0.,
        attn_drop_rate=0.
    )
    if pretrained:
        # Pretrained URLs from original code
        urls = {
            'T': 'https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
            'T2': 'https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
            'S': 'https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
            'B': 'https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
            'B_21k': 'https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar',
            'L': 'https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
            'L_21k': 'https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar',
            'L2': 'https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
            'L2_512_21k': 'https://huggingface.co/nvidia/MambaVision-L2-512-21K/resolve/main/mambavision_L2_21k_240m_512.pth.tar',
            'L3_256_21k': 'https://huggingface.co/nvidia/MambaVision-L3-256-21K/resolve/main/mambavision_L3_21k_740m_256.pth.tar',
            'L3_512_21k': 'https://huggingface.co/nvidia/MambaVision-L3-512-21K/resolve/main/mambavision_L3_21k_740m_512.pth.tar',
        }
        model.load_pretrained(urls[model_name], strict=False)
    return model
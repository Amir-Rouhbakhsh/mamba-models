"""
RMSNet: Residual Mamba Segmentation Network
============================================
Modifications over VM-UNet:
  1. SC-Attention Bridge (SCAB) on skip connections  [SCM-UNet]
  2. Residual Blocks in the decoder after each upsampling stage  [Lung-Mamba]

Shape flow (VMamba-T defaults, dims = [96,192,384,768]):
  Encoder skip_list (collected before each layer):
    skip_list[0]: (B, H/4,  W/4,  96)
    skip_list[1]: (B, H/8,  W/8,  192)
    skip_list[2]: (B, H/16, W/16, 384)
    skip_list[3]: (B, H/32, W/32, 768)

  Decoder layers_up (upsample happens INSIDE layer_up, before VSSBlocks):
    inx=0: bottleneck  768->768  (no skip, no upsample)
    inx=1: skip[-1]=768, add then upsample->384, VSSBlock->384, ResBlock->384
    inx=2: skip[-2]=384, add then upsample->192, VSSBlock->192, ResBlock->192
    inx=3: skip[-3]=192, add then upsample->96,  VSSBlock->96,  ResBlock->96

  SCAB bridge dims (on skip before add): [768, 384, 192]
  ResBlock dims   (after layer_up out):  [384, 192, 96]
"""

import torch
import torch.nn as nn
from .vmamba import VSSM


# ──────────────────────────────────────────────────────────────────
# 1. Residual Block
# ──────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    Standard two-layer conv residual block (Conv-BN-ReLU × 2 + shortcut).
    Operates on tensors shaped (B, C, H, W).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ──────────────────────────────────────────────────────────────────
# 2. Spatial Attention Bridge (SAB)
# ──────────────────────────────────────────────────────────────────
class SpatialAttentionBridge(nn.Module):
    """
    Eq. (9)-(12) in SCM-UNet:
      avg_pool + max_pool along channel dim  ->  concat  ->
      dilated 7×7 conv  ->  sigmoid  ->  element-wise scale.
    """
    def __init__(self):
        super().__init__()
        # kernel=7, dilation=3, padding=9 keeps spatial size unchanged
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, dilation=3, padding=9, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)           # (B,1,H,W)
        mx  = x.max(dim=1, keepdim=True).values     # (B,1,H,W)
        Ms  = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B,1,H,W)
        return x * Ms


# ──────────────────────────────────────────────────────────────────
# 3. Channel Attention Bridge (CAB)
# ──────────────────────────────────────────────────────────────────
class ChannelAttentionBridge(nn.Module):
    """
    Eq. (13)-(16) in SCM-UNet (single-feature variant):
      GAP  ->  FC (C->C//r->C)  ->  sigmoid  ->  residual add.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),   # <-- fixed: mid->channels
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        z  = self.gap(x).view(B, C)           # (B, C)
        ai = self.fc(z).view(B, C, 1, 1)      # (B, C, 1, 1)
        return x + ai * x                     # F'_i = F_i + a_i ⊙ F_i


# ──────────────────────────────────────────────────────────────────
# 4. SC-Attention Bridge (SCAB)
# ──────────────────────────────────────────────────────────────────
class SCAttentionBridge(nn.Module):
    """SAB followed by CAB in series, placed on each skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.sab = SpatialAttentionBridge()
        self.cab = ChannelAttentionBridge(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(self.sab(x))


# ──────────────────────────────────────────────────────────────────
# 5. RMSNet
# ──────────────────────────────────────────────────────────────────
class RMSNet(nn.Module):
    """
    Residual Mamba Segmentation Network.

    Encoder  : unchanged VSSM (VMamba backbone).
    Decoder  : for each skip stage —
                 SCAB(skip)  ->  add to decoder feat  ->
                 layer_up (upsample + VSSBlocks)  ->
                 ResidualBlock
    """

    # VMamba-T channel dims per encoder stage
    _ENC_DIMS = [96, 192, 384, 768]

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        depths: list = None,
        depths_decoder: list = None,
        drop_path_rate: float = 0.2,
        load_ckpt_path: str = None,
    ):
        super().__init__()
        depths          = depths         or [2, 2, 9, 2]
        depths_decoder  = depths_decoder or [2, 9, 2, 2]

        self.load_ckpt_path = load_ckpt_path
        self.num_classes    = num_classes

        # ── Backbone ──────────────────────────────────────────────────────
        self.vmunet = VSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

        # ── SCAB modules on skip connections ──────────────────────────────
        # skips used at inx=1,2,3 have dims 768, 384, 192
        scab_dims = [self._ENC_DIMS[3], self._ENC_DIMS[2], self._ENC_DIMS[1]]  # [768,384,192]
        self.scab_bridges = nn.ModuleList([SCAttentionBridge(d) for d in scab_dims])

        # ── ResidualBlocks after each decoder stage ────────────────────────
        # outputs of layers_up at inx=1,2,3 have dims 384, 192, 96
        res_dims = [self._ENC_DIMS[2], self._ENC_DIMS[1], self._ENC_DIMS[0]]   # [384,192,96]
        self.res_blocks = nn.ModuleList([ResidualBlock(d) for d in res_dims])

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # ── Encoder ───────────────────────────────────────────────────────
        x_enc, skip_list = self.vmunet.forward_features(x)
        # skip_list: [96-dim, 192-dim, 384-dim, 768-dim]  (B,H,W,C) tensors

        # ── Decoder ───────────────────────────────────────────────────────
        x_dec = x_enc
        for inx, layer_up in enumerate(self.vmunet.layers_up):

            if inx == 0:
                # Bottleneck: no skip connection, no residual block
                x_dec = layer_up(x_dec)

            else:
                # bridge_idx 0,1,2  for inx 1,2,3
                bridge_idx = inx - 1
                skip = skip_list[-(inx)]   # 768, 384, 192 dim for inx=1,2,3

                # Apply SCAB: convert (B,H,W,C) <-> (B,C,H,W) for conv layers
                skip_2d = skip.permute(0, 3, 1, 2).contiguous()   # (B,C,H,W)
                skip_2d = self.scab_bridges[bridge_idx](skip_2d)
                skip    = skip_2d.permute(0, 2, 3, 1).contiguous() # (B,H,W,C)

                # Add refined skip, then upsample + VSSBlocks
                x_dec = layer_up(x_dec + skip)

                # Apply ResidualBlock  (dims: 384, 192, 96)
                B, H, W, C = x_dec.shape
                x_2d  = x_dec.permute(0, 3, 1, 2).contiguous()    # (B,C,H,W)
                x_2d  = self.res_blocks[bridge_idx](x_2d)
                x_dec = x_2d.permute(0, 2, 3, 1).contiguous()      # (B,H,W,C)

        # ── Final upsampling + classification head ─────────────────────────
        logits = self.vmunet.forward_final(x_dec)

        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return logits

    # ── Pretrained weight loading ──────────────────────────────────────────
    def load_from(self):
        if self.load_ckpt_path is None:
            return

        ckpt = torch.load(self.load_ckpt_path, map_location='cpu')
        pretrained = ckpt['model']

        # Encoder weights
        model_dict = self.vmunet.state_dict()
        enc_matched = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(enc_matched)
        self.vmunet.load_state_dict(model_dict)
        print(f"Encoder: loaded {len(enc_matched)}/{len(pretrained)} keys")

        # Decoder weights (mirror encoder layers)
        mirror = {'layers.0': 'layers_up.3', 'layers.1': 'layers_up.2',
                  'layers.2': 'layers_up.1', 'layers.3': 'layers_up.0'}
        dec_pretrained = {}
        for k, v in pretrained.items():
            for src, dst in mirror.items():
                if src in k:
                    dec_pretrained[k.replace(src, dst)] = v
                    break

        model_dict = self.vmunet.state_dict()
        dec_matched = {k: v for k, v in dec_pretrained.items() if k in model_dict}
        model_dict.update(dec_matched)
        self.vmunet.load_state_dict(model_dict)
        print(f"Decoder: loaded {len(dec_matched)}/{len(dec_pretrained)} keys")
        print("RMSNet checkpoint loaded.")

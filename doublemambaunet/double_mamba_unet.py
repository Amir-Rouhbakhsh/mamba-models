"""
DoubleMambaUNet
================

A "Double U-Net" (Jha et al., 2020) rebuilt with Mamba (VSSM / VMamba) encoders
and decoders instead of VGG-19 / plain-CNN blocks, following the same overall
topology used in RMSNet (Residual Mamba Segmentation Network).

Topology (mirrors the DoubleU-Net block diagram exactly, network-for-network):

    INPUT ---------------------------------------------------------------+
      |                                                                  |
      v                                                                  |
  MambaEncoder1 --skip1[0..2]--> (kept for BOTH decoders)                |
      |                                                                  |
      v                                                                  |
     ASPP1                                                               |
      |                                                                  |
      v                                                                  |
  MambaDecoder1 (skip1 only, SCAB + ResidualBlock per stage)             |
      |                                                                  |
      v                                                                  |
   OUTPUT 1 (sigmoid mask)                                               |
      |                                                                  |
      v                                                                  |
  MULTIPLY  <----------------------------------------------------------- +
      |
      v
  MambaEncoder2 --skip2[0..2]-->
      |
      v
     ASPP2
      |
      v
  MambaDecoder2 (skip1 AND skip2, fused per stage, SCAB + ResidualBlock)
      |
      v
   OUTPUT 2 (sigmoid mask)
      |
      v
  CONCATENATE(OUTPUT1, OUTPUT2)  -->  final 2-channel output

Design notes / how this maps onto DoubleU-Net's paper description
-------------------------------------------------------------------
- Network 1 == DoubleU-Net's first U-Net (there: VGG-19 encoder). Here we
  can't use an ImageNet-pretrained VGG (no Mamba equivalent exists), so
  MambaEncoder1 is a VSSM encoder trained from scratch -- exactly like
  MambaEncoder2, just with independent weights.
- Network 2 == DoubleU-Net's second U-Net, encoder built from scratch. Same
  here: MambaEncoder2 is a second, independently-initialized VSSM encoder.
- The paper's SE blocks -> replaced by RMSNet's SC-Attention Bridge (SAB+CAB)
  on every skip connection, as in your rmsnet.py.
- The paper's plain conv decoder stage -> replaced by VSSLayer_up (Mamba
  blocks) followed by a ResidualBlock, as in your rmsnet.py.
- The paper's ASPP -> re-implemented as a small conv-based ASPP that operates
  on the (permuted-to-channel-first) bottleneck feature map, run once after
  each encoder, exactly where the diagram places it.
- decoder1 uses ONLY skip1 (skip connections from encoder1) -- matches
  "In the first decoder, we only use skip connection from the first encoder."
- decoder2 uses BOTH skip1 and skip2 at every stage, fused via
  concat + linear projection back to the stage's native channel width before
  being added into the Mamba decoder stream -- matches "in the second
  decoder, we use skip connection from both the encoders."
- MULTIPLY: gated_input = input_image * output1, exactly as in DoubleU-Net.
- Final output is the channel-concatenation of output1 and output2 (2
  channels), exactly as in the diagram. For training you will typically
  supervise both channels against the same ground-truth mask (deep
  supervision) -- see the loss note at the bottom of this file.

A note on a bug fixed relative to your rmsnet.py
--------------------------------------------------
In rmsnet.py, `self.scab_bridges` is built over `ENCODER_DIMS[:3]` (96, 192,
384) but is indexed with `bridge_idx = inx - 1` (0, 1, 2) against skip
tensors pulled from `skip_list[skip_idx]` where `skip_idx = len(skip_list) -
inx` (3, 2, 1) -- i.e. skip channel widths (768, 384, 192), NOT (96, 192,
384). That mismatch would raise a shape error the first time SCAB's Linear
layer runs. Below, `SCAttentionBridge` instances are built with the *actual*
channel width of the skip tensor they are applied to at each stage, which
also happens to be the channel-correct thing to do for the dual-skip fusion
in decoder2.

Environment
-----------
Needs `vmamba.py` (your VMamba/VSSM source) importable from the same
directory or on `sys.path`, and a working `mamba_ssm` install -- i.e. this
must run in Colab (or another Linux+CUDA box), not native Windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vmamba import (
    PatchEmbed2D,
    PatchMerging2D,
    PatchExpand2D,
    Final_PatchExpand2D,
    VSSLayer,
    VSSLayer_up,
)


# ─────────────────────────────────────────────────────────────
# 1. Building blocks reused / fixed from rmsnet.py
# ─────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """Two-layer conv residual block, applied channel-first after each
    Mamba decoder stage (same as rmsnet.py)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class SpatialAttentionBridge(nn.Module):
    """Avg+max channel pooling -> dilated 7x7 conv -> sigmoid gate."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, dilation=3, padding=9, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True).values
        spatial = torch.cat([avg_pool, max_pool], dim=1)
        return x * self.sigmoid(self.conv(spatial))


class ChannelAttentionBridge(nn.Module):
    """GAP -> FC -> sigmoid -> residual recalibration."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        z = self.gap(x).view(B, C)
        a = self.fc(z).view(B, C, 1, 1)
        return x + a * x


class SCAttentionBridge(nn.Module):
    """SAB followed by CAB, applied channel-first. `in_channels` must match
    the actual channel width of whatever tensor is passed in."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.sab = SpatialAttentionBridge()
        self.cab = ChannelAttentionBridge(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(self.sab(x))


# ─────────────────────────────────────────────────────────────
# 2. ASPP (conv-based, operates channel-first on the bottleneck)
# ─────────────────────────────────────────────────────────────
class ASPP(nn.Module):
    """Same 5-branch design as the DoubleU-Net TF implementation
    (global pool + 1x1 + three atrous 3x3 convs @ 6/12/18), reimplemented in
    PyTorch. Input/output are both channel-first (B, C, H, W); the caller is
    responsible for permuting Mamba's (B, H, W, C) features in and out."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        def branch(dilation):
            k = 1 if dilation == 1 else 3
            pad = 0 if dilation == 1 else dilation
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, padding=pad,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b1 = branch(1)
        self.b6 = branch(6)
        self.b12 = branch(12)
        self.b18 = branch(18)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        y_pool = F.interpolate(self.pool_branch(x), size=(H, W),
                                mode="bilinear", align_corners=False)
        y = torch.cat([y_pool, self.b1(x), self.b6(x), self.b12(x), self.b18(x)], dim=1)
        return self.project(y)


class MambaASPP(nn.Module):
    """Wraps ASPP with the (B,H,W,C) <-> (B,C,H,W) permutes Mamba needs, and
    projects back to the original channel width so it drops in between the
    encoder bottleneck and the decoder without changing dims."""

    def __init__(self, channels: int, bottleneck_channels: int = None):
        super().__init__()
        bottleneck_channels = bottleneck_channels or channels
        self.aspp = ASPP(channels, bottleneck_channels)
        self.project_back = (
            nn.Identity() if bottleneck_channels == channels
            else nn.Conv2d(bottleneck_channels, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        x_cf = x.permute(0, 3, 1, 2).contiguous()
        x_cf = self.project_back(self.aspp(x_cf))
        return x_cf.permute(0, 2, 3, 1).contiguous()


# ─────────────────────────────────────────────────────────────
# 3. Mamba encoder (used twice: encoder1, encoder2 -- independent weights,
#    no pretraining, exactly the "built from scratch" role of DoubleU-Net's
#    encoder2, applied to BOTH branches here)
# ─────────────────────────────────────────────────────────────
class MambaEncoder(nn.Module):
    DIMS = [96, 192, 384, 768]

    def __init__(self, in_chans=3, patch_size=4, depths=(2, 2, 2, 2),
                 d_state=16, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        dims = self.DIMS
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0],
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i, dim in enumerate(dims):
            self.layers.append(VSSLayer(
                dim=dim,
                depth=depths[i],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=None if i == len(dims) - 1 else PatchMerging2D,
            ))

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) -> bottleneck (B, H/32, W/32, 768), skip_list[0..3]
        (channels-last, at resolutions H/4, H/8, H/16, H/32)."""
        skip_list = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list


# ─────────────────────────────────────────────────────────────
# 4. Skip fusion for decoder2 (concat both encoders' skip -> project back
#    to native channel width -> SCAB)
# ─────────────────────────────────────────────────────────────
class DualSkipFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Linear(2 * channels, channels, bias=False)
        self.scab = SCAttentionBridge(channels)

    def forward(self, skip1: torch.Tensor, skip2: torch.Tensor) -> torch.Tensor:
        # both (B, H, W, C) at the same stage resolution/width
        fused = self.project(torch.cat([skip1, skip2], dim=-1))     # (B,H,W,C)
        fused_cf = fused.permute(0, 3, 1, 2).contiguous()
        fused_cf = self.scab(fused_cf)
        return fused_cf.permute(0, 2, 3, 1).contiguous()


# ─────────────────────────────────────────────────────────────
# 5. Decoders
# ─────────────────────────────────────────────────────────────
class MambaDecoder1(nn.Module):
    """Mirrors VSSLayer_up stages, single-source skip (encoder1 only),
    SCAB on the skip + ResidualBlock after each stage -- same pattern as
    rmsnet.py, with corrected SCAB channel widths."""

    dims_decoder = [768, 384, 192, 96]
    skip_dims = [768, 384, 192]  # channel width of skip used at stage 1,2,3

    def __init__(self, depths_decoder=(2, 2, 2, 2), d_state=16, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers_up = nn.ModuleList()
        for i, dim in enumerate(self.dims_decoder):
            self.layers_up.append(VSSLayer_up(
                dim=dim,
                depth=depths_decoder[i],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_decoder[:i]):sum(depths_decoder[:i + 1])],
                norm_layer=norm_layer,
                upsample=None if i == 0 else PatchExpand2D,
            ))

        self.scab = nn.ModuleList([SCAttentionBridge(c) for c in self.skip_dims])
        self.res_blocks = nn.ModuleList([
            ResidualBlock(c, c) for c in self.dims_decoder[1:]  # 384, 192, 96
        ])

    def forward(self, x: torch.Tensor, skip_list: list) -> torch.Tensor:
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
                continue

            skip_idx = len(skip_list) - inx          # 3, 2, 1
            bridge_idx = inx - 1                       # 0, 1, 2

            skip = skip_list[skip_idx].permute(0, 3, 1, 2).contiguous()
            skip = self.scab[bridge_idx](skip)
            skip = skip.permute(0, 2, 3, 1).contiguous()

            x = layer_up(x + skip)

            x_cf = x.permute(0, 3, 1, 2).contiguous()
            x_cf = self.res_blocks[bridge_idx](x_cf)
            x = x_cf.permute(0, 2, 3, 1).contiguous()

        return x


class MambaDecoder2(nn.Module):
    """Same stage structure as MambaDecoder1, but at every stage fuses
    skip1[stage] (from encoder1) AND skip2[stage] (from encoder2) via
    DualSkipFusion before adding into the decoder stream -- matches
    DoubleU-Net's decoder2 using skip connections from both encoders."""

    dims_decoder = [768, 384, 192, 96]
    skip_dims = [768, 384, 192]

    def __init__(self, depths_decoder=(2, 2, 2, 2), d_state=16, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers_up = nn.ModuleList()
        for i, dim in enumerate(self.dims_decoder):
            self.layers_up.append(VSSLayer_up(
                dim=dim,
                depth=depths_decoder[i],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_decoder[:i]):sum(depths_decoder[:i + 1])],
                norm_layer=norm_layer,
                upsample=None if i == 0 else PatchExpand2D,
            ))

        self.fusion = nn.ModuleList([DualSkipFusion(c) for c in self.skip_dims])
        self.res_blocks = nn.ModuleList([
            ResidualBlock(c, c) for c in self.dims_decoder[1:]
        ])

    def forward(self, x: torch.Tensor, skip_list_1: list, skip_list_2: list) -> torch.Tensor:
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
                continue

            skip_idx = len(skip_list_1) - inx
            bridge_idx = inx - 1

            fused_skip = self.fusion[bridge_idx](
                skip_list_1[skip_idx], skip_list_2[skip_idx]
            )
            x = layer_up(x + fused_skip)

            x_cf = x.permute(0, 3, 1, 2).contiguous()
            x_cf = self.res_blocks[bridge_idx](x_cf)
            x = x_cf.permute(0, 2, 3, 1).contiguous()

        return x


# ─────────────────────────────────────────────────────────────
# 6. Output head (mirrors VSSM.forward_final, but explicit + reusable twice)
# ─────────────────────────────────────────────────────────────
class MambaOutputHead(nn.Module):
    def __init__(self, in_dim: int = 96, dim_scale: int = 4, num_classes: int = 1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.final_up = Final_PatchExpand2D(dim=in_dim, dim_scale=dim_scale, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(in_dim // dim_scale, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.final_up(x)               # (B, H, W, C) -> full res, C/dim_scale
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.final_conv(x)
        return torch.sigmoid(x)


# ─────────────────────────────────────────────────────────────
# 7. Full DoubleMambaUNet
# ─────────────────────────────────────────────────────────────
class DoubleMambaUNet(nn.Module):
    """
    Args:
        in_chans: input image channels (3 for RGB Kvasir-SEG frames)
        num_classes: mask channels per branch (1 for binary polyp segmentation)
        depths / depths_decoder: Mamba block depths per stage, used for BOTH
            encoders and BOTH decoders. Kept lighter than the RMSNet default
            (2,2,9,2) since you now have two full encoder-decoder Mamba
            networks in memory at once -- bump this up if your GPU allows.
        aspp_channels: bottleneck width used inside ASPP's internal convs.

    Forward returns a (B, 2*num_classes, H, W) tensor:
        [:, :num_classes]  -> Output1 (Network 1's mask)
        [:, num_classes:]  -> Output2 (Network 2's mask, the final prediction)
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 1,
                 patch_size: int = 4, depths=(2, 2, 2, 2),
                 depths_decoder=(2, 2, 2, 2), d_state: int = 16,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1, aspp_channels: int = 256):
        super().__init__()

        # ---- Network 1 ----
        self.encoder1 = MambaEncoder(in_chans, patch_size, depths, d_state,
                                      drop_rate, attn_drop_rate, drop_path_rate)
        self.aspp1 = MambaASPP(channels=768, bottleneck_channels=aspp_channels)
        self.decoder1 = MambaDecoder1(depths_decoder, d_state, drop_rate,
                                       attn_drop_rate, drop_path_rate)
        self.head1 = MambaOutputHead(in_dim=96, dim_scale=patch_size, num_classes=num_classes)

        # ---- Network 2 ----
        self.encoder2 = MambaEncoder(in_chans, patch_size, depths, d_state,
                                      drop_rate, attn_drop_rate, drop_path_rate)
        self.aspp2 = MambaASPP(channels=768, bottleneck_channels=aspp_channels)
        self.decoder2 = MambaDecoder2(depths_decoder, d_state, drop_rate,
                                       attn_drop_rate, drop_path_rate)
        self.head2 = MambaOutputHead(in_dim=96, dim_scale=patch_size, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # ---- Network 1: input -> Output1 ----
        bottleneck1, skip1 = self.encoder1(x)
        bottleneck1 = self.aspp1(bottleneck1)
        dec1 = self.decoder1(bottleneck1, skip1)
        output1 = self.head1(dec1)                       # (B, num_classes, H, W)

        # ---- Multiply gate: same spatial size as x, broadcast over channels ----
        gated = x * output1

        # ---- Network 2: gated input -> Output2 (uses skip1 AND skip2) ----
        bottleneck2, skip2 = self.encoder2(gated)
        bottleneck2 = self.aspp2(bottleneck2)
        dec2 = self.decoder2(bottleneck2, skip1, skip2)
        output2 = self.head2(dec2)

        # ---- Concatenate ----
        return torch.cat([output1, output2], dim=1)


# ─────────────────────────────────────────────────────────────
# Training loss note (not implemented here, just guidance):
#
#   logits = model(images)                       # (B, 2, H, W)
#   out1, out2 = logits[:, :1], logits[:, 1:]
#   loss = bce_dice(out1, mask) + bce_dice(out2, mask)   # deep supervision
#   # at inference time, use out2 as the final prediction, same as DoubleU-Net
#
# Ask if you'd like this wired into a full Kvasir-SEG training loop (matching
# the metric suite -- Jaccard/Dice/Precision/Recall/Specificity/HD95 -- used
# in your other notebooks).
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Quick shape sanity check (needs mamba_ssm installed + CUDA, i.e. Colab).
    model = DoubleMambaUNet(in_chans=3, num_classes=1)
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: (1, 2, 256, 256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

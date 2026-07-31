"""
BB-Swin-Unet: bounding-box-gated skip connections for Swin-Unet.

Adapts the BB-UNet idea (Fig. 5: BB-Conv branch = maxpool + 2 convs + sigmoid,
multiplied element-wise into each skip connection before concatenation) to
Swin-Unet, whose skip connections are token sequences (B, L, C), not spatial
feature maps (B, C, H, W).

Key design decision (this is the part that answers "it's not a conv layer"):
the BB-Conv branch itself stays fully convolutional -- it operates on the 2D
bbox map, downsampled with adaptive max-pooling to match each stage's token
grid resolution. Only at the very last step is the result flattened to
(B, L, C) tokens, so it lines up with the corresponding x_downsample[i] and
can be multiplied into it elementwise -- exactly the circled-x gate from
Fig. 5, just relocated to just before the flatten instead of staying 2D
all the way through.

Swin-Unet's decoder (forward_up_features) only ever concatenates
x_downsample[0], x_downsample[1], x_downsample[2] (the 3 encoder stages
before the bottleneck) -- x_downsample[3] IS the bottleneck and is never
used as a skip, so it is not gated.

Requires the original swin_transformer_unet_skip_expand_decoder_sys.py
(SwinTransformerSys) to already be importable in your project, unmodified.
"""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class BBConvGate2D(nn.Module):
    """Fig. 5's BB-Conv branch (maxpool -> conv -> conv -> sigmoid), adapted to
    produce a per-token gate for one Swin skip connection.

    Stays 2D/convolutional the whole way through -- the bbox map is
    max-pooled down to this stage's (h, w) token grid (max-pool, not average,
    so a patch is marked "inside a box" if any pixel in it was), then two 3x3
    convs + sigmoid produce a (B, C, h, w) gate map with the same channel
    width as the skip connection it will multiply. Flattening to (B, h*w, C)
    happens only in the return statement, purely to match the skip tensor's
    token layout.
    """

    def __init__(self, out_dim, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, bbox_map, token_resolution):
        """
        bbox_map: (B, in_channels, H, W) -- the full-resolution bbox map fed
                  into the model (same one used for the BB-UNet's skip gates).
        token_resolution: (h, w) -- this stage's token grid size, e.g. (56, 56).
        Returns: (B, h*w, out_dim) gate in [0, 1], ready to multiply into the
                 matching x_downsample[i] token tensor.
        """
        h, w = token_resolution
        pooled = F.adaptive_max_pool2d(bbox_map, output_size=(h, w))   # (B, in_channels, h, w)
        gate = self.sigmoid(self.conv2(self.act(self.conv1(pooled))))  # (B, out_dim, h, w)
        gate = gate.flatten(2).transpose(1, 2)                          # (B, h*w, out_dim)
        return gate


class BBSwinUnet(nn.Module):
    """Swin-Unet with bounding-box-gated skip connections.

    Input convention matches the BB-UNet notebook: a single fused tensor
    (B, bb_channels + in_chans, H, W), bbox channel(s) first, RGB after --
    same as `fused = torch.cat([bbmap, image], dim=0)` in the dataset.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1,          # 1 = binary segmentation (logits, use BCE/Dice like your UNet)
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 bb_channels=1):         # number of bbox-map channels (1 for a single class, e.g. Kvasir 'polyp')
        super(BBSwinUnet, self).__init__()
        self.num_classes = num_classes
        self.bb_channels = bb_channels

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )

        # Skip connections used by forward_up_features are x_downsample[0..num_layers-2]
        # (the bottleneck, x_downsample[num_layers-1], is never concatenated -- see
        # SwinTransformerSys.forward_up_features: it only indexes x_downsample[3-inx]
        # for inx=1,2,3, i.e. indices 2,1,0 when num_layers=4).
        patches_resolution = self.swin_unet.patches_resolution
        num_layers = self.swin_unet.num_layers

        self.skip_resolutions = [
            (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
            for i in range(num_layers - 1)
        ]
        skip_dims = [int(embed_dim * 2 ** i) for i in range(num_layers - 1)]

        self.bb_gates = nn.ModuleList([
            BBConvGate2D(out_dim=d, in_channels=bb_channels) for d in skip_dims
        ])

    def forward(self, x):
        bb = x[:, :self.bb_channels, :, :]
        img = x[:, self.bb_channels:, :, :]

        if img.size(1) == 1:
            img = img.repeat(1, 3, 1, 1)

        x_enc, x_downsample = self.swin_unet.forward_features(img)

        # Gate each skip connection with its matching BB-Conv branch output --
        # this is the circled-x multiply from Fig. 5, applied in token space.
        gated_downsample = list(x_downsample)
        for i, (res, gate_module) in enumerate(zip(self.skip_resolutions, self.bb_gates)):
            gate = gate_module(bb, res)                     # (B, L_i, C_i)
            gated_downsample[i] = x_downsample[i] * gate     # elementwise multiply

        x_dec = self.swin_unet.forward_up_features(x_enc, gated_downsample)
        logits = self.swin_unet.up_x4(x_dec)
        return logits

    def load_from(self, pretrained_path):
        """Optional: load ImageNet-pretrained Swin-T encoder weights (e.g.
        swin_tiny_patch4_window7_224.pth). The bb_gates are new modules with
        no pretrained counterpart, so they're left at their random init --
        strict=False handles that. Skip entirely if training from scratch."""
        if pretrained_path is None:
            print("none pretrain")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        if "model" not in pretrained_dict:
            print("---start load pretrained model by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            return

        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained model of swin encoder---")
        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k: v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{}; shape pretrain:{}; shape model:{}".format(
                        k, full_dict[k].shape, model_dict[k].shape))
                    del full_dict[k]
        self.swin_unet.load_state_dict(full_dict, strict=False)


if __name__ == "__main__":
    # Quick shape sanity check -- run this file directly to verify wiring
    # before dropping BBSwinUnet into your training notebook.
    model = BBSwinUnet(img_size=224, num_classes=1, bb_channels=1)
    dummy_bb = torch.zeros(2, 1, 224, 224)
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_fused = torch.cat([dummy_bb, dummy_img], dim=1)   # (2, 4, 224, 224)
    out = model(dummy_fused)
    print("Output shape:", out.shape)   # expected: (2, 1, 224, 224)

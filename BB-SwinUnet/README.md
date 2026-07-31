# BB-Swin-Unet: Bounding-Box-Gated Swin-Unet for Polyp Segmentation

A Swin-Unet variant whose skip connections are gated by object-detector bounding boxes. A YOLOv8 detector localizes polyps; its predicted boxes are rasterized into binary maps and injected into Swin-Unet's skip connections through a dedicated convolutional gating branch — giving a transformer segmentation backbone an explicit location prior it otherwise has no mechanism to use.

> Adapts the BB-UNet location-prior formulation (bounding-box binary maps processed through a small conv branch and multiplied element-wise into skip connections before decoder concatenation) — originally a purely convolutional mechanism — to Swin-Unet, whose skip connections are token sequences `(B, L, C)` rather than spatial feature maps `(B, C, H, W)`.

## Why

Swin-Unet integrates global context well through shifted-window self-attention, but — like the U-Net it's modeled on — has no explicit mechanism for incorporating *where* an object of interest is expected to be; localization is left entirely to mask supervision. This project treats bounding boxes from a lightweight, fast-converging detector as a location prior and gates each skip connection with it, rather than relying on the segmentation network to learn localization implicitly.

The key design problem this solves: a convolutional gating branch can't be applied directly to a token sequence. **BB-Swin-Unet keeps the gating branch itself fully convolutional** — it operates on the 2D bounding-box map, adaptive-max-pooled to each encoder stage's token-grid resolution — and only flattens its output to `(B, L, C)` token layout at the very last step, immediately before the element-wise multiply into the matching skip tensor.

## Pipeline

```
Kvasir-SEG (images + masks)
        │
        ▼
Binary masks → bounding boxes (connected-component analysis, single class: "polyp")
        │
        ▼
YOLOv8n detector training  (Ultralytics)
        │
        ▼
Trained detector → binary bounding-box maps (one per image)
        │
        ▼
BB-Swin-Unet training
  BB-Conv gate operates on the 2D bbox map, downsampled per
  encoder stage, then flattened to token layout only immediately
  before gating the matching skip connection
        │
        ▼
Evaluation: Jaccard, Dice, Accuracy,
Precision, Recall, Specificity, HD / HD95
```

## Architecture

Swin-Unet's decoder concatenates three encoder-stage skip connections (`x_downsample[0..2]`); its deepest stage is the bottleneck itself and is never used as a skip. BB-Swin-Unet adds one `BBConvGate2D` module per skip connection:

1. The bounding-box map is adaptive-max-pooled down to that stage's token-grid resolution (e.g. 56×56, 28×28, 14×14 for a 224×224 input with patch size 4).
2. Two 3×3 convolutions + sigmoid produce a gate map with the same channel width as that stage's skip connection — the same max-pool → conv → conv → sigmoid pattern as the original convolutional BB-Conv branch.
3. The gate is flattened to `(B, L, C)` and multiplied element-wise into the corresponding `x_downsample[i]` token tensor before it reaches the decoder.

The rest of the Swin-Unet backbone (`SwinTransformerSys`) is used unmodified — `BBSwinUnet` is a thin wrapper that calls its `forward_features`, gates the returned skip list, then hands the gated list to its existing `forward_up_features` and `up_x4`.

## Dataset

[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) — 1,000 gastrointestinal polyp images with pixel-level ground-truth masks, binary segmentation task (background vs. polyp). Bounding-box labels for detector training are derived directly from the segmentation masks via connected-component analysis (a mask may contain more than one polyp region), so no separate box annotation is needed.

## Repository Structure

```
.
├── data/                          # Kvasir-SEG images/ and masks/ (not tracked; see Dataset)
├── yolo_dataset/                  # generated YOLO-format images/labels + data.yaml
├── bb_swin_unet.py                # BBConvGate2D + BBSwinUnet wrapper
├── swin_transformer_unet_skip_expand_decoder_sys.py   # unmodified Swin-Unet backbone
├── notebooks/
│   └── bbswinunet_kvasir_pipeline.ipynb   # YOLO stage + BB-Swin-Unet training/eval
├── models/                        # saved checkpoints (best/final)
├── logs/                          # per-epoch CSV training log
└── results/                       # metric curves, test-set predictions, test metrics CSV
```

## Getting Started

### Requirements
```
torch
torchvision
torchmetrics
albumentations
opencv-python
ultralytics
medpy
timm
einops
pandas
matplotlib
pyyaml
```

### Usage
1. Download [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) and point `DATA_DIR` in the notebook's config cell at its `images/`/`masks/` folders.
2. Make sure `bb_swin_unet.py` and `swin_transformer_unet_skip_expand_decoder_sys.py` are importable from the notebook's working directory.
3. Run `notebooks/bbswinunet_kvasir_pipeline.ipynb` top-to-bottom. It derives YOLO labels from the masks, trains the detector, builds bounding-box maps, trains BB-Swin-Unet, and evaluates on a held-out test split, saving metrics/plots to `results/`.

## Metrics

Jaccard/IoU, Dice (F1), Accuracy, Precision, Recall, Specificity, and Hausdorff Distance (HD / HD95).

## Results

_TODO: fill in once training completes — e.g. BB-Swin-Unet vs. a vanilla (ungated) Swin-Unet baseline on the same split._

## Acknowledgements

Bounding-box gating formulation adapted from the BB-UNet location-prior literature. Swin-Unet backbone based on the [official Swin-Unet implementation](https://github.com/HuCaoFighting/Swin-Unet). Detector: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). Dataset: [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/).

## License

_TODO: add a license (e.g. MIT, Apache-2.0) before publishing._

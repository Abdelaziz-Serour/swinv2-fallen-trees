# SwinV2 + Mask R-CNN — Fallen Trees Instance Segmentation

This project fine-tunes **Swin Transformer V2** as a backbone for **Mask R-CNN** using PyTorch and Torchvision for **instance segmentation** of fallen trees.
It uses a COCO-format dataset and evaluates performance with pycocotools (COCO mAP).

-------------------------------------------------------------------------------
## Repository Structure
```
swinv2-fallen-trees/
├─ configs/              # YAML configuration files
│  └─ defaults.yaml
├─ notebooks/            # Original training notebook
│  └─ SWINV2_Final_Training.ipynb
├─ scripts/              # Helper shell scripts
│  ├─ train.sh
│  └─ eval.sh
├─ src/                  # All training/inference code
│  ├─ dataset_coco.py
│  ├─ transforms.py
│  ├─ model_swinv2_maskrcnn.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ infer.py
├─ data/                 # (not committed) place your COCO dataset here
├─ requirements.txt
├─ .gitignore
└─ README.md
```
-------------------------------------------------------------------------------
## Installation
```bash
git clone https://github.com/<your-username>/swinv2-fallen-trees.git
cd swinv2-fallen-trees
```
# Create a virtual environment (recommended)
```bash
python -m venv .venv
```
# Windows: .venv\Scripts\activate
# Linux/Mac:
``` bash
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Format (COCO)

Expected layout (change in configs/defaults.yaml if different):
```
data/fallen_trees_coco_tiles/
├─ images/
│  ├─ train/
│  └─ val/
└─ annotations/
   ├─ train.json
   └─ val.json
```
-------------------------------------------------------------------------------

## Configuration

Edit `configs/defaults.yaml` to set:
- data paths (`data.root`, etc.)
- `data.num_classes` (includes background, e.g., 2 = background + 1 class)
- training hyperparameters (epochs, batch_size, lr, weight_decay, step_size, gamma)
- model settings (`swin_name`, image sizes)
- output directories under `paths`

-------------------------------------------------------------------------------

## Training
```bash
scripts/train.sh
```
# or
```
python -m src.train --config configs/defaults.yaml
```
Checkpoints will be saved in:
runs/swinv2_maskrcnn_tiles/checkpoints/epoch_XXX.pth

-------------------------------------------------------------------------------

## Evaluation (COCO mAP)

```bash
scripts/eval.sh runs/swinv2_maskrcnn_tiles/checkpoints/epoch_020.pth
```
# or
```bash
python -m src.evaluate --config configs/defaults.yaml --ckpt runs/.../epoch_020.pt
```
This prints standard COCO metrics for bbox and segm.

-------------------------------------------------------------------------------

## Inference (single image)
```bash
python -m src.infer \
  --config configs/defaults.yaml \
  --ckpt runs/swinv2_maskrcnn_tiles/checkpoints/epoch_020.pth \
  --image path/to/sample.jpg \
  --out outputs/pred.png \
  --thr 0.5
```
Outputs an image with predicted boxes drawn.

-------------------------------------------------------------------------------

## Components

- Backbone: SwinV2 (from `timm`)
- Detection/Segmentation head: Mask R-CNN (from `torchvision`)
- Dataset loader: minimal COCO reader (boxes, labels, masks)
- Evaluation: COCO mAP (bbox & segm) via `pycocotools`
- Inference: quick visualization utility

-------------------------------------------------------------------------------

## Tips

- Set `data.num_classes` correctly (background + your categories).
- If you switch to a larger SwinV2 backbone, consider increasing image size.
- On Colab, mount Drive and update dataset/output paths in the YAML (avoid hardcoded /content paths in code).

-------------------------------------------------------------------------------

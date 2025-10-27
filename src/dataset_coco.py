"""
Minimal COCO instance dataset for torchvision Mask R-CNN.

- Returns images as tensors and targets dict with: boxes, labels, masks, image_id
- Handles polygon segmentations (list of lists) via pycocotools to decode masks.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, List
import json
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as maskUtils


class CocoInstanceDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: Path, ann_file: Path, transforms=None):
        self.img_dir = Path(img_dir)
        self.ann_file = Path(ann_file)
        self.transforms = transforms

        with open(self.ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # contiguous category mapping (background=0)
        cats = coco["categories"]
        self.cat_to_contig = {}
        for i, c in enumerate(sorted(cats, key=lambda x: x["id"])):
            self.cat_to_contig[c["id"]] = i + 1  # start from 1

        self.ids = sorted(self.images.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.ids[idx]
        info = self.images[img_id]
        img_path = self.img_dir / info["file_name"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        anns = self.anns_by_img.get(img_id, [])
        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.cat_to_contig.get(ann["category_id"], 1))

            seg = ann.get("segmentation", None)
            if isinstance(seg, list) and len(seg) > 0:
                rles = maskUtils.frPyObjects(seg, h, w)
                rle = maskUtils.merge(rles)
                m = maskUtils.decode(rle)  # HxW
            elif isinstance(seg, dict) and "counts" in seg:
                m = maskUtils.decode(seg)
            else:
                m = np.zeros((h, w), dtype=np.uint8)
            masks.append(m.astype(np.uint8))

        if len(masks) == 0:
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            masks_tensor = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

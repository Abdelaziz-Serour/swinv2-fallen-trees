"""
COCO evaluation:
- Exports bbox detections JSON
- Runs COCOeval for bbox
- (Optional) Exports segmentation detections if masks are present and runs segm eval
"""
import argparse
import json
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils

from src.dataset_coco import CocoInstanceDataset, collate_fn
from src.transforms import build_transforms
from src.model_swinv2_maskrcnn import build_model


@torch.no_grad()
def export_detections(model, loader, device, out_bbox_json: Path, out_segm_json: Path = None, score_thr: float = 0.0):
    model.eval()
    bbox_results = []
    segm_results = [] if out_segm_json is not None else None

    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        for tgt, pred in zip(targets, outputs):
            image_id = int(tgt["image_id"].item())
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            # bbox results
            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                if s < score_thr:
                    continue
                w, h = float(x2 - x1), float(y2 - y1)
                bbox_results.append({
                    "image_id": image_id,
                    "category_id": int(l),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(s),
                })

            # segm results (RLE per detection) if requested
            if segm_results is not None and "masks" in pred:
                # pred["masks"] is [N, 1, H, W] float tensor with logits/sigmoid
                masks = pred["masks"].squeeze(1).cpu().numpy()  # [N,H,W]
                for m, s, l in zip(masks, scores, labels):
                    if s < score_thr:
                        continue
                    m_bin = (m > 0.5).astype(np.uint8)  # threshold
                    rle = maskUtils.encode(np.asfortranarray(m_bin))
                    rle["counts"] = rle["counts"].decode("ascii")  # bytes -> str for JSON
                    segm_results.append({
                        "image_id": image_id,
                        "category_id": int(l),
                        "segmentation": rle,
                        "score": float(s),
                    })

    out_bbox_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bbox_json, "w") as f:
        json.dump(bbox_results, f)
    print(f"Saved bbox detections to {out_bbox_json}")

    if segm_results is not None:
        with open(out_segm_json, "w") as f:
            json.dump(segm_results, f)
        print(f"Saved segm detections to {out_segm_json}")


def run_coco_eval(gt_json: str, dt_json: str, iou_type: str):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main(cfg_path: str, ckpt_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = build_transforms(cfg["model"]["image_size"])

    ds = CocoInstanceDataset(Path(cfg["data"]["val_images"]), Path(cfg["data"]["val_json"]), transforms=T)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_model(
        num_classes=cfg["data"]["num_classes"],
        swin_name=cfg["model"]["swin_name"],
        min_size_train=cfg["model"]["min_size_train"],
        max_size_train=cfg["model"]["max_size_train"],
        min_size_test=cfg["model"]["min_size_test"],
        max_size_test=cfg["model"]["max_size_test"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)  # allow raw state_dict or full dict
    model.load_state_dict(state, strict=False)

    eval_dir = Path(cfg["paths"]["eval_dir"])
    bbox_json = eval_dir / "val_detections_bbox.json"
    segm_json = eval_dir / "val_detections_segm.json"

    # Export detections (both bbox and segm)
    export_detections(model, loader, device, bbox_json, out_segm_json=segm_json, score_thr=0.0)

    # COCO eval
    print("\n== COCO bbox metrics ==")
    run_coco_eval(cfg["data"]["val_json"], str(bbox_json), "bbox")
    print("\n== COCO segm metrics ==")
    run_coco_eval(cfg["data"]["val_json"], str(segm_json), "segm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/defaults.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)

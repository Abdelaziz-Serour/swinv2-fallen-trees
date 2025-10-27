"""
Single-image inference + quick visualization (boxes only by default).
"""
import argparse
from pathlib import Path
import yaml
import torch
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as F

from src.model_swinv2_maskrcnn import build_model


@torch.no_grad()
def main(cfg_path: str, ckpt_path: str, image_path: str, out_path: str, score_thr: float = 0.5):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=cfg["data"]["num_classes"],
        swin_name=cfg["model"]["swin_name"],
        min_size_train=cfg["model"]["min_size_train"],
        max_size_train=cfg["model"]["max_size_train"],
        min_size_test=cfg["model"]["min_size_test"],
        max_size_test=cfg["model"]["max_size_test"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    t = F.to_tensor(img).to(device).unsqueeze(0)
    pred = model(t)[0]

    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2), s in zip(pred["boxes"].cpu().numpy(), pred["scores"].cpu().numpy()):
        if s < score_thr:
            continue
        cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, arr)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/defaults.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/pred.png")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    main(args.config, args.ckpt, args.image, args.out, score_thr=args.thr)

"""
Training script:
- Reads YAML config
- Builds datasets, dataloaders, model
- AdamW + StepLR
- Optional AMP mixed precision
- Saves epoch checkpoints
"""
import argparse
import time
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset_coco import CocoInstanceDataset, collate_fn
from src.transforms import build_transforms
from src.model_swinv2_maskrcnn import build_model


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_ckpt(model, optimizer, epoch, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
        path,
    )


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = build_transforms(cfg["model"]["image_size"])

    train_ds = CocoInstanceDataset(Path(cfg["data"]["train_images"]), Path(cfg["data"]["train_json"]), transforms=T)
    val_ds   = CocoInstanceDataset(Path(cfg["data"]["val_images"]), Path(cfg["data"]["val_json"]), transforms=T)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = build_model(
        num_classes=cfg["data"]["num_classes"],
        swin_name=cfg["model"]["swin_name"],
        min_size_train=cfg["model"]["min_size_train"],
        max_size_train=cfg["model"]["max_size_train"],
        min_size_test=cfg["model"]["min_size_test"],
        max_size_test=cfg["model"]["max_size_test"],
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["train"]["step_size"], gamma=cfg["train"]["gamma"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("mixed_precision", True) and torch.cuda.is_available())

    out_dir = Path(cfg["paths"]["out_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg["train"]["epochs"]
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = {}

        for i, (imgs, targets) in enumerate(train_loader, 1):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss_dict = model(imgs, targets)
                loss = sum(v for v in loss_dict.values())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(v.item())

            if i % 25 == 0:
                print(f"[ep {ep}/{epochs}] step {i}/{len(train_loader)} total_loss={loss.item():.4f}")

        sch.step()
        elapsed = time.time() - t0
        metrics_msg = " ".join([f"{k}={(v/len(train_loader)):.4f}" for k, v in running.items()])
        print(f"Epoch {ep} done in {elapsed:.1f}s | {metrics_msg}")

        save_ckpt(model, opt, ep, ckpt_dir / f"epoch_{ep:03d}.pth")

    print("Training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/defaults.yaml")
    args = ap.parse_args()
    main(args.config)

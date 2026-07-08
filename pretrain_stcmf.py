"""阶段0：6 波段跨模态 masked autoencoder 预训练。

用原生 6 波段卫星 + 3 波段 UAV，单时相，做 hybrid 掩码重建，产出供下游
``train_stcmf.py`` 加载的 encoder 权重。

示例：
    python pretrain_stcmf.py --samples 8 --epochs 1 --batch-size 2 --preset small --output-dir outputs/smoke
    python pretrain_stcmf.py --preset vit_base --epochs 100 --batch-size 16 --output-dir outputs/stcmf_v1
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.hybrid_plot import HybridPlotDataset
from models.stcmf import STCMFForPretrain, masked_mse_loss, preset_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/home/WUtongquan/dataset/DataPublication_final"))
    p.add_argument("--locations", nargs="*", default=None)
    p.add_argument("--timepoints", nargs="*", default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--preset", choices=["small", "base", "vit_base"], default="base")
    p.add_argument("--train-mode", choices=["mixed_mask", "satellite_to_uav", "hybrid"], default="hybrid")
    p.add_argument("--satellite-to-uav-prob", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/stcmf_pretrain"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def recon_loss(model: STCMFForPretrain, batch: dict, device: torch.device, train_mode: str, s2u_prob: float):
    sat = batch["sat_all"].to(device)
    uav = batch["uav_rgb"].to(device)
    mask = model.make_masks(sat.shape[0], device, train_mode, s2u_prob)
    out = model(sat, uav, mask)
    loss_sat = masked_mse_loss(out["sat_pred_patches"], sat, out["sat_masked"], model.patch_size)
    loss_uav = masked_mse_loss(out["uav_pred_patches"], uav, out["uav_masked"], model.patch_size)
    return loss_sat + loss_uav, loss_sat, loss_uav


@torch.no_grad()
def evaluate(model: STCMFForPretrain, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        loss, _, _ = recon_loss(model, batch, device, "satellite_to_uav", 1.0)
        total += loss.item()
    return total / max(1, len(loader))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = HybridPlotDataset(
        root_dir=args.data_root,
        locations=args.locations,
        timepoints=args.timepoints,
        max_samples=args.samples,
    )
    print(f"loaded {len(dataset)} paired HYBRID HIPS samples (6-band satellite)")

    val_size = int(len(dataset) * args.val_fraction)
    val_size = val_size if val_size >= 1 and len(dataset) - val_size >= 1 else 0
    if val_size:
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = dataset, None

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds else None

    config = preset_config(args.preset)
    model = STCMFForPretrain(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_val, history = 0, float("inf"), []

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val_loss", best_val))
        history = list(ckpt.get("history", []))
        print(f"resumed {args.resume} from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            loss, loss_sat, loss_uav = recon_loss(model, batch, device, args.train_mode, args.satellite_to_uav_prob)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", sat=f"{loss_sat.item():.4f}", uav=f"{loss_uav.item():.4f}")

        train_loss = total / max(1, len(loader))
        val_loss = evaluate(model, val_loader, device) if val_loader else train_loss
        psnr = 10.0 * math.log10(1.0 / max(val_loss, 1e-8))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} approx_psnr={psnr:.2f}dB")

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "preset": args.preset,
            "history": history,
            "best_val_loss": min(best_val, val_loss),
        }
        torch.save(ckpt, args.output_dir / "checkpoint_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, args.output_dir / "checkpoint_best.pt")
            print(f"saved best checkpoint val_loss={best_val:.6f}")

    print(f"done. checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()

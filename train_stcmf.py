"""STCMF v2 端到端训练：输入多时相双模态影像 → 直接回归 yield。

单阶段、从头训练、只预测产量。带几何数据增强、Huber 损失、三种交叉验证划分，
以及"先证时序收益"的消融开关。

示例：
    # 形状自检
    python selftest.py
    # 消融阶梯（先证时序收益）
    python train_stcmf.py --modalities sat_ms+real_uav --no-temporal --no-crossmodal --folds 5   # 基线
    python train_stcmf.py --modalities sat_ms+real_uav --no-crossmodal --folds 5                 # +时序
    python train_stcmf.py --modalities sat_ms+real_uav --folds 5                                 # +跨模态融合
    # 泛化复核
    python train_stcmf.py --modalities sat_ms+real_uav --cv-mode group_kfold
    python train_stcmf.py --modalities sat_ms+real_uav --cv-mode lolo
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.hybrid_plot import HybridPlotSeriesDataset
from models.stcmf_net import STCMFNet

MODALITY_MAP = {
    "sat_ms": (True, False),
    "real_uav": (False, True),
    "sat_ms+real_uav": (True, True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/home/WUtongquan/dataset/DataPublication_final"))
    p.add_argument("--locations", nargs="*", default=None)
    p.add_argument("--timepoints", nargs="*", default=None)
    p.add_argument("--samples", type=int, default=None, help="max plots（smoke 用）")
    p.add_argument("--sat-size", type=int, default=224)
    p.add_argument("--uav-size", type=int, default=448)
    p.add_argument("--base", type=int, default=32, help="backbone 基宽，先 32 起步")
    p.add_argument("--modalities", choices=list(MODALITY_MAP), default="sat_ms+real_uav")
    # 默认基线（最优且最快）：ablation 已证明 Stage B/C attention 无增益，故默认关闭，需要复现消融时再开。
    p.add_argument("--temporal", action="store_true", help="开启 Stage B 时序 attention（默认关）")
    p.add_argument("--crossmodal", action="store_true", help="开启 Stage C 跨模态 attention（默认关）")
    p.add_argument("--bidirectional-cross", action="store_true")
    p.add_argument("--cv-mode", choices=["random_kfold", "group_kfold", "lolo"], default="random_kfold")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------- 数据增强（双模态、多时相同步几何变换）
def augment_batch(sat: torch.Tensor, uav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sat, uav = sat.clone(), uav.clone()
    for b in range(sat.shape[0]):
        if random.random() < 0.5:
            sat[b] = torch.flip(sat[b], dims=[-1])
            uav[b] = torch.flip(uav[b], dims=[-1])
        if random.random() < 0.5:
            sat[b] = torch.flip(sat[b], dims=[-2])
            uav[b] = torch.flip(uav[b], dims=[-2])
        k = random.randint(0, 3)
        if k:
            sat[b] = torch.rot90(sat[b], k, dims=[-2, -1])
            uav[b] = torch.rot90(uav[b], k, dims=[-2, -1])
    return sat, uav


# ---------------------------------------------------------------- 数据元信息 & 划分
def series_meta(dataset: HybridPlotSeriesDataset):
    recs = [dataset._plot_label(dataset.groups[pid]) for pid in dataset.plot_ids]
    yields = np.array([r.yield_value for r in recs], dtype=np.float32)
    locations = [r.location for r in recs]
    return list(dataset.plot_ids), yields, locations


def make_splits(args, valid_idx, plot_ids, locations):
    if args.cv_mode == "random_kfold":
        kf = KFold(n_splits=min(args.folds, len(valid_idx)), shuffle=True, random_state=args.seed)
        return list(kf.split(valid_idx))
    if args.cv_mode == "group_kfold":
        groups = [plot_ids[i] for i in valid_idx]
        n = min(args.folds, len(set(groups)))
        return list(GroupKFold(n_splits=n).split(valid_idx, groups=groups))
    locs = [locations[i] for i in valid_idx]
    splits = []
    for loc in sorted(set(locs)):
        val = [j for j, l in enumerate(locs) if l == loc]
        train = [j for j, l in enumerate(locs) if l != loc]
        if train and val:
            splits.append((np.array(train), np.array(val)))
    return splits


# ---------------------------------------------------------------- 训练 / 评估
def train_epoch(model, loader, optimizer, device, stats, args, flags):
    use_sat, use_uav, temporal, crossmodal = flags
    model.train()
    total = 0.0
    for batch in loader:
        sat = batch["sat_seq"].to(device)
        uav = batch["uav_seq"].to(device)
        valid = batch["valid"].to(device)
        if not args.no_augment:
            sat, uav = augment_batch(sat, uav)
        pred = model(sat, uav, valid, use_sat=use_sat, use_uav=use_uav, temporal=temporal, crossmodal=crossmodal)

        y = batch["yield"].to(device)
        m = torch.isfinite(y) & (y >= 0)
        if not m.any():
            continue
        target = (y[m] - stats["mean"]) / stats["std"]
        loss = nn.functional.huber_loss(pred[m], target, delta=args.huber_delta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, stats, flags):
    use_sat, use_uav, temporal, crossmodal = flags
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        sat = batch["sat_seq"].to(device)
        uav = batch["uav_seq"].to(device)
        valid = batch["valid"].to(device)
        pred = model(sat, uav, valid, use_sat=use_sat, use_uav=use_uav, temporal=temporal, crossmodal=crossmodal)
        pred = pred.cpu().numpy() * stats["std"] + stats["mean"]
        y = batch["yield"].numpy()
        m = np.isfinite(y) & (y >= 0)
        y_true.extend(y[m].tolist())
        y_pred.extend(pred[m].tolist())
    metrics = {}
    if len(y_true) >= 2:
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    return metrics, y_true, y_pred


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    use_sat, use_uav = MODALITY_MAP[args.modalities]
    temporal = args.temporal
    crossmodal = args.crossmodal
    flags = (use_sat, use_uav, temporal, crossmodal)

    dataset = HybridPlotSeriesDataset(
        root_dir=args.data_root,
        locations=args.locations,
        timepoints=args.timepoints,
        max_plots=args.samples,
        sat_image_size=args.sat_size,
        uav_image_size=args.uav_size,
    )
    num_timepoints = len(dataset.tp_order)
    plot_ids, yields, locations = series_meta(dataset)
    valid_idx = [i for i in range(len(dataset)) if np.isfinite(yields[i]) and yields[i] >= 0]
    if len(valid_idx) < 2:
        raise ValueError("Not enough valid yield labels.")
    print(f"plots={len(dataset)} valid_yield={len(valid_idx)} T={num_timepoints} "
          f"modalities={args.modalities} temporal={temporal} crossmodal={crossmodal} cv={args.cv_mode}")

    splits = make_splits(args, valid_idx, plot_ids, locations)
    fold_metrics = []
    pred_rows = []
    for fold, (train_pos, val_pos) in enumerate(splits, start=1):
        train_idx = [valid_idx[i] for i in train_pos]
        val_idx = [valid_idx[i] for i in val_pos]
        train_y = yields[train_idx]
        stats = {"mean": float(train_y.mean()), "std": float(max(train_y.std(), 1e-6))}

        model = STCMFNet(
            base=args.base,
            num_timepoints=num_timepoints,
            sat_size=args.sat_size,
            uav_size=args.uav_size,
            dropout=args.dropout,
            bidirectional_cross=args.bidirectional_cross,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for _ in tqdm(range(args.epochs), desc=f"fold {fold}/{len(splits)}"):
            train_epoch(model, train_loader, optimizer, device, stats, args, flags)
            scheduler.step()

        metrics, yt, yp = evaluate(model, val_loader, device, stats, flags)
        val_locs = [locations[i] for i in val_idx]  # val plots are all valid -> order aligns with yt/yp
        pred_rows.extend((fold, loc, t, p) for loc, t, p in zip(val_locs, yt, yp))
        fold_metrics.append(metrics)
        print(f"fold={fold} {metrics}")

    keys = sorted({k for m in fold_metrics for k in m})
    print("\n==== STCMF v2 CV summary ====")
    print(f"modalities={args.modalities} temporal={temporal} crossmodal={crossmodal} cv={args.cv_mode}")
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        print(f"{k}={np.mean(vals):.3f}±{np.std(vals):.3f}")

    out = f"preds_{args.cv_mode}_{args.modalities}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "location", "y_true", "y_pred"])
        w.writerows(pred_rows)
    print(f"saved {out} ({len(pred_rows)} rows)")


if __name__ == "__main__":
    main()

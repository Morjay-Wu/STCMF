"""数据自检：扫描配对样本、按地点/时相统计、打印 yield/nitrogen 标签分布。

    python inspect_data.py --data-root /home/WUtongquan/dataset/DataPublication_final
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from data.hybrid_plot import HybridPlotDataset, HybridPlotSeriesDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/home/WUtongquan/dataset/DataPublication_final"))
    p.add_argument("--locations", nargs="*", default=None)
    p.add_argument("--timepoints", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = HybridPlotDataset(root_dir=args.data_root, locations=args.locations, timepoints=args.timepoints)
    print(f"paired single-timepoint samples: {len(ds)}")
    print(f"locations: {sorted(ds.locations)}")
    print(f"timepoints: {ds.timepoints}")

    by_loc_tp = Counter((r.location, r.timepoint) for r in ds.records)
    print("\nper location/timepoint counts:")
    for (loc, tp), n in sorted(by_loc_tp.items()):
        print(f"  {loc:16s} {tp:4s} {n}")

    yields = np.array([r.yield_value for r in ds.records], dtype=np.float32)
    valid = yields[np.isfinite(yields) & (yields >= 0)]
    print(f"\nyield: valid={valid.size}/{yields.size} mean={valid.mean():.2f} std={valid.std():.2f} min={valid.min():.2f} max={valid.max():.2f}")

    n_cls = Counter(r.nitrogen_class for r in ds.records)
    print("nitrogen class distribution:")
    for cls, n in sorted(n_cls.items()):
        print(f"  class {cls}: {n}")

    series = HybridPlotSeriesDataset(root_dir=args.data_root, locations=args.locations, timepoints=args.timepoints)
    print(f"\naggregated plots (for temporal mode): {len(series)} (timepoints/plot up to {len(series.tp_order)})")


if __name__ == "__main__":
    main()

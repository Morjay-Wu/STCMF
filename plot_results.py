"""从训练输出的 preds_*.csv 画出版级结果图：预测 vs 真实散点。

用法：
    先在服务器跑训练生成 CSV，拷回项目根目录，再：
        python plot_results.py

产出：
    Latex/figs/pred_scatter.png       随机 k-fold 预测 vs 真实（按地点着色）
    Latex/figs/pred_scatter_lolo.png  LOLO 预测 vs 真实（若有对应 CSV）
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "Latex" / "figs"
LOC_COLORS = {
    "Ames": "#3E7CB1",
    "Crawfordsville": "#E38E28",
    "Lincoln": "#4CAF72",
    "MOValley": "#8250A6",
    "Scottsbluff": "#D1495B",
}


def read_preds(csv_path: Path):
    """读 preds CSV（列：fold,location,y_true,y_pred），返回 dict。"""
    import csv

    rows = {"fold": [], "location": [], "y_true": [], "y_pred": []}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows["fold"].append(int(r["fold"]))
            rows["location"].append(r["location"])
            rows["y_true"].append(float(r["y_true"]))
            rows["y_pred"].append(float(r["y_pred"]))
    for k in ("y_true", "y_pred"):
        rows[k] = np.asarray(rows[k])
    return rows


def metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return r2, rmse, mae


def scatter(csv_path: Path, out_path: Path, title: str):
    d = read_preds(csv_path)
    yt, yp, locs = d["y_true"], d["y_pred"], d["location"]
    r2, rmse, mae = metrics(yt, yp)

    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    # 1:1 参考线
    ax.plot([lo, hi], [lo, hi], ls="--", color="#888", lw=1.4, zorder=1,
            label="1:1")
    # 最小二乘拟合线
    a, b = np.polyfit(yt, yp, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, a * xs + b, color="#222", lw=1.6, zorder=2,
            label=f"fit: y={a:.2f}x+{b:.1f}")

    for loc in sorted(set(locs)):
        m = np.array([l == loc for l in locs])
        ax.scatter(yt[m], yp[m], s=22, alpha=0.7, edgecolor="none",
                   color=LOC_COLORS.get(loc, "#555"), label=loc, zorder=3)

    txt = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nn = {len(yt)}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=13, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed yield", fontsize=13)
    ax.set_ylabel("Predicted yield", fontsize=13)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}  (R2={r2:.3f} RMSE={rmse:.2f} n={len(yt)})")


def main():
    jobs = [
        ("preds_random_kfold_sat_ms+real_uav.csv", "pred_scatter.png",
         "STCMF: predicted vs. observed yield (random $k$-fold)"),
        ("preds_lolo_sat_ms+real_uav.csv", "pred_scatter_lolo.png",
         "STCMF: predicted vs. observed yield (LOLO)"),
    ]
    found = False
    for csv_name, png_name, title in jobs:
        csv_path = ROOT / csv_name
        if csv_path.exists():
            scatter(csv_path, OUT_DIR / png_name, title)
            found = True
        else:
            print(f"skip: {csv_name} 不存在（先跑 train_stcmf.py 生成）")
    if not found:
        print("\n没有找到任何 preds_*.csv。先在服务器上运行："
              "\n  python train_stcmf.py --cv-mode random_kfold"
              "\n  python train_stcmf.py --cv-mode lolo"
              "\n再把 CSV 拷回项目根目录后重跑本脚本。")


if __name__ == "__main__":
    main()

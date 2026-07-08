"""重画消融柱状图 figs/ablation_r2.png，数值与论文 Table tab:main 保持一致。

数字有变动时改 VALS 再重跑：python plot_ablation.py
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "DejaVu Sans"

LABELS = [
    "End-to-end\nsingle time point",
    "STCMF plain\n(mean + dual pool)",
    "+ temporal\nattention",
    "+ temporal\n+ cross-modal",
]
VALS = [0.722, 0.823, 0.820, 0.818]          # +temporal: 0.826 -> 0.820
ERRS = [np.nan, 0.012, 0.016, 0.016]         # 灰柱无误差棒；后三根为折间 std
COLORS = ["#9E9E9E", "#3E7CB1", "#8250A6", "#E38E28"]

PLAIN, PLAIN_ERR = 0.823, 0.012

fig, ax = plt.subplots(figsize=(9.6, 6))
x = np.arange(len(VALS))

ax.axhspan(PLAIN - PLAIN_ERR, PLAIN + PLAIN_ERR, color="#3E7CB1", alpha=0.10, zorder=0)
ax.axhline(PLAIN, ls="--", color="#3E7CB1", alpha=0.6, lw=1.5, zorder=1)

ax.bar(x, VALS, width=0.62, color=COLORS, zorder=3)

for i, (v, e) in enumerate(zip(VALS, ERRS)):
    if not np.isnan(e):
        ax.errorbar(i, v, yerr=e, ecolor="#222", elinewidth=2,
                    capsize=6, capthick=2, zorder=5)

for i, (v, e) in enumerate(zip(VALS, ERRS)):
    top = v + (0 if np.isnan(e) else e)
    ax.text(i, top + 0.004, f"{v:.3f}", ha="center", va="bottom",
            fontweight="bold", fontsize=15)

ax.set_title("Architecture ablation: attention adds no distinguishable gain",
             fontweight="bold", fontsize=16, pad=14)
ax.set_ylabel("Yield $R^2$ (random $k$-fold)", fontsize=14)
ax.set_ylim(0.68, 0.86)
ax.set_yticks(np.arange(0.68, 0.861, 0.02))
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.spines[["top", "right"]].set_visible(False)

ax.text(len(VALS) - 0.55, 0.693,
        "shaded band = plain model's ±0.012 fold-to-fold spread",
        ha="right", va="bottom", style="italic", fontsize=11,
        color="#3E7CB1", alpha=0.85)

fig.tight_layout()
out = Path(__file__).parent / "Latex" / "figs" / "ablation_r2.png"
fig.savefig(out, dpi=200)
print(f"saved {out}")

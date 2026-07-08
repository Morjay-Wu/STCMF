"""HYBRID HIPS V3.5 数据加载（原生 6 波段卫星 + 3 波段 UAV）。

改写自 ``crossmodal_crop_canopy_repro/data/hybrid_hips.py``。关键区别：
``_read_satellite`` 返回**全部 6 个波段**，不再砍成 RGB。

提供两种数据集：
- ``HybridPlotDataset``        : 单时相样本（阶段0 预训练 / 阶段1 单时相监督）。
- ``HybridPlotSeriesDataset``  : 按 plot 聚合的多时相样本（阶段2 时序融合）。

目录约定：
    DataPublication_final/
      GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv
      Satellite/{location}/{TP}/{location}-{TP}-{experiment}_{range}_{row}.TIF
      UAV/{location}/{TP}/{location}-{TP}-{experiment}_{range}_{row}.PNG
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
except ImportError:  # pragma: no cover
    rasterio = None
    NotGeoreferencedWarning = Warning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SAT_CHANNELS = 6
UAV_CHANNELS = 3

# CSV 里 experiment 列存在拼写错误，与磁盘文件名不一致。
# 例：MOValley 的 176 行写成 "Hyrbrids"，而文件名是 "Hybrids"，
# 导致整地点匹配失败（连带 nitrogen 的某个类别全空）。此处按文件名规范化。
EXPERIMENT_ALIASES = {"Hyrbrids": "Hybrids"}


@dataclass(frozen=True)
class PlotRecord:
    location: str
    timepoint: str
    experiment: str
    range_no: str
    row_no: str
    sat_path: Path
    uav_path: Path
    yield_value: float
    nitrogen_value: float
    nitrogen_class: int
    days_to_anthesis: float

    @property
    def plot_id(self) -> str:
        """跨时相唯一的 plot 标识，用于 GroupKFold（绝不作为模型输入）。"""
        return f"{self.location}-{self.experiment}-{self.range_no}-{self.row_no}"


def _int_string(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _safe_float(value: object, default: float = -1.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _resize_chw(image: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    image = image.unsqueeze(0)
    kwargs: dict[str, object] = {"size": (image_size, image_size), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(image, **kwargs)[0]


def _normalize_satellite_bands(sat: np.ndarray) -> torch.Tensor:
    sat = sat.astype(np.float32)
    out = np.zeros_like(sat, dtype=np.float32)
    for channel in range(sat.shape[0]):
        band = sat[channel]
        valid = band[band > 0]
        if valid.size == 0:
            continue
        lo = np.percentile(valid, 1)
        hi = np.percentile(valid, 99)
        if hi <= lo:
            hi, lo = float(valid.max()), float(valid.min())
        if hi <= lo:
            continue
        out[channel] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return torch.from_numpy(out)


class _HybridBase:
    """共享的发现 / 读取逻辑。"""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 224,
        locations: list[str] | None = None,
        timepoints: list[str] | None = None,
        csv_file: str | Path | None = None,
        sat_image_size: int | None = None,
        uav_image_size: int | None = None,
    ):
        if rasterio is None:
            raise ImportError("rasterio is required for reading Satellite .TIF files.")
        self.root_dir = Path(root_dir)
        # 卫星与无人机可用不同分辨率（默认都回退到 image_size）。
        self.sat_image_size = sat_image_size if sat_image_size is not None else image_size
        self.uav_image_size = uav_image_size if uav_image_size is not None else image_size
        self.csv_file = Path(csv_file) if csv_file else self.root_dir / "GroundTruth" / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"

        if locations is None or timepoints is None:
            discovered = self._discover_paired_splits()
            if locations is None:
                locations = sorted({loc for loc, _ in discovered})
            if timepoints is None:
                timepoints = sorted({tp for _, tp in discovered})
        self.locations = set(locations)
        self.timepoints = list(timepoints)

    def _discover_paired_splits(self) -> set[tuple[str, str]]:
        satellite_root = self.root_dir / "Satellite"
        uav_root = self.root_dir / "UAV"
        pairs: set[tuple[str, str]] = set()
        if not satellite_root.exists() or not uav_root.exists():
            return pairs
        for location_dir in satellite_root.iterdir():
            if not location_dir.is_dir():
                continue
            uav_location = uav_root / location_dir.name
            if not uav_location.exists():
                continue
            sat_tps = {p.name for p in location_dir.iterdir() if p.is_dir()}
            uav_tps = {p.name for p in uav_location.iterdir() if p.is_dir()}
            for timepoint in sat_tps & uav_tps:
                pairs.add((location_dir.name, timepoint))
        return pairs

    def _nitrogen_class_map(self, df: pd.DataFrame) -> dict[float, int]:
        values = sorted(float(v) for v in df["poundsOfNitrogenPerAcre"].dropna().unique())
        return {value: idx for idx, value in enumerate(values)}

    def _build_records(self, max_samples: int | None) -> list[PlotRecord]:
        df = pd.read_csv(self.csv_file)
        df = df.dropna(subset=["location", "experiment", "range", "row"])
        nitrogen_map = self._nitrogen_class_map(df)
        records: list[PlotRecord] = []
        for _, row in df.iterrows():
            location = str(row["location"])
            if location not in self.locations:
                continue
            experiment = _int_string(row["experiment"])
            experiment = EXPERIMENT_ALIASES.get(experiment, experiment)
            range_no = _int_string(row["range"])
            row_no = _int_string(row["row"])
            nitrogen_value = _safe_float(row.get("poundsOfNitrogenPerAcre"), 0.0)
            nitrogen_class = nitrogen_map.get(nitrogen_value, 0)
            for timepoint in self.timepoints:
                base = f"{location}-{timepoint}-{experiment}_{range_no}_{row_no}"
                sat_path = self.root_dir / "Satellite" / location / timepoint / f"{base}.TIF"
                uav_path = self.root_dir / "UAV" / location / timepoint / f"{base}.PNG"
                if not sat_path.exists() or not uav_path.exists():
                    continue
                records.append(
                    PlotRecord(
                        location=location,
                        timepoint=timepoint,
                        experiment=experiment,
                        range_no=range_no,
                        row_no=row_no,
                        sat_path=sat_path,
                        uav_path=uav_path,
                        yield_value=_safe_float(row.get("yieldPerAcre")),
                        nitrogen_value=nitrogen_value,
                        nitrogen_class=nitrogen_class,
                        days_to_anthesis=_safe_float(row.get("daysToAnthesis")),
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    return records
        return records

    def _read_satellite(self, path: Path) -> torch.Tensor:
        with rasterio.open(path, "r") as src:
            sat_all = src.read().astype(np.float32)
        sat = _normalize_satellite_bands(sat_all)
        sat = _resize_chw(sat, self.sat_image_size, mode="bilinear").float()
        # 保证恰好 6 波段（个别文件可能多/少波段时裁剪或零填充）。
        if sat.shape[0] >= SAT_CHANNELS:
            sat = sat[:SAT_CHANNELS]
        else:
            pad = torch.zeros(SAT_CHANNELS - sat.shape[0], *sat.shape[1:])
            sat = torch.cat([sat, pad], dim=0)
        return sat

    def _read_uav(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return _resize_chw(tensor, self.uav_image_size, mode="bilinear").float()


class HybridPlotDataset(_HybridBase, Dataset):
    """单时相样本：每条记录是一个 (plot, timepoint)。"""

    def __init__(self, root_dir, image_size=224, locations=None, timepoints=None, csv_file=None, max_samples=None, sat_image_size=None, uav_image_size=None):
        _HybridBase.__init__(self, root_dir, image_size, locations, timepoints, csv_file, sat_image_size, uav_image_size)
        self.records = self._build_records(max_samples=max_samples)
        if not self.records:
            raise ValueError("No paired Satellite/UAV records found. Check root_dir/locations/timepoints.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        r = self.records[index]
        return {
            "sat_all": self._read_satellite(r.sat_path),
            "uav_rgb": self._read_uav(r.uav_path),
            "yield": torch.tensor(r.yield_value, dtype=torch.float32),
            "nitrogen": torch.tensor(r.nitrogen_class, dtype=torch.long),
            "nitrogen_value": torch.tensor(r.nitrogen_value, dtype=torch.float32),
            "days_to_anthesis": torch.tensor(r.days_to_anthesis, dtype=torch.float32),
            "plot_id": r.plot_id,
            "location": r.location,
            "timepoint": r.timepoint,
        }


class HybridPlotSeriesDataset(_HybridBase, Dataset):
    """多时相样本：按 plot 聚合 ``self.timepoints`` 个时相，缺失时相用零填充并标记 valid=False。"""

    def __init__(self, root_dir, image_size=224, locations=None, timepoints=None, csv_file=None, max_plots=None, sat_image_size=None, uav_image_size=None):
        _HybridBase.__init__(self, root_dir, image_size, locations, timepoints, csv_file, sat_image_size, uav_image_size)
        records = self._build_records(max_samples=None)
        self.tp_order = list(self.timepoints)
        groups: dict[str, dict[str, PlotRecord]] = {}
        for r in records:
            groups.setdefault(r.plot_id, {})[r.timepoint] = r
        self.plot_ids = list(groups.keys())
        if max_plots is not None:
            # 打乱后再取，避免 smoke 小样本恰好全落在缺 yield 的数据段。
            random.Random(0).shuffle(self.plot_ids)
            self.plot_ids = self.plot_ids[:max_plots]
        self.groups = groups
        if not self.plot_ids:
            raise ValueError("No plots found for series aggregation.")

    def __len__(self) -> int:
        return len(self.plot_ids)

    def _plot_label(self, by_tp: dict[str, PlotRecord]) -> PlotRecord:
        # 同一 plot 各时相 label 相同；取任一存在的时相。
        return next(iter(by_tp.values()))

    def __getitem__(self, index: int) -> dict[str, object]:
        plot_id = self.plot_ids[index]
        by_tp = self.groups[plot_id]
        sat_seq, uav_seq, valid = [], [], []
        for tp in self.tp_order:
            r = by_tp.get(tp)
            if r is None:
                sat_seq.append(torch.zeros(SAT_CHANNELS, self.sat_image_size, self.sat_image_size))
                uav_seq.append(torch.zeros(UAV_CHANNELS, self.uav_image_size, self.uav_image_size))
                valid.append(False)
            else:
                sat_seq.append(self._read_satellite(r.sat_path))
                uav_seq.append(self._read_uav(r.uav_path))
                valid.append(True)
        ref = self._plot_label(by_tp)
        return {
            "sat_seq": torch.stack(sat_seq, dim=0),   # [T, 6, H, W]
            "uav_seq": torch.stack(uav_seq, dim=0),   # [T, 3, H, W]
            "valid": torch.tensor(valid, dtype=torch.bool),
            "yield": torch.tensor(ref.yield_value, dtype=torch.float32),
            "nitrogen": torch.tensor(ref.nitrogen_class, dtype=torch.long),
            "nitrogen_value": torch.tensor(ref.nitrogen_value, dtype=torch.float32),
            "days_to_anthesis": torch.tensor(ref.days_to_anthesis, dtype=torch.float32),
            "plot_id": plot_id,
            "location": ref.location,
        }

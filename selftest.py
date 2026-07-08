"""不依赖真实数据的形状自检：随机张量过 STCMFNet，验证端到端前向与各消融路径。

    python selftest.py
"""

from __future__ import annotations

import torch

from models.stcmf_net import STCMFNet


def _model():
    # base=16 让自检快一些（dim=64, heads=4 整除）。
    return STCMFNet(base=16, num_timepoints=3, sat_size=224, uav_size=448, heads=4)


def test_full_fusion() -> None:
    model = _model()
    sat = torch.rand(2, 3, 6, 224, 224)
    uav = torch.rand(2, 3, 3, 448, 448)
    valid = torch.ones(2, 3, dtype=torch.bool)
    y = model(sat, uav, valid, use_sat=True, use_uav=True, temporal=True, crossmodal=True)
    assert y.shape == (2,), y.shape
    print("[ok] full: 时序 + 跨网格(196 query × 784 kv) cross-attention →", tuple(y.shape))


def test_missing_timepoint() -> None:
    model = _model()
    sat = torch.rand(2, 3, 6, 224, 224)
    uav = torch.rand(2, 3, 3, 448, 448)
    valid = torch.tensor([[True, True, True], [True, True, False]])  # 第2个样本缺 TP3
    y = model(sat, uav, valid, temporal=True, crossmodal=True)
    assert y.shape == (2,)
    print("[ok] 缺时相 mask（valid=[...,False]）正常前向 →", tuple(y.shape))


def test_ablations() -> None:
    model = _model()
    sat = torch.rand(2, 3, 6, 224, 224)
    uav = torch.rand(2, 3, 3, 448, 448)
    valid = torch.ones(2, 3, dtype=torch.bool)
    configs = [
        ("baseline 无时序无跨模态", dict(temporal=False, crossmodal=False)),
        ("+时序", dict(temporal=True, crossmodal=False)),
        ("+跨模态融合", dict(temporal=True, crossmodal=True)),
    ]
    for name, kw in configs:
        y = model(sat, uav, valid, use_sat=True, use_uav=True, **kw)
        assert y.shape == (2,)
        print(f"[ok] 消融 {name} →", tuple(y.shape))


def test_single_modality() -> None:
    model = _model()
    sat = torch.rand(2, 3, 6, 224, 224)
    uav = torch.rand(2, 3, 3, 448, 448)
    valid = torch.ones(2, 3, dtype=torch.bool)
    ys = model(sat, None, valid, use_sat=True, use_uav=False)
    yu = model(None, uav, valid, use_sat=False, use_uav=True)
    assert ys.shape == (2,) and yu.shape == (2,)
    print("[ok] 单模态消融 sat_ms / real_uav →", tuple(ys.shape), tuple(yu.shape))


if __name__ == "__main__":
    torch.manual_seed(0)
    test_full_fusion()
    test_missing_timepoint()
    test_ablations()
    test_single_modality()
    print("\nAll STCMFNet shape self-tests passed.")

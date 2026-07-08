"""STCMF v2: 端到端时空跨模态产量预测模型（从头训练，只输出 yield）。

数据流（一个 plot）：
    sat_seq [B,T,6,224,224]  +  uav_seq [B,T,3,448,448]  +  valid [B,T]
      │ Stage A 共享轻量 backbone(base32, stride16, 不GAP)
      │   sat → [B,C,14,14] (196 token, 光谱)
      │   uav → [B,C,28,28] (784 token, 高分辨率空间细节)
      │ Stage B 模态内时序(per-modality, 各自分辨率) → sat_tok[B,196,C] / uav_tok[B,784,C]
      │ Stage C 跨模态融合: 卫星光谱 query 无人机高分辨率细节 → [B,196,C]
      │ Stage D AttnPool → z → RegHead
      ▼ yield [B]

设计要点见 plan：不过早 GAP、模态内独立时序、base32 起步、cross-attention 不砍 UAV 分辨率。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SAT_CHANNELS = 6
UAV_CHANNELS = 3


# --------------------------------------------------------------------------- backbone
class BasicBlock(nn.Module):
    def __init__(self, inp: int, out: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(out)
        self.c2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(out)
        self.down = None
        if stride != 1 or inp != out:
            self.down = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        return F.relu(o + idt, inplace=True)


class LightCNNFeat(nn.Module):
    """从头轻量 ResNet，输出 stride16 的 feature map（不做 GAP）。224→14，448→28。"""

    def __init__(self, in_chans: int, base: int = 32, layers: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self._make(base, base, layers[0], 1)
        self.layer2 = self._make(base, base * 2, layers[1], 2)
        self.layer3 = self._make(base * 2, base * 4, layers[2], 2)
        self.out_dim = base * 4

    @staticmethod
    def _make(inp: int, out: int, n: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(inp, out, stride)] + [BasicBlock(out, out, 1) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B, base*4, H/16, W/16]


# --------------------------------------------------------------------------- 注意力组件
class AttnPool(nn.Module):
    """可学习 query 对一组 token 做注意力聚合。"""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class TemporalEncoder(nn.Module):
    """Stage B：模态内时序。对每个空间位置跨时相做 attention，再 AttnPool 掉 T。"""

    def __init__(self, dim: int, hw: int, num_timepoints: int, heads: int = 4, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, hw, dim))
        self.time = nn.Parameter(torch.zeros(1, num_timepoints, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.pool = AttnPool(dim, heads)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.time, std=0.02)

    def forward(self, fmap_seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # fmap_seq: [B,T,C,h,w]; valid: [B,T] -> 返回 [B, hw, C]
        b, t, c, h, w = fmap_seq.shape
        hw = h * w
        x = fmap_seq.permute(0, 1, 3, 4, 2).reshape(b, t, hw, c)          # [B,T,hw,C]
        x = x + self.pos.unsqueeze(1) + self.time.unsqueeze(2)            # 加 空间pos + 时间emb
        x = x.permute(0, 2, 1, 3).reshape(b * hw, t, c)                   # [B*hw, T, C]
        mask = (~valid).unsqueeze(1).expand(b, hw, t).reshape(b * hw, t)  # True=padding
        x = self.tr(x, src_key_padding_mask=mask)
        pooled = self.pool(x, key_padding_mask=mask)                      # [B*hw, C]
        return pooled.reshape(b, hw, c)


class CrossBlock(nn.Module):
    """Stage C：卫星光谱 token 作 query，attend 无人机高分辨率 token。"""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.nq(q), self.nkv(kv), self.nkv(kv))
        q = q + h
        return q + self.ff(self.norm2(q))


# --------------------------------------------------------------------------- 主模型
class STCMFNet(nn.Module):
    def __init__(
        self,
        base: int = 32,
        num_timepoints: int = 3,
        sat_size: int = 224,
        uav_size: int = 448,
        heads: int = 4,
        temporal_layers: int = 1,
        dropout: float = 0.1,
        bidirectional_cross: bool = False,
    ):
        super().__init__()
        self.num_timepoints = num_timepoints
        self.bidirectional_cross = bidirectional_cross
        dim = base * 4
        self.dim = dim

        self.sat_backbone = LightCNNFeat(SAT_CHANNELS, base)
        self.uav_backbone = LightCNNFeat(UAV_CHANNELS, base)
        sat_hw = (sat_size // 16) ** 2  # 224 -> 14x14 = 196
        uav_hw = (uav_size // 16) ** 2  # 448 -> 28x28 = 784

        self.sat_temporal = TemporalEncoder(dim, sat_hw, num_timepoints, heads, temporal_layers, dropout)
        self.uav_temporal = TemporalEncoder(dim, uav_hw, num_timepoints, heads, temporal_layers, dropout)
        self.mod_emb = nn.Parameter(torch.zeros(1, 2, dim))

        self.cross = CrossBlock(dim, heads, dropout)
        self.cross_rev = CrossBlock(dim, heads, dropout) if bidirectional_cross else None

        self.pool = AttnPool(dim, heads)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        nn.init.normal_(self.mod_emb, std=0.02)

    def _backbone_seq(self, seq: torch.Tensor, backbone: LightCNNFeat) -> torch.Tensor:
        # seq: [B,T,Cin,H,W] -> [B,T,C,h,w]，时相间共享 backbone
        b, t = seq.shape[:2]
        flat = seq.reshape(b * t, *seq.shape[2:])
        f = backbone(flat)
        return f.reshape(b, t, *f.shape[1:])

    @staticmethod
    def _masked_mean_time(fmap_seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # 无时序基线：对有效时相取均值 -> [B, hw, C]
        b, t, c, h, w = fmap_seq.shape
        x = fmap_seq.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
        m = valid.float().view(b, t, 1, 1)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

    def _aggregate_time(self, fmap_seq, valid, encoder, temporal: bool) -> torch.Tensor:
        return encoder(fmap_seq, valid) if temporal else self._masked_mean_time(fmap_seq, valid)

    def forward(
        self,
        sat_seq: torch.Tensor | None,
        uav_seq: torch.Tensor | None,
        valid: torch.Tensor | None = None,
        use_sat: bool = True,
        use_uav: bool = True,
        temporal: bool = False,   # 默认基线：三时相平均（ablation 证明 Stage B attention 无增益）
        crossmodal: bool = False,  # 默认基线：双模态池化平均（Stage C cross-attention 无增益）
    ) -> torch.Tensor:
        ref = sat_seq if use_sat else uav_seq
        b, t = ref.shape[:2]
        if valid is None:
            valid = torch.ones(b, t, dtype=torch.bool, device=ref.device)

        sat_tok = uav_tok = None
        if use_sat:
            sat_f = self._backbone_seq(sat_seq, self.sat_backbone)               # [B,T,C,14,14]
            sat_tok = self._aggregate_time(sat_f, valid, self.sat_temporal, temporal) + self.mod_emb[:, 0]
        if use_uav:
            uav_f = self._backbone_seq(uav_seq, self.uav_backbone)               # [B,T,C,28,28]
            uav_tok = self._aggregate_time(uav_f, valid, self.uav_temporal, temporal) + self.mod_emb[:, 1]

        if use_sat and use_uav:
            if crossmodal:
                fused = self.cross(sat_tok, uav_tok)                             # [B,196,C]
                if self.cross_rev is not None:
                    rev = self.cross_rev(uav_tok, sat_tok)
                    z = 0.5 * (self.pool(fused) + self.pool(rev))
                else:
                    z = self.pool(fused)
            else:
                z = 0.5 * (self.pool(sat_tok) + self.pool(uav_tok))
        elif use_sat:
            z = self.pool(sat_tok)
        else:
            z = self.pool(uav_tok)

        return self.head(z).squeeze(-1)

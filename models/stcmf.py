"""STCMF: Spatio-Temporal Cross-Modal Fusion model.

原生支持 6 波段卫星 + 3 波段 UAV。包含三部分，共享同一个编码器主干：

- ``STCMFEncoder``     : patch embed + 位置/模态(/时间)编码 + Transformer encoder。
- ``STCMFForPretrain`` : encoder + decoder + 双重建头，做跨模态 masked autoencoder 预训练。
- ``STCMFForTraits``   : encoder + 性状头(yield / nitrogen / flowering)，下游监督融合。

设计借鉴自旧项目 ``crossmodal_crop_canopy_repro/models/mm_mae.py``，
最关键的区别：卫星分支是原生 6 通道，而不是只用 RGB 3 波段。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

SAT_CHANNELS = 6
UAV_CHANNELS = 3


# vit_base 对齐旧项目；small 用于 smoke。
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "small": {
        "embed_dim": 96,
        "encoder_depth": 1,
        "encoder_heads": 4,
        "decoder_dim": 128,
        "decoder_depth": 1,
        "decoder_heads": 4,
        "visible_tokens": 16,
    },
    "base": {
        "embed_dim": 384,
        "encoder_depth": 6,
        "encoder_heads": 6,
        "decoder_dim": 256,
        "decoder_depth": 2,
        "decoder_heads": 4,
        "visible_tokens": 66,
    },
    "vit_base": {
        "embed_dim": 768,
        "encoder_depth": 12,
        "encoder_heads": 12,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 8,
        "visible_tokens": 66,
    },
}


def preset_config(name: str) -> dict[str, int]:
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Available: {', '.join(MODEL_PRESETS)}")
    return dict(MODEL_PRESETS[name])


@dataclass
class MaskInfo:
    sat_visible: torch.Tensor
    uav_visible: torch.Tensor
    sat_masked: torch.Tensor
    uav_masked: torch.Tensor


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    assert h == w and h % patch_size == 0
    grid = h // patch_size
    x = x.reshape(b, c, grid, patch_size, grid, patch_size)
    x = torch.einsum("bcphqw->bpqhwc", x)
    return x.reshape(b, grid * grid, patch_size * patch_size * c)


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    b, n, _ = patches.shape
    grid = int(math.sqrt(n))
    assert grid * grid == n
    x = patches.reshape(b, grid, grid, patch_size, patch_size, channels)
    x = torch.einsum("bpqhwc->bcphqw", x)
    return x.reshape(b, channels, grid * patch_size, grid * patch_size)


def masked_mse_loss(pred_patches: torch.Tensor, target_image: torch.Tensor, masked: torch.Tensor, patch_size: int) -> torch.Tensor:
    target = patchify(target_image, patch_size)
    loss = ((pred_patches - target) ** 2).mean(dim=-1)
    return (loss * masked.float()).sum() / masked.float().sum().clamp_min(1.0)


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class STCMFEncoder(nn.Module):
    """共享编码器：原生 6 波段卫星 + 3 波段 UAV 的跨模态 Transformer encoder。

    支持单时相（默认）与多时相（``num_timepoints>1`` 时叠加时间编码）。
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        visible_tokens: int = 66,
        num_timepoints: int = 1,
        alpha_sat: float = 4.0,
        alpha_uav: float = 1.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.visible_tokens = visible_tokens
        self.num_timepoints = num_timepoints
        self.alpha_sat = alpha_sat
        self.alpha_uav = alpha_uav

        self.sat_embed = PatchEmbed(image_size, patch_size, SAT_CHANNELS, embed_dim)
        self.uav_embed = PatchEmbed(image_size, patch_size, UAV_CHANNELS, embed_dim)
        self.num_patches = self.sat_embed.num_patches

        self.sat_pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.uav_pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.sat_modality = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.uav_modality = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 时间编码：阶段2 多时相时使用；单时相时退化为 0 号。
        self.temporal = nn.Parameter(torch.zeros(1, max(num_timepoints, 1), embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self) -> None:
        for p in [self.sat_pos, self.uav_pos, self.sat_modality, self.uav_modality, self.temporal]:
            nn.init.normal_(p, std=0.02)
        self.apply(_init_module)

    # ---- 掩码采样（预训练用，借鉴旧 mm_mae） -------------------------------
    def sample_visible_masks(self, batch_size: int, device: torch.device) -> MaskInfo:
        alpha = torch.tensor([self.alpha_sat, self.alpha_uav], device=device)
        proportions = torch.distributions.Dirichlet(alpha).sample((batch_size,))
        sat_counts = torch.clamp((proportions[:, 0] * self.visible_tokens).round().long(), 1, self.visible_tokens - 1)

        sat_visible = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        uav_visible = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        for i in range(batch_size):
            sat_perm = torch.randperm(self.num_patches, device=device)
            uav_perm = torch.randperm(self.num_patches, device=device)
            sat_visible[i, sat_perm[: sat_counts[i]]] = True
            uav_visible[i, uav_perm[: self.visible_tokens - sat_counts[i]]] = True
        return MaskInfo(sat_visible, uav_visible, ~sat_visible, ~uav_visible)

    def satellite_to_uav_masks(self, batch_size: int, device: torch.device) -> MaskInfo:
        sat_visible = torch.ones(batch_size, self.num_patches, dtype=torch.bool, device=device)
        uav_visible = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        return MaskInfo(sat_visible, uav_visible, ~sat_visible, ~uav_visible)

    def make_masks(self, batch_size: int, device: torch.device, train_mode: str, satellite_to_uav_prob: float = 0.5) -> MaskInfo:
        if train_mode == "mixed_mask":
            return self.sample_visible_masks(batch_size, device)
        if train_mode == "satellite_to_uav":
            return self.satellite_to_uav_masks(batch_size, device)
        if train_mode == "hybrid":
            if torch.rand((), device=device).item() < satellite_to_uav_prob:
                return self.satellite_to_uav_masks(batch_size, device)
            return self.sample_visible_masks(batch_size, device)
        raise ValueError("train_mode must be one of: mixed_mask, satellite_to_uav, hybrid")

    def embed_tokens(self, sat_all: torch.Tensor, uav_rgb: torch.Tensor, timepoint: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.temporal[:, timepoint : timepoint + 1]
        sat_tokens = self.sat_embed(sat_all) + self.sat_pos + self.sat_modality + t
        uav_tokens = self.uav_embed(uav_rgb) + self.uav_pos + self.uav_modality + t
        return sat_tokens, uav_tokens

    # ---- 预训练编码：按 mask 选可见 token（每样本可见数不同，逐样本处理） ----
    def encode_visible(self, sat_all: torch.Tensor, uav_rgb: torch.Tensor, mask_info: MaskInfo) -> tuple[torch.Tensor, list[int], list[int]]:
        b = sat_all.shape[0]
        sat_tokens, uav_tokens = self.embed_tokens(sat_all, uav_rgb)
        visible, sat_counts, uav_counts = [], [], []
        for i in range(b):
            sat_i = sat_tokens[i, mask_info.sat_visible[i]]
            uav_i = uav_tokens[i, mask_info.uav_visible[i]]
            sat_counts.append(int(sat_i.shape[0]))
            uav_counts.append(int(uav_i.shape[0]))
            visible.append(torch.cat([sat_i, uav_i], dim=0))
        encoded = self.encoder_norm(self.encoder(torch.stack(visible, dim=0)))
        return encoded, sat_counts, uav_counts

    # ---- 下游融合编码：全 token 可见，规则形状，可批量处理；支持单模态消融 ----
    def encode_fused(self, sat_all: torch.Tensor | None, uav_rgb: torch.Tensor | None, use_sat: bool = True, use_uav: bool = True) -> torch.Tensor:
        tokens = []
        if use_sat:
            assert sat_all is not None
            tokens.append(self.sat_embed(sat_all) + self.sat_pos + self.sat_modality + self.temporal[:, 0:1])
        if use_uav:
            assert uav_rgb is not None
            tokens.append(self.uav_embed(uav_rgb) + self.uav_pos + self.uav_modality + self.temporal[:, 0:1])
        if not tokens:
            raise ValueError("encode_fused needs at least one modality")
        x = self.encoder_norm(self.encoder(torch.cat(tokens, dim=1)))
        return x.mean(dim=1)

    # ---- 阶段2：多时相融合编码（plot 的 T 个时相，按时间编码后联合 attention） ----
    def encode_fused_temporal(self, sat_seq: torch.Tensor, uav_seq: torch.Tensor, valid: torch.Tensor | None = None, use_sat: bool = True, use_uav: bool = True) -> torch.Tensor:
        """sat_seq: [B, T, 6, H, W]; uav_seq: [B, T, 3, H, W]; valid: [B, T] bool。

        把所有 (模态, 时相) 的 patch token 拼成一条长序列做联合 attention，再 mean pool。
        """
        b, t = sat_seq.shape[:2]
        seqs = []
        key_mask = []  # True = padding（缺时相），供 encoder 忽略
        for ti in range(t):
            t_emb = self.temporal[:, min(ti, self.temporal.shape[1] - 1)].unsqueeze(1)
            if use_sat:
                s = self.sat_embed(sat_seq[:, ti]) + self.sat_pos + self.sat_modality + t_emb
                seqs.append(s)
            if use_uav:
                u = self.uav_embed(uav_seq[:, ti]) + self.uav_pos + self.uav_modality + t_emb
                seqs.append(u)
            if valid is not None:
                pad = (~valid[:, ti]).unsqueeze(1).expand(b, self.num_patches)
                per_mod = [pad] * (int(use_sat) + int(use_uav))
                key_mask.extend(per_mod)
        x = torch.cat(seqs, dim=1)
        mask = torch.cat(key_mask, dim=1) if key_mask else None
        x = self.encoder_norm(self.encoder(x, src_key_padding_mask=mask))
        if mask is None:
            return x.mean(dim=1)
        keep = (~mask).float().unsqueeze(-1)
        return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)


class STCMFForPretrain(nn.Module):
    """阶段0：6 波段跨模态 masked autoencoder。"""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        visible_tokens: int = 66,
        num_timepoints: int = 1,
        alpha_sat: float = 4.0,
        alpha_uav: float = 1.0,
    ):
        super().__init__()
        self.encoder = STCMFEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            visible_tokens=visible_tokens,
            num_timepoints=num_timepoints,
            alpha_sat=alpha_sat,
            alpha_uav=alpha_uav,
        )
        self.patch_size = patch_size
        num_patches = self.encoder.num_patches

        self.decoder_proj = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, num_patches * 2, decoder_dim))
        self.decoder_modality = nn.Parameter(torch.zeros(1, 2, decoder_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.sat_head = nn.Linear(decoder_dim, patch_size * patch_size * SAT_CHANNELS)
        self.uav_head = nn.Linear(decoder_dim, patch_size * patch_size * UAV_CHANNELS)

        for p in [self.mask_token, self.decoder_pos, self.decoder_modality]:
            nn.init.normal_(p, std=0.02)
        self.decoder.apply(_init_module)
        _init_module(self.decoder_proj)
        _init_module(self.sat_head)
        _init_module(self.uav_head)

    @property
    def num_patches(self) -> int:
        return self.encoder.num_patches

    def make_masks(self, batch_size: int, device: torch.device, train_mode: str, satellite_to_uav_prob: float = 0.5) -> MaskInfo:
        return self.encoder.make_masks(batch_size, device, train_mode, satellite_to_uav_prob)

    def forward(self, sat_all: torch.Tensor, uav_rgb: torch.Tensor, mask_info: MaskInfo | None = None) -> dict[str, torch.Tensor]:
        b = sat_all.shape[0]
        device = sat_all.device
        if mask_info is None:
            mask_info = self.encoder.sample_visible_masks(b, device)
        n = self.num_patches

        encoded, sat_counts, uav_counts = self.encoder.encode_visible(sat_all, uav_rgb, mask_info)

        full = self.mask_token.repeat(b, n * 2, 1)
        for i in range(b):
            sc, uc = sat_counts[i], uav_counts[i]
            sat_enc = self.decoder_proj(encoded[i, :sc])
            uav_enc = self.decoder_proj(encoded[i, sc : sc + uc])
            sat_full = full[i, :n]
            uav_full = full[i, n:]
            sat_full[mask_info.sat_visible[i]] = sat_enc
            uav_full[mask_info.uav_visible[i]] = uav_enc
            full[i, :n] = sat_full
            full[i, n:] = uav_full

        modality = torch.cat(
            [
                self.decoder_modality[:, :1].expand(b, n, -1),
                self.decoder_modality[:, 1:].expand(b, n, -1),
            ],
            dim=1,
        )
        decoded = self.decoder_norm(self.decoder(full + self.decoder_pos + modality))
        sat_pred = torch.sigmoid(self.sat_head(decoded[:, :n]))
        uav_pred = torch.sigmoid(self.uav_head(decoded[:, n:]))
        return {
            "sat_pred_patches": sat_pred,
            "uav_pred_patches": uav_pred,
            "sat_recon": unpatchify(sat_pred, self.patch_size, SAT_CHANNELS),
            "uav_recon": unpatchify(uav_pred, self.patch_size, UAV_CHANNELS),
            "sat_masked": mask_info.sat_masked,
            "uav_masked": mask_info.uav_masked,
        }


class STCMFForTraits(nn.Module):
    """阶段1起：监督融合产量/氮/开花期预测。下游不 mask，两模态全 token。"""

    def __init__(self, encoder: STCMFEncoder, embed_dim: int, num_classes: int, with_flowering: bool = False):
        super().__init__()
        self.encoder = encoder
        self.with_flowering = with_flowering
        self.yield_head = self._mlp_head(embed_dim, 1)
        self.nitrogen_head = self._mlp_head(embed_dim, num_classes)
        self.flowering_head = self._mlp_head(embed_dim, 1) if with_flowering else None

    @staticmethod
    def _mlp_head(embed_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, out_dim),
        )

    def forward(
        self,
        sat_all: torch.Tensor | None,
        uav_rgb: torch.Tensor | None,
        use_sat: bool = True,
        use_uav: bool = True,
        temporal: bool = False,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if temporal:
            feat = self.encoder.encode_fused_temporal(sat_all, uav_rgb, valid=valid, use_sat=use_sat, use_uav=use_uav)
        else:
            feat = self.encoder.encode_fused(sat_all, uav_rgb, use_sat=use_sat, use_uav=use_uav)
        out = {
            "yield": self.yield_head(feat).squeeze(-1),
            "nitrogen": self.nitrogen_head(feat),
        }
        if self.flowering_head is not None:
            out["flowering"] = self.flowering_head(feat).squeeze(-1)
        return out

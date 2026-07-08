# STCMFNet v2 基线模型 — 完整前向传播过程

> 源文件：[stcmf_net.py](models/stcmf_net.py)
>
> 本文档追踪的是**基线配置**（`temporal=False, crossmodal=False`），即最终定型的最优模型。
> Stage B（时序 Attention）和 Stage C（跨模态 Cross-Attention）经消融实验证明无增益，默认关闭。

---

## 全局符号约定

| 符号 | 默认值 | 含义 |
|------|--------|------|
| $B$ | 8 | 批大小 |
| $T$ | 3 | 时相数（TP1 / TP2 / TP3） |
| $C$ | 128 | 特征通道数（`base × 4 = 32 × 4`） |
| $H_s, W_s$ | 224 | 卫星输入空间尺寸 |
| $H_u, W_u$ | 448 | 无人机输入空间尺寸 |

---

## 端到端数据流总览

```
输入
  sat_seq  [B, 3, 6, 224, 224]     ─┐
  uav_seq  [B, 3, 3, 448, 448]     ─┤
  valid    [B, 3]  (bool)           ─┘
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          │            Stage A: 双模态 Backbone                  │
          │  sat: [B,3,6,224,224] ──→ [B,3,128,14,14]           │
          │  uav: [B,3,3,448,448] ──→ [B,3,128,28,28]          │
          └──────────────┬──────────────────────┬───────────────┘
                         │                      │
          ┌──────────────▼──────────┐  ┌────────▼──────────────┐
          │ 时相均值聚合 + 模态编码  │  │ 时相均值聚合 + 模态编码 │
          │ sat_tok [B, 196, 128]   │  │ uav_tok [B, 784, 128] │
          └──────────────┬──────────┘  └────────┬──────────────┘
                         │                      │
          ┌──────────────▼──────────┐  ┌────────▼──────────────┐
          │   AttnPool → z_sat      │  │   AttnPool → z_uav    │
          │       [B, 128]          │  │       [B, 128]         │
          └──────────────┬──────────┘  └────────┬──────────────┘
                         │                      │
                         └──────────┬───────────┘
                                    │
                      z = 0.5 × (z_sat + z_uav)
                                [B, 128]
                                    │
                         ┌──────────▼──────────┐
                         │    回归预测头 Head    │
                         │  [B,128] → [B,1]     │
                         └──────────┬──────────┘
                                    │
                              yield_pred [B]
```

---

## Stage A: 双模态 Backbone 特征提取

> 代码位置：[LightCNNFeat](models/stcmf_net.py#L45-L71) + [_backbone_seq](models/stcmf_net.py#L173-L178)

卫星和无人机各有**独立的 Backbone 实例**（`sat_backbone` 和 `uav_backbone`），但结构完全相同，都是一个轻量 ResNet。同一模态的**不同时相共享同一个 Backbone 权重**。

### A.1 时相展平

先将时间维度折叠进 batch 维度，以便统一过 Backbone：

```
sat_seq [B, T, 6, 224, 224]  →  reshape  →  [B×T, 6, 224, 224]
uav_seq [B, T, 3, 448, 448]  →  reshape  →  [B×T, 3, 448, 448]
```

### A.2 Backbone 逐层维度变化（以卫星为例，`base=32`）

[LightCNNFeat](models/stcmf_net.py#L45-L71) 由 Stem + 三个 ResNet Stage 组成，总下采样倍率 = 16。

#### Stem

| 操作 | 输出尺寸 |
|------|----------|
| 输入 | `[B×T, 6, 224, 224]` |
| `Conv2d(6, 32, kernel=7, stride=2, pad=3)` | `[B×T, 32, 112, 112]` |
| `BatchNorm2d(32)` + `ReLU` | `[B×T, 32, 112, 112]` |
| `MaxPool2d(kernel=3, stride=2, pad=1)` | `[B×T, 32, 56, 56]` |

#### Layer 1（2 × [BasicBlock](models/stcmf_net.py#L27-L42), stride=1）

| 操作 | 输出尺寸 |
|------|----------|
| `BasicBlock(32 → 32, stride=1)` × 2 | `[B×T, 32, 56, 56]` |

> 通道不变、空间不变，纯粹增加非线性深度。

#### Layer 2（2 × BasicBlock, stride=2）

| 操作 | 输出尺寸 |
|------|----------|
| `BasicBlock(32 → 64, stride=2)` | `[B×T, 64, 28, 28]` |
| `BasicBlock(64 → 64, stride=1)` | `[B×T, 64, 28, 28]` |

> 通道翻倍 32→64，空间减半 56→28。第一个 Block 含 downsample 捷径（1×1 Conv stride=2）。

#### Layer 3（2 × BasicBlock, stride=2）

| 操作 | 输出尺寸 |
|------|----------|
| `BasicBlock(64 → 128, stride=2)` | `[B×T, 128, 14, 14]` |
| `BasicBlock(128 → 128, stride=1)` | `[B×T, 128, 14, 14]` |

> 通道再翻倍 64→128，空间再减半 28→14。

#### BasicBlock 内部结构（残差块）

每个 [BasicBlock](models/stcmf_net.py#L27-L42) 的内部流程：

```
input x ──────────────────────────────────┐
    │                                     │ (identity / downsample)
    ├─→ Conv2d(3×3, stride) → BN → ReLU  │
    ├─→ Conv2d(3×3, stride=1) → BN       │
    │                                     │
    └─→ output = ReLU(conv_out + shortcut)┘
```

> [!IMPORTANT]
> **关键设计：不做 GAP（全局平均池化）**。Backbone 输出的是完整的空间 feature map，而非压缩成单一向量。这保留了空间结构信息，供后续的注意力池化使用。

### A.3 Backbone 输出汇总

```
卫星: [B×T, 128, 14, 14]  →  reshape  →  [B, T, 128, 14, 14]
无人机: [B×T, 128, 28, 28]  →  reshape  →  [B, T, 128, 28, 28]
```

| 模态 | 输入 | Backbone 输出 | 空间 Token 数 |
|------|------|---------------|---------------|
| 卫星 (6 波段) | `[B, T, 6, 224, 224]` | `[B, T, 128, 14, 14]` | 196 |
| 无人机 (RGB) | `[B, T, 3, 448, 448]` | `[B, T, 128, 28, 28]` | 784 |

> [!NOTE]
> 无人机输入分辨率 448 是卫星 224 的两倍，经过相同 stride=16 的 Backbone 后，空间分辨率保持了 4 倍的差异（14² vs 28²），这正是设计意图——保留无人机的高分辨率空间细节。

---

## 时相均值聚合

> 代码位置：[_masked_mean_time](models/stcmf_net.py#L180-L186)

基线配置下，不使用时序 Transformer（Stage B 关闭），直接对有效时相取**加权均值**。

### 逐步矩阵变化（以卫星为例）

```python
# 输入: fmap_seq [B, T, C, h, w] = [B, 3, 128, 14, 14]

# Step 1: Permute + Reshape → 展平空间维度
fmap_seq.permute(0,1,3,4,2).reshape(B, T, h*w, C)
# [B, 3, 128, 14, 14] → [B, 3, 14, 14, 128] → [B, 3, 196, 128]

# Step 2: 构造有效时相掩码
m = valid.float().view(B, T, 1, 1)   # [B, 3, 1, 1]  (1.0 = 有效, 0.0 = 缺失)

# Step 3: 加权求和 / 有效时相数
result = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
# 分子: [B, 3, 196, 128] × [B, 3, 1, 1] → sum(dim=1) → [B, 196, 128]
# 分母: [B, 3, 1, 1] → sum(dim=1) → [B, 1, 1]  (值为有效时相数, 如 3)
# 结果: [B, 196, 128]
```

### 叠加模态编码

> 代码位置：[forward L209/L212](models/stcmf_net.py#L209-L212)

```
sat_tok = 时相均值结果 + mod_emb[:, 0]    # [B, 196, 128] + [1, 1, 128] → [B, 196, 128]
uav_tok = 时相均值结果 + mod_emb[:, 1]    # [B, 784, 128] + [1, 1, 128] → [B, 784, 128]
```

> `mod_emb` 是形状为 `[1, 2, 128]` 的可学习参数，通过广播机制加到每个空间 token 上，让模型能区分"这个特征来自卫星还是无人机"。

### 聚合结果

| 模态 | 聚合前 | 聚合后 |
|------|--------|--------|
| 卫星 | `[B, T, 128, 14, 14]` | `sat_tok [B, 196, 128]` |
| 无人机 | `[B, T, 128, 28, 28]` | `uav_tok [B, 784, 128]` |

---

## Stage D: 注意力池化 (AttnPool)

> 代码位置：[AttnPool](models/stcmf_net.py#L75-L87)

AttnPool 使用一个**可学习的 query 向量**，通过多头注意力机制对空间 token 序列进行"软加权聚合"，将变长的空间序列压缩为单个固定维度的向量。

### AttnPool 内部结构

```
self.q    = nn.Parameter(randn(1, 1, 128))         # 可学习 query
self.attn = nn.MultiheadAttention(128, heads=4)     # 4 头注意力
```

### 卫星分支池化过程

```
Q = self.q.expand(B, 1, 128)                        # [1, 1, 128] → [B, 1, 128]
K = V = sat_tok                                      # [B, 196, 128]

# MultiheadAttention 内部:
#   每个头的维度 d_k = 128 / 4 = 32
#
#   Q_head = Q × W_Q  →  [B, 1, 32]   (× 4 heads)
#   K_head = K × W_K  →  [B, 196, 32] (× 4 heads)
#   V_head = V × W_V  →  [B, 196, 32] (× 4 heads)
#
#   Attention = softmax(Q_head × K_head^T / √32)  →  [B, 1, 196]
#   Output_head = Attention × V_head               →  [B, 1, 32]
#
#   Concat 4 heads → [B, 1, 128]
#   × W_O → [B, 1, 128]

out.squeeze(1)  →  z_sat [B, 128]
```

### 无人机分支池化过程

```
Q = self.q.expand(B, 1, 128)                        # [B, 1, 128]  (同一个 query)
K = V = uav_tok                                      # [B, 784, 128]

# 注意力权重: softmax(Q × K^T / √32)  →  [B, 1, 784]
#   → 学到的权重告诉模型"关注 28×28 空间网格中的哪些区域"
# 加权输出:   Attention × V              →  [B, 1, 128]

out.squeeze(1)  →  z_uav [B, 128]
```

> [!TIP]
> 卫星和无人机**共享同一个 AttnPool 模块**（包括同一个可学习 query `self.q` 和同一组 W_Q / W_K / W_V / W_O 参数）。模型通过之前添加的 `mod_emb` 模态编码来区分两种输入。

---

## 双模态融合

> 代码位置：[forward L223](models/stcmf_net.py#L223)

融合方式极其简洁——等权平均：

```python
z = 0.5 * (self.pool(sat_tok) + self.pool(uav_tok))
```

```
z_sat  [B, 128]  ─┐
                   ├─→  逐元素相加 → × 0.5  →  z [B, 128]
z_uav  [B, 128]  ─┘
```

> [!NOTE]
> 消融实验证明，这种简单的平均融合（而非 Stage C 的 Cross-Attention 融合）就是最优方案。原因：2131 个样本的数据量太小，产量是全局聚合量，复杂的注意力参数缺乏足够数据支撑。

---

## 回归预测头 (Head)

> 代码位置：[head 定义](models/stcmf_net.py#L164-L170)

```python
self.head = nn.Sequential(
    nn.LayerNorm(128),
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
)
```

### 逐层维度变化

| 层 | 操作 | 输出维度 |
|----|------|----------|
| 1 | `LayerNorm(128)` — 归一化 | `[B, 128]` |
| 2 | `Linear(128 → 64)` — 降维 | `[B, 64]` |
| 3 | `GELU()` — 激活 | `[B, 64]` |
| 4 | `Dropout(0.1)` — 正则化 | `[B, 64]` |
| 5 | `Linear(64 → 1)` — 标量回归 | `[B, 1]` |
| 6 | `.squeeze(-1)` — 去掉末尾维度 | `[B]` |

最终输出 `yield_pred [B]`，每个样本一个产量预测值，训练时使用 **Huber Loss** 进行优化。

---

## 完整维度变化一览表

以默认参数 `base=32, T=3, sat_size=224, uav_size=448` 为例：

| 阶段 | 卫星分支维度 | 无人机分支维度 |
|------|-------------|---------------|
| **输入** | `[B, 3, 6, 224, 224]` | `[B, 3, 3, 448, 448]` |
| 展平时相 | `[B×3, 6, 224, 224]` | `[B×3, 3, 448, 448]` |
| Stem Conv7+Pool | `[B×3, 32, 56, 56]` | `[B×3, 32, 112, 112]` |
| Layer1 (stride=1) | `[B×3, 32, 56, 56]` | `[B×3, 32, 112, 112]` |
| Layer2 (stride=2) | `[B×3, 64, 28, 28]` | `[B×3, 64, 56, 56]` |
| Layer3 (stride=2) | `[B×3, 128, 14, 14]` | `[B×3, 128, 28, 28]` |
| 恢复时间维 | `[B, 3, 128, 14, 14]` | `[B, 3, 128, 28, 28]` |
| 时相均值聚合 | `[B, 196, 128]` | `[B, 784, 128]` |
| + 模态编码 | `[B, 196, 128]` | `[B, 784, 128]` |
| **AttnPool** | **`[B, 128]`** | **`[B, 128]`** |

| 阶段 | 融合后维度 |
|------|-----------|
| 双模态平均融合 | `[B, 128]` |
| LayerNorm | `[B, 128]` |
| Linear(128→64) + GELU + Dropout | `[B, 64]` |
| Linear(64→1) + squeeze | **`[B]`** ← 产量预测 |

---

## 损失函数

```python
loss = F.huber_loss(yield_pred, yield_target, delta=1.0)
```

使用 **Huber Loss**（而非 MSE），对异常值更鲁棒——当预测误差小于 δ=1.0 时退化为 L2，大于时退化为 L1，避免少数极端产量值主导梯度。

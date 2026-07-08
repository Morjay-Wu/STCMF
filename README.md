# STCMF — Spatio-Temporal Cross-Modal Fusion for Crop Yield Prediction

An end-to-end network that predicts plot-level crop yield directly from
**multi-temporal** dual-modal imagery — **6-band satellite** + **RGB UAV** —
using image input only, with no agronomic metadata.

A single network with a single loss jointly integrates three cues a yield
predictor needs: **temporal progression**, **spectral content**, and
**spatial resolution**. The backbone deliberately avoids early global average
pooling, preserving spatial feature maps (satellite 14×14, UAV 28×28) that are
aggregated across time points and fused by a shared attention-pooling head.

This repository accompanies the STCMF paper and releases the model, training,
and evaluation code. The architecture is documented step by step in
[docs_architecture.md](docs_architecture.md).

## Architecture

```
Input: sat_seq [B,3,6,224,224] + uav_seq [B,3,3,448,448] + valid [B,3]

Stage A  Modality-specific lightweight backbone (base 32, stride 16, no GAP)
           satellite 224 -> [B,C,14,14]  (196 spatial tokens)
           UAV       448 -> [B,C,28,28]  (784 spatial tokens)
Aggregate  Masked temporal mean over valid time points + modality embedding
Pool       Shared attention-pooling head -> z_sat, z_uav
Fuse       z = 0.5 * (z_sat + z_uav)
Head       LayerNorm -> Linear -> GELU -> Dropout -> Linear -> yield
Loss       Huber (delta = 1.0)
```

Two optional modules can be switched on for ablation: a per-modality
**temporal attention** block (`--temporal`) and a **cross-modal attention**
block (`--crossmodal`). See [docs_architecture.md](docs_architecture.md) for the
full forward pass.

## Repository layout

```
models/stcmf_net.py    STCMFNet + LightCNNFeat + TemporalEncoder + CrossBlock + AttnPool
data/hybrid_plot.py    HybridPlotSeriesDataset (multi-temporal aggregation, per-modality resolution)
train_stcmf.py         Single-stage end-to-end training (yield only, augmentation, Huber, CV modes)
selftest.py            Data-free shape self-test
inspect_data.py        Data sanity checks
plot_ablation.py       Ablation figure
plot_results.py        Results figures
models/stcmf.py        Legacy v1 cross-modal MAE model (kept for reference)
pretrain_stcmf.py      Legacy v1 MAE pretraining (kept for reference)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Shape self-test (no data required):

```bash
python selftest.py
```

Train the final model (masked-mean temporal aggregation + dual-modal pooling):

```bash
python train_stcmf.py --modalities sat_ms+real_uav --folds 5
```

Ablation — optional attention modules:

```bash
python train_stcmf.py --modalities sat_ms+real_uav --temporal --folds 5
python train_stcmf.py --modalities sat_ms+real_uav --temporal --crossmodal --folds 5
```

Cross-location evaluation (leave-one-location-out):

```bash
python train_stcmf.py --modalities sat_ms+real_uav --cv-mode lolo
```

Smoke run (small batch to check the pipeline):

```bash
python train_stcmf.py --samples 64 --epochs 2 --batch-size 2 --base 16 --folds 2
```

## Key arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--base` | 32 | Backbone base width; lower to 16/24 if overfitting |
| `--uav-size` / `--sat-size` | 448 / 224 | UAV / satellite input size |
| `--temporal` / `--crossmodal` | off | Enable optional attention modules (ablation) |
| `--epochs` | 60 | Training epochs |
| `--batch-size` | 8 | Reduce for limited GPU memory |
| `--cv-mode` | random_kfold | random_kfold / group_kfold / lolo |

## Data

The model consumes the HYBRID HIPS dataset family (co-registered satellite and
UAV imagery of hybrid maize plots). The raw imagery is not redistributed here;
`data/hybrid_plot.py` provides the loader that assembles multi-temporal,
dual-modal plot samples from that dataset.

## License

Released under the [MIT License](LICENSE).

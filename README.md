# LightGCN Recommendation Pipeline

A reproducible PyTorch engineering project for graph-based collaborative filtering with **LightGCN**, developed from the original SIGIR 2020 PyTorch implementation by `gusye1234/LightGCN-PyTorch`.

The LightGCN architecture itself is not presented as a new model here. This project focuses on training/evaluation correctness, reproducibility, profiling, device compatibility, robust dataset handling, and reproducible Top-K inference.

## Highlights

- **Leakage-safe model selection:** deterministic per-user validation holdout; validation Recall@20 selects the checkpoint instead of the benchmark test set.
- **Full-data final training:** after selecting an epoch budget on validation, the original training interactions can be restored and trained for a fixed number of epochs before one final test evaluation.
- **Known-positive masking:** validation evaluation masks training positives; the final test of a validation-selected run masks both training and held-out validation positives.
- **CPU / CUDA / Apple MPS support:** CUDA sparse propagation stays on CUDA; MPS uses a CPU sparse-matrix fallback for compatibility.
- **Profiling:** optional per-epoch timing separates sampling, transfer, forward/graph propagation, backward/optimizer, and residual overhead.
- **Robust loaders:** whitespace-tolerant Gowalla/Yelp2018/Amazon-book parsing plus a validation-safe LastFM loader.
- **Exact Top-K inference:** `predict.py` filters known positives and retrieves recommendations with exact inner-product scoring plus `torch.topk`.
- **Run artifacts:** training writes `summary.json`, optional `history.csv`, and a selected checkpoint under `results/<run_name>/`.

## Final benchmark results

All values below are **Top-20**, seed `2020`, LightGCN with 3 propagation layers and 64-dimensional embeddings. Hyperparameter/epoch selection was performed on validation; the reported final rows were then retrained with the complete original training interactions using the locked epoch budget and evaluated once on the benchmark test split.

| Dataset | Batch | Fixed epochs | Precision@20 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|
| Gowalla | 8192 | 1240 | 0.055831 | 0.182207 | 0.155238 |
| Yelp2018 | 8192 | 660 | 0.027713 | 0.061661 | 0.050565 |
| Amazon-book | 32768 | 1960 | 0.017061 | 0.041401 | 0.031836 |
| LastFM | 2048 | 430 | 0.075484 | 0.267808 | 0.209243 |

The original repository README reports 3-layer Gowalla results of Precision@20 `0.05589`, Recall@20 `0.1824`, NDCG@20 `0.1547`, and Yelp2018 results of Precision@20 `0.0285`, Recall@20 `0.06347`, NDCG@20 `0.05238`. The original README does not provide a directly comparable Amazon-book or LastFM result table, so this project does not invent reference numbers for those datasets.

## Validation selection

| Dataset | Validation setting | Selected epoch | Selection note |
|---|---|---:|---|
| Gowalla | 10% per-user holdout | 1240 | early-stopping selection |
| Yelp2018 | 10% per-user holdout | 660 | early-stopping selection |
| Amazon-book | 10% per-user holdout | 1960 | best Recall@20 within a fixed 2000-epoch budget; max epoch reached |
| LastFM | 10% per-user holdout | 430 | early-stopping selection |

For Amazon-book, epoch 1960 must **not** be described as an early-stopping result: the 2000-epoch budget ended with patience `2/5`.

## Performance audit

Profiling showed that Python negative sampling was not the dominant bottleneck on the tested Apple Silicon machine. Repeated full-graph propagation plus backward/optimizer work dominated training time.

Selected measurements:

| Dataset / setting | Batches/epoch | Training time/epoch | Interpretation |
|---|---:|---:|---|
| Gowalla, batch 2048, CPU | 396 | ~136 s | profiling baseline |
| Gowalla, batch 8192, CPU | 99 | ~31.3 s | ~4.35x epoch-level speedup |
| Amazon-book, batch 8192, CPU | 291 | ~340 s | profiling baseline |
| Amazon-book, batch 32768, CPU | 73 | ~108 s | ~3.14x epoch-level speedup |

These are **epoch-level** speedups, not claims of equivalent convergence speed. Larger batches reduce optimizer updates per epoch and can change the training trajectory. On the Gowalla validation-safe runs, the measured end-to-end time changed from about 31.9 h (batch 2048) to 22.4 h (batch 8192), about a 30% reduction rather than 4.35x.

## Dataset layout

The project uses processed benchmark data from the original LightGCN implementation.

For Gowalla, Yelp2018, and Amazon-book, each line of `train.txt` / `test.txt` is:

```text
user_id item_id_1 item_id_2 item_id_3 ...
```

LastFM uses the original repository's `data1.txt`, `test1.txt`, and `trustnetwork.txt` files.

Place data under:

```text
data/
├── gowalla/train.txt
├── gowalla/test.txt
├── yelp2018/train.txt
├── yelp2018/test.txt
├── amazon-book/train.txt
├── amazon-book/test.txt
└── lastfm/
    ├── data1.txt
    ├── test1.txt
    └── trustnetwork.txt
```

`data/`, checkpoints, and generated run directories are excluded from version control.

## Installation

```bash
pip install -r requirements.txt
```

## Tests

Run the lightweight audit suite from the repository root:

```bash
python tests/test_lightgcn_math.py
python tests/test_evaluation_masking.py
```

Expected output:

```text
LightGCN math audit: PASS
Evaluation masking audit: PASS
```

## Training protocols

### 1. Validation-safe development

Example:

```bash
cd code
python main.py \
  --dataset gowalla \
  --model lgn \
  --recdim 64 \
  --layer 3 \
  --lr 0.001 \
  --decay 0.0001 \
  --bpr_batch 8192 \
  --epochs 2000 \
  --topks "[20]" \
  --seed 2020 \
  --val_ratio 0.1 \
  --eval_every 20 \
  --patience 5 \
  --device cpu \
  --tensorboard 0 \
  --grad_clip 0 \
  --run_name gowalla_validation
```

The validation split is deterministic per user. Validation interactions are removed from the training graph. Intermediate checkpoint selection uses validation Recall@20, not test Recall@20.

### 2. Full-data fixed-epoch final training

After locking an epoch budget using validation, restore the full original training interactions with `--val_ratio 0` and train for that fixed budget. Example for the locked Gowalla run:

```bash
cd code
python main.py \
  --dataset gowalla \
  --model lgn \
  --recdim 64 \
  --layer 3 \
  --lr 0.001 \
  --decay 0.0001 \
  --bpr_batch 8192 \
  --epochs 1240 \
  --topks "[20]" \
  --seed 2020 \
  --val_ratio 0 \
  --device cpu \
  --tensorboard 0 \
  --grad_clip 0 \
  --run_name gowalla_fulltrain_batch8192_epoch1240
```

With `--val_ratio 0`, there is no intermediate validation/test model selection; the test split is evaluated after the requested fixed epoch count.

## Profiling

Add:

```bash
--profile 1
```

to a short run. Each epoch prints `sample`, `h2d`, `forward+graph`, `backward+step`, and total time.

Gradient clipping is optional. `--grad_clip 0` disables it and is the reference-aligned default.


## Inference

`code/predict.py` performs exact full-item scoring:

1. compute LightGCN user/item embeddings;
2. score items by user-item inner product;
3. mask items already interacted with by the user;
4. return `torch.topk` items.

It does **not** claim to implement FAISS or approximate nearest-neighbor retrieval.

## Project structure

```text
.
├── code/                  # data loading, LightGCN, BPR training, evaluation, inference
├── tests/                 # math and evaluation-masking audits
├── results/               # curated CSV summaries
├── requirements.txt
├── .gitignore
└── README.md
```

## Acknowledgements

This project is a secondary development based on the original PyTorch implementation of LightGCN:

- Original implementation: https://github.com/gusye1234/LightGCN-PyTorch
- Paper: *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR 2020.

The LightGCN architecture and substantial parts of the original code framework come from the upstream implementation. This repository focuses on validation-safe model selection, reproducible full-data retraining, device compatibility, profiling, robust data loading, experiment organization, and exact Top-K inference.

The upstream GitHub repository does not currently expose a LICENSE file in its root. For that reason, this derivative repository does not add a new open-source license of its own. The upstream authors and repository are explicitly credited here.

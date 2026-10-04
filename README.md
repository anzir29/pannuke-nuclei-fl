# Nuclei Segmentation and Classification from Histopathology Images using Federated Learning for End-Edge Platform

This repository is a re-creation of the code for the article:

> Chowdhury AA, Hasan Mahmud SM, Palash Uddin Md, Kadry S, Kim J-Y, Nam Y (2025).
> **Nuclei segmentation and classification from histopathology images using federated learning for end-edge platform.**
> *PLoS One* 20(7): e0322749.
> [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322749](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322749)

## Why this is a re-creation

The experiments in the article were carried out in 2021–2022. The original source code, trained models and
raw outputs were stored on a machine that was lost in 2022 due to an unavoidable hardware failure, and they
could not be recovered. This repository re-creates the full pipeline from the methods described in the article,
so that the results can be reproduced from the public PanNuke dataset.

## Pipeline

1. **Segmentation** of nuclei (nucleus vs. background): FCN, U-Net, SegNet, ResUNet.
2. **Classification** of each segmented nucleus into Neoplastic, Inflammatory, Connective, Dead or Epithelial:
   VGG16, VGG19, ResNet50, DenseNet121, InceptionV2.
3. **Hyperband** hyperparameter optimization (Keras Tuner) of SegNet and DenseNet121.
4. **Post-training quantization** with the TensorFlow Lite converter (float32, FP16, dynamic range, full INT8).
5. **Federated learning** (FedAvg, FedProx, FedBN) with INT8 export of the federated models.

## Reproduced results

Results on the 20% test split of PanNuke (single run, seed 0), compared with the values reported in the article.

**Segmentation: SegNet (Tables 2–3 of the article)**

| Metric | Article | Re-created | Re-created / Article |
|---|---|---|---|
| MPA (%) | 91.4 | 93.0 | 101.8% |
| MIoU (%) | 63.5 | 82.4 | 129.8% |
| FWIoU (%) | 90.8 | 89.4 | 98.5% |
| CPA (%) | 44.7 | 76.9 | 172.0% |
| IoU (%) | 35.5 | 71.9 | 202.5% |

**Classification: DenseNet121 (Table 6 of the article)**

| Metric | Article | Re-created | Re-created / Article |
|---|---|---|---|
| Specificity | 0.84 | 0.953 | 113.5% |
| Sensitivity | 0.86 | 0.795 | 92.4% |
| F1-score | 0.83 | 0.781 | 94.1% |
| MCC | 0.67 | 0.764 | 114.0% |
| Accuracy | 0.83 | 0.833 | 100.4% |

**Summary.** 7 of the 10 metrics match or exceed the values reported in the article. The remaining three
reach 92–99% of the reported values (FWIoU −1.4 points, sensitivity −0.065, F1 −0.049).

Exact agreement is not expected from a re-created implementation. The differences can be attributed to:

- **Hyperparameter search:** Hyperband samples configurations at random, so the search explores a different
  set of configurations in every run.
- **Random initialization:** the original random seeds are not known, and weight initialization, data shuffling
  and the train/test split all depend on them.
- **Implementation choices:** where the article does not fix a detail, standard choices were used (see `configs/default.yaml`).
- **Software and hardware:** the original experiments used TensorFlow 2.2 with CUDA 10.1; since 2022 TensorFlow,
  Keras, CUDA and cuDNN have changed substantially, and the experiments here were run on different GPUs.

Per-class results, Hyperband trials, quantization and federated-learning results are in `results/tables/`.

## Repository structure

```
pannuke-nuclei-fl/
├── configs/
│   ├── default.yaml          all settings: data split, training, Hyperband, quantization, federated learning
│   └── minimal.yaml          main results only (SegNet + DenseNet121), builds on default.yaml
├── pannuke_fl/               the Python package
│   ├── run.py                command-line entry point; runs each stage (see below)
│   ├── data.py               PanNuke loading, train/val/test split, nucleus crops, client partitions
│   ├── seg_models.py         FCN, U-Net, SegNet (with pooling-index unpooling), ResUNet
│   ├── cls_models.py         VGG16, VGG19, ResNet50, DenseNet121, InceptionV2 classifiers
│   ├── recipe.py             losses, optimizer and learning-rate schedule, data augmentation, test-time augmentation
│   ├── train.py              training, evaluation, Hyperband search, TFLite quantization
│   ├── tune_worker.py        runs a single Hyperband trial in its own process
│   ├── federated.py          FedAvg, FedProx and FedBN simulation
│   ├── metrics.py            segmentation (PA, MPA, MIoU, FWIoU, CPA, IoU) and classification (SN, SP, F1, MCC, ACC) metrics
│   └── utils.py              config loading, seeding, file helpers
├── scripts/
│   ├── run_minimal.sh        main results on one seed, two GPUs
│   ├── run_all.sh            all experiments, several seeds, two GPUs
│   ├── make_tables.py        collects results into CSV tables (Tables 2, 3, 6, 7; Figs 4–7)
│   └── eta.py                estimates the remaining run time from the logs
├── tests/
│   ├── smoke_test.sh         end-to-end check on a small synthetic dataset
│   ├── make_synthetic.py     writes the synthetic PanNuke-shaped data
│   └── smoke*.yaml           tiny settings for the smoke test
├── results/tables/           result tables (CSV) and summary.md
└── requirements.txt
```

## Installation

Tested on Linux with Python 3.11, TensorFlow 2.16–2.17 and 2× NVIDIA RTX 4090 (24 GB).

```bash
conda create -n pannuke-fl python=3.11 -y
conda activate pannuke-fl
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Data

The experiments use the **PanNuke** dataset (Gamper et al., 2019, 2020): 7,901 H&E image patches (256×256) from
19 tissue types with 189,744 annotated nuclei in five classes. Download the three folds from the
[PanNuke page of the University of Warwick TIA Centre](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
(the Kaggle link in the article is a mirror of the same dataset) and unzip them into `data/pannuke`:

```
data/pannuke/Fold 1/images/fold1/images.npy, types.npy
data/pannuke/Fold 1/masks/fold1/masks.npy
... (Fold 2, Fold 3)
```

## Running

Check the installation on synthetic data first (ends with `SMOKE TEST PASSED`):

```bash
bash tests/smoke_test.sh
```

**Main results** (SegNet and DenseNet121: training, Hyperband, quantization, FedAvg), one seed:

```bash
nohup bash scripts/run_minimal.sh 0 > logs/run_minimal.log 2>&1 &
```

GPU 0 runs the SegNet chain and GPU 1 the DenseNet121 chain; the tables are written to `results/tables/`
at the end. **All experiments** (all models, FedProx/FedBN, client sweep, three seeds):

```bash
nohup bash scripts/run_all.sh configs/default.yaml 0 1 2 > logs/run_all.log 2>&1 &
```

Every stage saves its progress; running the same command again continues where it stopped.
Individual stages can also be run on their own:

| Command | Produces |
|---|---|
| `python -m pannuke_fl.run prep` | preprocessed data and nucleus crops |
| `python -m pannuke_fl.run seg --seed 0` | segmentation results (Tables 2–3) |
| `python -m pannuke_fl.run cls --seed 0` | classification results (Table 6, Fig 4) |
| `python -m pannuke_fl.run tune` | Hyperband search (Fig 5) |
| `python -m pannuke_fl.run opt --seed 0` | optimized and quantized models (Table 7, Fig 6) |
| `python -m pannuke_fl.run fl --seed 0` | federated learning (Table 7) |
| `python -m pannuke_fl.run flsweep --seed 0` | number of clients (Fig 7) |
| `python -m pannuke_fl.run requant --seed 0` | re-runs quantization of trained models |
| `python scripts/make_tables.py` | result tables |

Options: `--config` selects the configuration, `--models segnet densenet121` restricts the `seg` / `cls`
stages, and `--tasks seg` or `--tasks cls` restricts `tune` / `opt` / `fl` / `requant` to one model.

## Outputs

```
results/
├── tables/                   CSV tables and summary.md (the data underlying the reported results)
├── tune/                     Hyperband trials and best configurations
└── seed0/
    ├── seg/<model>/          metrics.json, model.keras, training history
    ├── cls/<model>/          metrics.json (incl. per class), model.keras, training history
    ├── opt/<model>/          optimized model, metrics.json, tflite/ (quantized models and results)
    └── fl/<method>_<task>_c5/   federated-learning results (final.json) and global model
```

## Citation

```bibtex
@article{chowdhury2025nuclei,
  title   = {Nuclei segmentation and classification from histopathology images using federated learning for end-edge platform},
  author  = {Chowdhury, Anjir Ahmed and Hasan Mahmud, S M and Palash Uddin, Md and Kadry, Seifedine and Kim, Jung-Yeon and Nam, Yunyoung},
  journal = {PLoS One},
  volume  = {20},
  number  = {7},
  pages   = {e0322749},
  year    = {2025},
  doi     = {10.1371/journal.pone.0322749}
}
```

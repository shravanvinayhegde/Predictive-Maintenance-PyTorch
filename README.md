# Predictive-Maintenance-PyTorch

**Remaining Useful Life (RUL) prediction for aircraft turbofan engines, built entirely on PyTorch, benchmarked on NASA's C-MAPSS engine degradation dataset.**

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/dataset-NASA%20C--MAPSS-lightgrey)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shravanvinayhegde/Predictive-Maintenance-PyTorch/blob/main/NASA_pytorch.ipynb)

---

## Overview

This repository implements a full predictive maintenance pipeline for jet engines: given a rolling window of multivariate sensor readings, predict how many operating cycles remain before the engine needs maintenance (its **Remaining Useful Life**, or RUL).

The core deliverable, [`NASA_pytorch.ipynb`](./NASA_pytorch.ipynb), takes the problem from raw NASA sensor logs to a trained, evaluated, and compared set of models:

- **Data engineering**: automated download/extraction, missing-value and constant-sensor checks, RUL label construction with the standard piecewise-linear cap, engine-grouped train/validation splitting (no data leakage), and train-set-only feature standardization.
- **Sequence modeling**: sliding-window construction so every model sees a short temporal history of sensor readings rather than a single isolated reading.
- **Four PyTorch architectures** (MLP, 1D-CNN, LSTM, GRU) trained with one shared, reusable training loop, plus a **scikit-learn Random Forest** used strictly as a non-deep-learning benchmark — not as part of the modeling approach.
- **Consistent evaluation** of all five models (MAE, RMSE, R²) on NASA's official, held-out test engines.

**Headline result (FD001, 10 epochs, single run):** the simple **MLP** achieved the best test MAE of **10.33 cycles** (R² = 0.891), ahead of the Random Forest benchmark (11.90) and the LSTM (12.38). See [Results](#results) for the full table and caveats.

Everything downstream of data loading — scaling, windowing, model definitions, training, evaluation — is implemented in PyTorch and plain Python (NumPy/pandas for data handling). scikit-learn is used only for `GroupShuffleSplit`, the regression scoring metrics, and the Random Forest benchmark.

---

## Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation

This project is built around, and organized by, the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset — one of the standard public benchmarks for RUL estimation and prognostics & health management (PHM) research.

- **Publisher**: NASA Ames Research Center, Prognostics Center of Excellence (PCoE)
- **Official repository page**: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **Nature of the data**: run-to-failure, multivariate time series generated from a physics-based turbofan engine simulation. Each engine in the dataset starts with some initial, unknown (but normal) manufacturing variation and wear, operates normally for a while, then develops a fault that grows until failure.

### The four C-MAPSS sub-datasets

The full C-MAPSS release is split into four sub-datasets of increasing complexity, varying by how many operating conditions and fault modes are simulated:

| Sub-dataset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| **FD001** | 100 | 100 | 1 | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC degradation) |
| FD003 | 100 | 100 | 1 | 2 (HPC + Fan degradation) |
| FD004 | 248 | 249 | 6 | 2 (HPC + Fan degradation) |

**This project uses FD001** — the simplest sub-dataset (single operating condition, single fault mode), which keeps the modeling problem focused on sequence learning rather than also having to disentangle multiple operating regimes. The same pipeline generalizes to FD002–FD004 with minor changes (see [Limitations & Possible Extensions](#limitations--possible-extensions)).

### Files and schema

For each sub-dataset, NASA provides three whitespace-separated text files with no header row:

| File | Contents |
|---|---|
| `train_FD001.txt` | Full run-to-failure sensor histories for every training engine |
| `test_FD001.txt` | Sensor histories for test engines, **truncated before failure** |
| `RUL_FD001.txt` | The true remaining cycles for each test engine at the point its series was truncated (ground truth for evaluation) |

Each row of the train/test files represents one engine at one operating cycle, with **26 columns**:

| Column(s) | Meaning |
|---|---|
| `unit_number` | Engine ID (1–100 for FD001) |
| `time_in_cycles` | Operating cycle number for that engine |
| `operational_setting_1..3` | Flight condition settings (altitude, Mach number, throttle resolver angle) |
| `sensor_measurement_1..21` | Raw readings from 21 onboard sensors |

The dataset is downloaded directly inside the notebook from NASA's PHM data mirror:

```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

The archive is a **zip nested inside a zip** (the outer download wraps a folder that itself contains `CMAPSSData.zip`), so extraction is done in two passes before the raw `.txt` files are reachable.

### Sensor reference (FD001, 21 channels)

| Sensor | Symbol | Description | Unit |
|---|---|---|---|
| sensor_measurement_1 | T2 | Total temperature at fan inlet | °R |
| sensor_measurement_2 | T24 | Total temperature at LPC outlet | °R |
| sensor_measurement_3 | T30 | Total temperature at HPC outlet | °R |
| sensor_measurement_4 | T50 | Total temperature at LPT outlet | °R |
| sensor_measurement_5 | P2 | Pressure at fan inlet | psia |
| sensor_measurement_6 | P15 | Total pressure in bypass-duct | psia |
| sensor_measurement_7 | P30 | Total pressure at HPC outlet | psia |
| sensor_measurement_8 | Nf | Physical fan speed | rpm |
| sensor_measurement_9 | Nc | Physical core speed | rpm |
| sensor_measurement_10 | epr | Engine pressure ratio (P50/P2) | — |
| sensor_measurement_11 | Ps30 | Static pressure at HPC outlet | psia |
| sensor_measurement_12 | phi | Ratio of fuel flow to Ps30 | pps/psi |
| sensor_measurement_13 | NRf | Corrected fan speed | rpm |
| sensor_measurement_14 | NRc | Corrected core speed | rpm |
| sensor_measurement_15 | BPR | Bypass ratio | — |
| sensor_measurement_16 | farB | Burner fuel-air ratio | — |
| sensor_measurement_17 | htBleed | Bleed enthalpy | — |
| sensor_measurement_18 | Nf_dmd | Demanded fan speed | rpm |
| sensor_measurement_19 | PCNfR_dmd | Demanded corrected fan speed | rpm |
| sensor_measurement_20 | W31 | HPT coolant bleed | lbm/s |
| sensor_measurement_21 | W32 | LPT coolant bleed | lbm/s |

In FD001, several of these channels never change value across the whole dataset — the notebook detects and drops these **constant sensors** programmatically with `nunique() <= 1` rather than hard-coding which ones they are. On the executed run it dropped 6 sensors (`sensor_measurement_1, 5, 10, 16, 18, 19`), leaving **18 of 24** candidate features (3 operational settings + 15 sensors).

> **Note:** the constant-column check is applied to the 21 sensor columns only. `operational_setting_3` is also constant in FD001 (always `100.0`), so it is kept, produces `NaN` when standardized (std = 0), and is zero-filled with `np.nan_to_num` before training. In practice the models therefore see 17 informative features plus one all-zero column.

### Building the RUL target

Training engines run all the way to failure, so the label is derived directly from each engine's own history:

```
RUL(cycle t) = (final cycle the engine reached) - t
```

Raw RUL would grow unboundedly for engines that lived a long time, and early-life degradation is effectively negligible — an engine at cycle 5 of a 300-cycle life isn't meaningfully "healthier" than one at cycle 5 of a 200-cycle life. To reflect that, the target is **capped with a piecewise-linear function** (`RUL_CAP = 125`), a standard construction in the RUL-prediction literature: RUL decreases linearly once true degradation sets in, but is flat at the cap before that.

For the test set, `RUL_FD001.txt` supplies the ground-truth RUL at the last recorded cycle of each truncated engine, and the same cap is applied for a fair, apples-to-apples comparison against the training target.

### Engine-grouped splitting (no leakage)

Rows are **not** split randomly. Consecutive cycles from the same engine are highly correlated, so a per-row random split would let the model train on cycles that sit right next to ones it's tested on — an easy way to get an optimistic score that doesn't reflect real generalization. This project uses scikit-learn's `GroupShuffleSplit`, grouping by `unit_number`, so **every cycle from a given engine ends up entirely on one side** of the train/validation split. Feature standardization (mean/std) is likewise computed **only from the training split** and then applied to validation and test, to avoid leaking test-set statistics into training.

---

## Repository structure

```
Predictive-Maintenance-PyTorch/
├── NASA_pytorch.ipynb          # Core project: NASA C-MAPSS RUL prediction pipeline
├── Experiments/                # Standalone PyTorch notebooks on other domains/datasets
│   ├── Animal_face_classification (1).ipynb   # Image classification, 5-model comparison
│   ├── Audio_Classification.ipynb             # Audio classification (Quran reciters dataset)
│   ├── GTSRB.ipynb                            # Traffic sign recognition, 8-model comparison
│   ├── Rice_Classification.ipynb              # Tabular classification (rice grain type)
│   └── Text_Classification_Transformers.ipynb # Text classification with Transformers
├── assets/                     # Plots exported from the executed notebook (used in this README)
├── docs-site/                  # Astro documentation site (deployed to GitHub Pages)
├── Quick-Reaps/                 # PyTorch fundamentals reference notebooks
│   ├── pytorch_01.ipynb        # Tensors and core PyTorch operations
│   ├── pytorch_02.ipynb        # Training workflow via linear regression
│   ├── pytorch_03.ipynb        # Neural network classification (binary & multi-class)
│   ├── pytorch_04.ipynb        # Computer vision on FashionMNIST
│   └── pytorch_05.ipynb        # Custom image datasets (Pizza/Steak/Sushi)
└── README.md
```

`NASA_pytorch.ipynb` is the project this README documents. `Experiments/` and `Quick-Reaps/` are separate, self-contained notebooks kept in the same repository for reference — they don't touch the C-MAPSS dataset or the RUL pipeline and can be ignored if you're only here for the predictive maintenance work.

---

## Pipeline

The notebook runs as a single linear pipeline, in this order:

1. **Download & extract** the C-MAPSS archive (two-pass unzip).
2. **Load** `train_FD001.txt` / `test_FD001.txt` / `RUL_FD001.txt` into pandas DataFrames with explicit column names (the source files have no header).
3. **Quick EDA** — check for missing values, count engines, identify and drop constant sensor columns.
4. **Engineer the RUL label** from each training engine's own max cycle, with the piecewise-linear cap (`RUL_CAP = 125`).
5. **Split by engine** into train/validation using `GroupShuffleSplit` (80/20, grouped by `unit_number`).
6. **Standardize features** using only the training split's mean/std.
7. **Slide a fixed-size window** (`WINDOW_SIZE = 30` cycles) over each engine's chronologically sorted history to build `(X, y)` sequences; the label for each window is the RUL at its final cycle. The official test set uses only each engine's most recent window, left-padded if an engine has fewer than 30 recorded cycles.
8. **Wrap sequences** in a custom `torch.utils.data.Dataset` / `DataLoader` (`BATCH_SIZE = 64`). NaNs from the zero-variance `operational_setting_3` column are replaced with `0.0` first.
9. **Train and evaluate** four PyTorch models plus the Random Forest benchmark with a single shared training/evaluation routine, then compare all five on MAE, RMSE, and R².

---

## Models

Every PyTorch model takes a `(batch, WINDOW_SIZE, n_features)` sequence and outputs a single scalar (predicted RUL) — same input/output contract, four different ways of processing what happens in between.

| Model | Approach | Key layers |
|---|---|---|
| **MLP** | Flattens the whole window into one vector, ignores time order entirely — the simplest baseline | `Flatten → Linear → ReLU → Linear → ReLU → Linear` |
| **1D-CNN** | Slides learnable filters along the time axis to pick up short local degradation trends | `Conv1d → ReLU → Conv1d → ReLU → AdaptiveAvgPool1d → Linear` |
| **LSTM** | Reads the window one cycle at a time, keeping a memory cell that can retain long-range information | `LSTM(batch_first=True) → Linear` on the final hidden state |
| **GRU** | Same recurrent idea as LSTM with simpler gating — fewer parameters, typically faster to train | `GRU(batch_first=True) → Linear` on the final hidden state |
| **Random Forest** *(benchmark, not PyTorch)* | Ensemble of decision trees on the flattened window — a sanity check for whether the extra complexity of a neural net is actually earning its keep | `RandomForestRegressor(n_estimators=100, max_depth=10)` |

All four PyTorch models share one training function: Adam optimizer, MSE loss, and per-epoch validation MAE tracking.

### Key hyperparameters

| Parameter | Value | Where it's used |
|---|---|---|
| `WINDOW_SIZE` | 30 cycles | Sequence length fed to every model |
| `RUL_CAP` | 125 cycles | Piecewise-linear RUL target cap |
| `BATCH_SIZE` | 64 | `DataLoader` batching |
| `EPOCHS` | 10 | Training loop |
| `LR` | 1e-3 | Adam optimizer learning rate |
| Train/val split | 80% / 20%, grouped by engine | `GroupShuffleSplit(test_size=0.2)` |
| Random seed | 42 | `torch.manual_seed`, `np.random.seed`, `random_state` throughout |

---

## Evaluation metrics

All five models — the four PyTorch networks and the Random Forest benchmark — are scored identically on the official NASA test engines using:

- **MAE** (Mean Absolute Error) — average absolute difference between predicted and true RUL, in cycles. The most directly interpretable metric here.
- **RMSE** (Root Mean Squared Error) — like MAE but penalizes large errors more heavily.
- **R²** (coefficient of determination) — how much of the variance in true RUL is explained by the model's predictions.

## Results

All numbers below come from the executed run saved in [`NASA_pytorch.ipynb`](./NASA_pytorch.ipynb) (PyTorch 2.11.0, CUDA GPU, seed 42). Models are scored on NASA's official FD001 test set: **100 engines**, one prediction per engine at its truncation point, with the true RUL capped at 125.

| Rank | Model | Test MAE (cycles) | Test RMSE | Test R² |
|---|---|---|---|---|
| 1 | **MLP** | **10.33** | **13.26** | **0.891** |
| 2 | Random Forest (benchmark) | 11.90 | 16.09 | 0.839 |
| 3 | LSTM | 12.38 | 15.80 | 0.845 |
| 4 | GRU | 13.12 | 16.75 | 0.825 |
| 5 | 1D-CNN | 15.32 | 21.44 | 0.714 |

![Test MAE by model and validation MAE over epochs](https://github.com/shravanvinayhegde/Predictive-Maintenance-PyTorch/blob/main/docs-site/dist/assets/results_comparison.png)

*Left: test MAE per model (lower is better). Right: validation MAE per epoch for the four PyTorch models.*

### Validation vs. test

| Model | Final-epoch validation MAE (cycles) | Test MAE (cycles) |
|---|---|---|
| MLP | 11.67 | 10.33 |
| 1D-CNN | 14.44 | 15.32 |
| LSTM | 15.07 | 12.38 |
| GRU | 15.78 | 13.12 |
| Random Forest | 12.28 | 11.90 |

Validation MAE is measured on all sliding windows of the 20 held-out engines, whereas test MAE is measured on one final window per test engine, so the two columns are not directly comparable. The ordering is broadly consistent between them.

### What the results show

- **The MLP won, even though it ignores time order.** Within a 30-cycle window of FD001 (one operating condition, one fault mode), the degradation signal appears to be captured well enough by the flattened window that explicit sequence modelling did not help in this setup.
- **The Random Forest benchmark beat three of the four PyTorch models on MAE** (LSTM, GRU, 1D-CNN). Only the MLP beat it on all three metrics. The LSTM edges the Random Forest slightly on RMSE (15.80 vs 16.09) and R² (0.845 vs 0.839), but not on MAE.
- **LSTM and GRU were still improving when training stopped.** Their validation MAE fell from about 62 to about 15–16 cycles over 10 epochs and was still dropping steeply, while the MLP and 1D-CNN had largely plateaued. The 10-epoch budget probably under-represents the recurrent models, so their ranking here should not be read as a verdict on recurrent architectures.
- **The 1D-CNN was the weakest model** on every metric, with a notably higher RMSE (21.44), which points to a few large errors on individual engines.
- **LSTM ≈ GRU**, as expected: the GRU is slightly worse on this run, with near-identical training curves.

### Caveats

- This is a **single run with one seed and one train/validation split**, evaluated on only 100 test engines. Differences of 1–2 cycles of MAE (for example MLP vs Random Forest, or LSTM vs GRU) are within what you would expect from seed-to-seed variance and should not be treated as conclusive.
- All models use the same fixed budget (10 epochs, Adam, lr = 1e-3, no scheduling, no early stopping, no hyperparameter search). This makes the comparison like-for-like, not tuned.
- The RUL target is capped at 125, and the test ground truth is capped identically, so these MAE values are on the capped scale and are not directly comparable to papers that report uncapped RUL or the asymmetric PHM score.

### Other notebook outputs

| Output | Value |
|---|---|
| Raw shapes | `train_df` (20,631 × 26), `test_df` (13,096 × 26), `rul_df` (100 × 1) |
| Missing values | 0 |
| Train / validation split | 80 engines (16,561 rows) / 20 engines (4,070 rows) |
| Sliding-window tensors | `X_train` (14,241, 30, 18), `X_val` (3,490, 30, 18), `X_test` (100, 30, 18) |
| Batches per epoch | 223 train, 55 validation |
| Random Forest validation MAE | 12.28 cycles |

Engine 1's raw sensor trends and its capped RUL target, as plotted in the notebook:

![Engine 1 sensor trends](https://github.com/shravanvinayhegde/Predictive-Maintenance-PyTorch/blob/main/docs-site/dist/assets/engine1_sensors.png)

![Engine 1 piecewise-linear RUL target](https://github.com/shravanvinayhegde/Predictive-Maintenance-PyTorch/blob/main/docs-site/dist/assets/rul_target_engine1.png)

---

## Installation

Requires Python 3.9+ and the following packages:

```bash
pip install torch pandas numpy matplotlib scikit-learn requests
```

A CUDA-capable GPU is optional — device selection (`cuda` if available, else `cpu`) is automatic.

## Usage

```bash
git clone https://github.com/shravanvinayhegde/Predictive-Maintenance-PyTorch.git
cd Predictive-Maintenance-PyTorch
jupyter notebook NASA_pytorch.ipynb
```

Or open it directly in Colab using the badge at the top of this README (recommended — the executed results above were produced on a Colab GPU). No manual dataset download is required — the first two cells fetch and extract the C-MAPSS FD001 files automatically. Run all cells top to bottom.

> **Running locally:** the download/extract cells and `find_file()` use Colab-style paths under `/content/`. Change `/content/` to a directory that exists on your machine (for example `./data/`) before running outside Colab.

---

## Limitations & possible extensions

- **Single sub-dataset**: only FD001 (1 operating condition, 1 fault mode) is used. FD002–FD004 add multiple operating conditions and a second fault mode, which typically requires per-condition normalization instead of a single global standardization.
- **Fixed training budget**: 10 epochs, no learning-rate scheduling, no early stopping, and no hyperparameter search — the comparison is meant to be a fair like-for-like baseline across architectures, not a tuned leaderboard entry.
- **Single train/validation split and single seed**: one `GroupShuffleSplit` and one training run per model, not k-fold cross-validation or repeated seeds, so reported metrics carry split-to-split and seed-to-seed variance.
- **Recurrent models look under-trained**: LSTM and GRU validation MAE was still falling at epoch 10, so a longer budget or early stopping on a larger epoch cap could change the ranking.
- **Constant operational setting not dropped**: `operational_setting_3` is constant in FD001 but is only zero-filled after standardization rather than removed (see the note under [Files and schema](#files-and-schema)).
- **No attention-based or Transformer sequence model** is included alongside the MLP/CNN/LSTM/GRU comparison.
- **No official PHM scoring function**: the standard C-MAPSS literature also reports an asymmetric "PHM score" that penalizes late predictions more than early ones; this project reports MAE/RMSE/R² only.

Natural next steps: extend the pipeline to FD002–FD004, add the asymmetric PHM scoring metric, train the recurrent models longer with early stopping, repeat runs over several seeds to report mean ± std, drop constant operational settings explicitly, and try a Transformer or attention-augmented recurrent model.

---

## Other notebooks in this repository

`Experiments/` and `Quick-Reaps/` are unrelated to the NASA predictive maintenance pipeline — they're separate PyTorch notebooks (image, audio, text, and tabular classification tasks, plus a set of PyTorch fundamentals references) kept in this repository for convenience. See the [repository structure](#repository-structure) above for what each one covers.

---

## References

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,"* International Conference on Prognostics and Health Management (PHM), 2008 — the original paper introducing the C-MAPSS dataset.
- NASA PCoE Data Set Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## License

No license file is currently included in this repository, which means default copyright applies and the code is not formally licensed for reuse. If you intend for others to use or build on this project, consider adding a license (e.g., MIT or Apache 2.0).

## Author

[@shravanvinayhegde](https://github.com/shravanvinayhegde)

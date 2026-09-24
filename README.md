# 🌧️ Drought Forecast Binary Framework

A framework for forecasting rare drought events using spatiotemporal neural networks (ConvLSTM) with dual temporal attention and transfer learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Related Projects](#related-projects)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow](#workflow)
- [Metrics](#metrics)
- [Results](#results)

---

## 🎯 Overview

This framework predicts extreme drought events (SPI ≤ a configurable severity threshold) 1 to 12 months ahead, using climate reanalysis data for Brazil's five geographic macro-regions.

### Key Features

- **🧠 ConvLSTM with Dual Temporal Attention**: Captures spatiotemporal patterns in climate data.
- **🔄 Transfer Learning**: Autoencoder pretraining for unsupervised representation learning.
- **⚖️ Focal Loss with Dynamic Balancing**: Handles the extreme class imbalance of rare drought events.
- **📊 Probability Calibration**: Platt Scaling or Isotonic Regression for well-calibrated probabilities.
- **🎯 Adaptive Threshold**: Automatic decision-threshold optimization based on MCC (Matthews Correlation Coefficient).
- **🌍 Spatial Post-Processing**: Morphological cleanup to reduce false positives in the predicted masks.
- **🔁 Reproducibility**: Fixed seeds and deterministic operations where supported.

---

## 🔗 Related Projects

**`drought_forecast_regression`** (sibling project, same repo root) is a derived version of this framework for **continuous SPI forecasting** instead of binary drought-event classification. It shares the same two-stage architecture (ConvLSTM Autoencoder pretraining + ConvLSTM Predictor, with a Transfer Learning vs. Scratch comparison and the same p/q grid), but:

- Predicts the continuous SPI value at each pixel (`SPI(x, y, t+q)`) instead of a thresholded drought/no-drought mask.
- Uses a linear regression output head (no sigmoid) and Masked MSE loss instead of Focal Loss/weighted BCE.
- Drops every classification-only component: SPI severity threshold, class-imbalance handling, decision-threshold search, probability calibration, and morphological post-processing of binary masks.
- Evaluates with Willmott's Index of Agreement (WI), RMSE, and MAE instead of CSI/MCC/F1/precision/recall.

See `../drought_forecast_regression/README.md` for its full documentation and migration notes.

---

## 🏗️ Architecture

### Predictor Model

```
Input (B, T, C, H, W)
    ↓
ConvLSTM Encoder (3 layers: 64 → 32 → 16)
    ↓
Dual Temporal Attention
    ├── Local Attention (short-term)
    ├── Global Attention (long-term)
    └── Adaptive Gate (dynamic combination)
    ↓
Prediction Decoder
    ↓
Logits (B, 1, H, W)
    ↓
Sigmoid → Probabilities
```

### Autoencoder (Pretraining)

```
Input Sequence (B, T, C, H, W)  # T = 12 months
    ↓
ConvLSTM Encoder
    ↓
Dual Temporal Attention
    ↓
Temporal Decoder (full-sequence reconstruction)
    ↓
Reconstructed Sequence (B, T, C_out, H, W)
```

---

## 📦 Installation

### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Steps

```bash
# Clone the repository
git clone https://github.com/your-username/drought-forecast-binary.git
cd drought-forecast-binary

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
rasterio>=1.3.0
scikit-image>=0.19.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
openpyxl>=3.0.0
joblib>=1.1.0
```

Note: model training/inference typically runs on an external server, whose installed package versions (e.g. numpy) may differ from this repo's local development machine - see `analyze_spi_distribution_shift.py` for a case where that mismatch (numpy 2.x on the server vs. 1.x locally) required an explicit pickle-compatibility shim. Keep these as minimum floors rather than exact pins unless you have the server's own `pip freeze` to pin against.

### Raw Data

The raw TerraClimate rasters are expected under `DATA_BASE_PATH`, defined in `config/paths.py`, organized as one subfolder per region (e.g. `Sul/`, `Norte/`) containing files named `{region}_{year}_{month}.tif`. Edit `DATA_BASE_PATH` there if your data lives elsewhere.

---

## 📁 Project Structure

```
drought_forecast_binary/
├── config/                              # Centralized configuration
│   ├── __init__.py
│   ├── experiment_config.py             # All experiment settings
│   └── paths.py                         # Path management
│
├── data/                                # Data handling
│   ├── __init__.py
│   ├── loader.py                        # load_region_timeseries
│   ├── preprocessing.py                 # Downsampling, validity mask, normalization
│   ├── dataset.py                       # ClimateDataset
│   ├── augmentation.py                  # ExtremeDroughtAugmenter
│   └── spi.py                           # SPI computation and cache
│
├── models/                              # Neural network models
│   ├── __init__.py
│   ├── convlstm_cell.py                 # ConvLSTMCell
│   ├── encoder.py                       # ConvLSTMEncoder
│   ├── decoder.py                       # ReconstructionDecoder, PredictionDecoder
│   ├── attention.py                     # DualTemporalAttention
│   ├── autoencoder.py                   # ConvLSTMAutoencoder
│   └── predictor.py                     # ConvLSTMPredictor
│
├── training/                            # Training logic
│   ├── __init__.py
│   ├── base.py                          # BaseTrainer
│   ├── autoencoder.py                   # AutoencoderTrainer
│   ├── predictor.py                     # PredictorTrainer
│   └── losses.py                        # FocalLoss, WeightedSmoothL1Loss, WeightedBCEWithLogitsLoss
│
├── evaluation/                          # Evaluation tools
│   ├── __init__.py
│   ├── metrics.py                       # CSI, MCC, F1, etc.
│   ├── threshold.py                     # ThresholdOptimizer
│   └── calibration.py                   # PlattCalibrator, IsotonicCalibrator
│
├── utils/                               # Utilities
│   ├── __init__.py
│   ├── logger.py                        # Colored logger
│   ├── reproducibility.py               # Seed management
│   ├── spatial.py                       # Morphological post-processing
│   └── geotiff.py                       # GeoTIFF export
│
├── experiments/                         # Experiment runners
│   ├── __init__.py
│   ├── precompute_spi.py                # SPI precomputation
│   ├── train_autoencoder.py             # Autoencoder training
│   ├── grid_search.py                   # Hyperparameter grid search
│   └── analyze_variable_importance.py   # Variable-importance analysis
│
├── inference/                           # Inference module
│   ├── __init__.py
│   ├── predictor.py                     # InferencePredictor
│   ├── run.py                           # Inference CLI
│   └── analyze.py                       # Inference-results analysis (metrics, spatial maps, calibration)
│
├── evaluate_autoencoder.py              # Standalone autoencoder evaluation report
│
├── outputs/                             # Outputs (created at runtime)
│   └── {region}/                        # Per region (Sul, Norte, etc.)
│       ├── autoencoder/                 # Trained autoencoder + normalizer
│       ├── spi_cache/                   # Cached SPI series
│       ├── grid_search/                 # Grid search models and results
│       ├── inferences/                  # Inference rasters (pred/truth/prob)
│       └── analysis/                    # Metrics and plots
│
├── requirements.txt                     # Dependencies
├── main.py                              # Main entry point
└── README.md                            # This file
```

---

## ⚙️ Configuration

All settings are centralized in `config/experiment_config.py`:

```python
@dataclass
class ExperimentConfig:
    region: str = "Sul"                                        # Must match a raw-data folder name
    data: DataConfig = field(default_factory=DataConfig)
    spi: SPIConfig = field(default_factory=SPIConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    p_values: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    q_values: List[int] = field(default_factory=lambda: [1, 3, 6, 9, 12])
    random_seed: int = 42
```

> **Note:** `precompute-spi` and `train-ae` (see [Usage](#-usage)) read `region`, `spi.scale`, and `spi.threshold` exclusively from these dataclass defaults — they have no corresponding CLI flags. To run those two steps for a different region or SPI configuration, edit `config/experiment_config.py` first. `grid-search` and `inference` do accept `--region`/`--threshold-spi` overrides.

### Climate Variables

The framework uses 7 TerraClimate bands:

| Band | Description | Unit |
|------|-------------|------|
| `pr` | Precipitation | mm/month |
| `pet` | Potential Evapotranspiration | mm/month |
| `soil` | Soil Moisture | mm |
| `srad` | Downward Shortwave Radiation | W/m² |
| `vap` | Vapor Pressure | kPa |
| `vs` | Wind Speed | m/s |
| `tavg` | Mean Air Temperature | °C |

---

## 🚀 Usage

### Main Command

```bash
python3 main.py <command> [options]
```

### 1. Precompute SPI

Region and SPI scale/threshold come from `config/experiment_config.py` (no CLI flags for this command):

```bash
python3 main.py precompute-spi
```

### 2. Train the Autoencoder

Also driven entirely by `config/experiment_config.py`. The optional `--force` flag is accepted but currently has no effect: by default (`AutoencoderConfig.train = True`) the command always retrains, even if a checkpoint already exists at `outputs/{region}/autoencoder/model.pth`. To make it skip retraining when a checkpoint is already present, set `AutoencoderConfig.train = False` instead.

```bash
python3 main.py train-ae
```

### 3. Grid Search

```bash
# Uses the region/threshold from config/experiment_config.py
python3 main.py grid-search

# Override region, SPI threshold, and selection metric
python3 main.py grid-search --region Sul --threshold-spi -2.0 --optimization-metric mcc

# Force transfer learning from the pretrained autoencoder
python3 main.py grid-search --use-transfer-learning
```

### 4. Inference

By default, inference reuses the decision threshold that was already fixed on the validation set during grid search (stored in the model checkpoint) — it is never re-optimized on the test set:

```bash
# Automatic: uses the best configuration found by the grid search,
# with the threshold fixed during grid-search validation
python3 main.py inference

# Specific model
python3 main.py inference --region Sul --p 12 --q 1 --model-type pretrained

# Explicit fixed decision threshold (overrides the checkpoint's value)
python3 main.py inference --region Sul --p 12 --q 1 --threshold 0.30

# Recompute the threshold from validation data instead of reusing the
# checkpoint's value (still never touches the test set to choose it)
python3 main.py inference --recalibrate

# Without probability calibration
python3 main.py inference --p 9 --q 1 --no-calibration

# List available trained models for the current region/threshold
python3 main.py inference --list-models
```

### 5. Standalone Analysis Scripts

These are not wired into `main.py` and are run directly:

```bash
# Detailed autoencoder reconstruction/latent-space report
python3 evaluate_autoencoder.py

# Inference results report (timeseries metrics, spatial maps, calibration curve)
python3 -m inference.analyze --region Sul --p 12 --q 1 --model-type pretrained
# or, pointing directly at a result directory:
python3 -m inference.analyze --pred-dir outputs/Sul/inferences/threshold_1.5/p12_q1/pretrained/test_original

# Variable-importance analysis for a region
python3 experiments/analyze_variable_importance.py --region Sul --threshold -2.0
```

---

## 🔄 Workflow

### 1. Preprocessing
```
precompute_spi.py
├── Loads the region's rasters
├── Computes SPI with a rolling window (config.spi.scale)
├── Analyzes summary statistics
└── Saves the result to cache
```

### 2. Pretraining (optional)
```
train_autoencoder.py
├── Normalizes data seasonally
├── Trains the autoencoder with full-sequence reconstruction
└── Saves the model and the normalizer
```

### 3. Grid Search
```
grid_search.py
├── For each (p, q) in p_values × q_values:
│   ├── Builds the datasets (with augmentation/transfer learning if enabled)
│   ├── Trains the model with Focal Loss
│   ├── Optimizes the decision threshold (MCC by default)
│   └── Saves the checkpoint
└── Produces a report of the best configurations
```

### 4. Inference
```
inference/run.py
├── Loads the best (or specified) model
├── Loads the calibrator, if available
├── Resolves the decision threshold BEFORE touching test data:
│     explicit --threshold > --recalibrate (validation only) > checkpoint's value
├── Runs predictions over the test period using that fixed threshold
└── Exports GeoTIFF rasters (probability, binary prediction, ground truth)
```

### 5. Analysis
```
inference/analyze.py
├── Computes per-timestep metrics
├── Generates timeseries plots
├── Spatial frequency and bias maps
├── Calibration curves
└── Best/worst prediction figures
```

---

## 📊 Metrics

### Primary Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **MCC** | Matthews Correlation Coefficient | [-1, 1] |
| **CSI** | Critical Success Index | [0, 1] |
| **F1** | F1 Score | [0, 1] |
| **Precision** | Fraction of positive predictions that are correct | [0, 1] |
| **Recall** | Fraction of actual events detected | [0, 1] |

### Supporting Metrics

| Metric | Description |
|--------|-------------|
| **FAR** | False Alarm Ratio |
| **Bias** | Ratio of predicted to observed positives |
| **ECE** | Expected Calibration Error |

---


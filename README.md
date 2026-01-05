# Predicting Mass of Supersymmetric Particles via Regression on Decay Chain

Machine learning models for predicting the mass of intermediate supersymmetric (SUSY) particles using Graph Neural Networks (GNNs) and contrastive learning techniques.


## Codebase Structure

### Directory Organization

```
SUSY-Mass-Regression/
├── Root-level modules/       # Core utilities and shared components
├── gnn_baseline/             # Baseline GNN supervised learning
├── gnn_transformed/          # GNN with data augmentation
├── siamese/                  # Contrastive learning (SimCLR) models
├── lorentz_addition/         # Physics-based baseline approach
├── model_evaluation/         # Evaluation scripts and analysis tools
├── raw_data/                 # Original ROOT files (dataset 1)
├── raw_data_set2/            # Original ROOT files (dataset 2)
├── raw_data_background/      # Background events for robustness testing
├── processed_data/           # Preprocessed NumPy arrays (dataset 1)
├── processed_data_set2/      # Preprocessed NumPy arrays (dataset 2)
└── processed_data_background/ # Preprocessed background data
```

### Core Utility Modules (Root Level)

#### Data Processing & Configuration
- **`config_loader.py`**: Loads YAML configuration files and computes absolute paths for data directories.
- **`ROOT_utils.py`**: Extracts particle features from ROOT physics data files, creates Lorentz four-vectors, and performs momentum calculations.
- **`utils.py`**: Core utilities including coordinate transformations, data normalization/scaling, dataset loading, Lorentz addition calculations, and regression metrics computation.

#### Data Augmentation
- **`transformation.py`**: Provides data augmentation for contrastive learning, including random particle deletion and identity transformations. Creates augmented view pairs for SimCLR training and single augmented views for evaluation.

#### Model Components
- **`graph_embeddings.py`**: Custom TensorFlow/Keras layer implementing a Graph Neural Network with edge-based message passing. Transforms particle graphs to fixed-size embeddings via global pooling.

- **`loss_functions.py`**: Custom loss functions for contrastive learning including SimCLR's NT-Xent loss and VICReg loss (variance-invariance-covariance regularization).

- **`callbacks.py`**: Pre-configured callback sets for different training scenarios: standard callbacks with early stopping, no-stop callbacks for fixed-epoch training, and finetuning callbacks for progressive encoder freezing.

- **`simCLR_model.py`**: SimCLR model implementation with GNN encoder and projection head for self-supervised contrastive learning.

- **`downstream_model.py`**: Models for finetuning pretrained encoders on downstream regression tasks, with support for progressive encoder freezing strategies.

#### Visualization
- **`plotting.py`**: Visualization functions for creating histograms, performance comparison plots, variance analysis, and error bar visualizations across different mass ranges.

### Training Modules

#### `gnn_baseline/`
- **`gnn.py`**: Standard supervised learning GNN
  - Architecture: GraphEmbeddings → Dense layers → Single output
  - Trains on original (non-augmented) data
  - Uses early stopping and learning rate reduction
  - Outputs: `best_model.keras`, `results.json`, `training_logs.csv`

#### `gnn_transformed/`
- **`gnn_transformed.py`**: GNN trained with data augmentation
  - Same architecture as baseline but with random particle deletion
  - No early stopping, uses learning rate scheduling
  - Evaluates robustness to input perturbations

#### `siamese/`
- **`siamese.py`**: SimCLR contrastive learning model
  - Self-supervised pretraining with augmented view pairs
  - Saves encoder separately: `best_model_encoder.keras`
  - Generates embeddings: `train_embeddings.npz`, `val_embeddings.npz`, `test_embeddings.npz`

- **`train_finetune.py`**: Downstream task with progressive finetuning
  - Initial epochs: encoder trainable with low learning rate
  - Later epochs: encoder frozen with higher learning rate
  - Uses `FinetunedNN` model with `FinetuningCallback`

- **`train_no_finetune.py`**: Downstream task with frozen encoder
  - Only trains downstream layers
  - Tests pure representation quality

#### `lorentz_addition/`
- **`lorentz_addition.py`**: Physics-based baseline approach
  - Naively sums 4-momentum vectors of all visible particles
  - Assumes MET pseudorapidity = 0
  - Serves as performance baseline for ML models
  - `lorentz_addition_set2.py`: Same approach for dataset 2

### Model Evaluation

#### `model_evaluation/`
- **`helpers.py`**: Shared utilities for loading models, creating performance tracking structures, and aggregating metrics across all model types.

- **`same_event_type.py`**: Evaluates models file-by-file (by event type), generating dual histograms for selected event types and accuracy plots across all mass points.

- **`full_dataset.py`**: Evaluates on the complete concatenated test dataset with variance analysis using prediction buckets.

- **`transformed_same_event_type.py`** & **`transformed_full_dataset.py`**: Robustness evaluation on augmented/transformed inputs to test model performance under particle deletion.

- **`background_test.py`**: Tests models on background events (non-SUSY Standard Model processes) to evaluate false positive rates and model specificity.

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This codebase also requires ROOT (CERN's data analysis framework) for reading `.root` files. Follow instructions at: https://root.cern/install/#download-a-pre-compiled-binary-distribution


### 2. Download Data

Place your ROOT data files in the appropriate directories:

**Dataset 1 (4 particles, 6 features):**
```
raw_data/
├── train_qX_qWY_qqqlv_X200_Y60.root
├── train_qX_qWY_qqqlv_X250_Y80.root
├── ...
└── test_qX_qWY_qqqlv_X400_Y160.root
```

**Dataset 2 (12 particles, 7 features):**
```
raw_data_set2/
├── train_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A2500_X1000_Y500.root
├── train_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A3300_X1400_Y550.root
├── ...
└── test_bA_bgX_bgqqqqY_bgqqqqqqqqlv_A6500_X3000_Y1000.root
```

**Background data (Standard Model events):**
```
raw_data_background/
└── test_TbqqTblv.root
```

### 3. Preprocess Data

Convert ROOT files to NumPy arrays:

**For Dataset 1:**
```bash
python3 preprocess_data.py
```

**For Dataset 2:**
```bash
python3 preprocess_data.py --dataset set2
```

**For Background data:**
```bash
python3 preprocess_data_background.py
```

This creates the following directory structure:
```
processed_data/
├── train/
│   ├── train_qX_qWY_qqqlv_X200_Y60.npz
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each `.npz` file contains:
- `X`: Input features, shape `(n_events, n_particles, n_features)`
- `y`: Target masses, shape `(n_events,)`
- `y_eta`: MET pseudorapidity, shape `(n_events,)`


## Configuration Files

### `config.yaml` (Dataset 1)

Controls settings for the 4-particle, 6-feature dataset:

**Directory paths:**
```yaml
RAW_DATA_DIRECTORY: raw_data
PROCESSED_DATA_DIRECTORY: processed_data
```

**Data structure parameters:**
```yaml
DECAY_CHAIN: ["P1"]             # Single decay chain
N_PARTICLES: 4                  # Number of particles per event
N_FEATURES: 6                   # Features: [pt, eta, phi, one-hot_MET, one-hot_Lepton, one-hot_Other]
MET_IDS: [12]                   # Particle ID for missing transverse energy
LEPTON_IDS: [11]                # Particle ID for leptons
TRAIN_TEST_SPLIT: 0.2           # 20% for validation+test
SCALABLE_FEATURES: [0, 1, 2]    # Scale pt, eta, phi (not one-hot encoded features)
```

**Evaluation files:**
```yaml
eval_data_files:                # Test files for per-event-type evaluation
  - test_qX_qWY_qqqlv_X200_Y60.npz
  - test_qX_qWY_qqqlv_X250_Y80.npz
  - test_qX_qWY_qqqlv_X300_Y100.npz
  - test_qX_qWY_qqqlv_X350_Y130.npz
  - test_qX_qWY_qqqlv_X400_Y160.npz
```

**Model hyperparameters:**
```yaml
# GNN Baseline
GNN_BASELINE_F_R_LAYER_SIZES: [96, 64, 32]     # Relation network layers
GNN_BASELINE_F_O_LAYER_SIZES: [128, 64, 32]    # Object network layers
GNN_BASELINE_PHI_C_LAYER_SIZES: [16, 8, 4]     # Downstream dense layers
GNN_BASELINE_LEARNING_RATE: 0.0005
GNN_BASELINE_EARLY_STOPPING_PATIENCE: 6
GNN_BASELINE_REDUCE_LR_PATIENCE: 2

# GNN Transformed (with augmentation)
GNN_TRANSFORMED_LEARNING_RATE: 0.0005
GNN_TRANSFORMED_LEARNING_RATE_DECAY: 0.9

# Siamese (SimCLR contrastive learning)
SIAMESE_PHI_C_LAYER_SIZES: [16]                # Encoder output layers
SIAMESE_PROJ_HEAD_LAYER_SIZES: [24, 32]        # Projection head layers
SIAMESE_LEARNING_RATE: 0.0001

# Downstream finetuning
DOWNSTREAM_FREEZE_EPOCH: 3                      # Freeze encoder after this epoch
DOWNSTREAM_LEARNING_RATE: 0.000005             # LR when encoder trainable
DOWNSTREAM_FROZEN_LEARNING_RATE: 0.0001        # LR when encoder frozen
DOWNSTREAM_LR_DECAY: 0.90
DOWNSTREAM_LAYER_SIZES: [8, 4]                 # Downstream head layers

# Training
BATCHSIZE: 128
EPOCHS: 20                                      # GNN baseline/transformed
FINETUNE_EPOCHS: 15                            # Downstream finetuning
SIAMESE_EPOCHS: 30                             # SimCLR pretraining
SIMCLR_LOSS_TEMP: 0.1                          # Temperature for contrastive loss

# Augmentation
NUM_PARTICLES_TO_DELETE: 1                     # Number of particles to randomly delete

# Output
RUN_ID: 3                                      # Model version ID
DATASET_NAME: ""                               # Suffix for output folders
```

### `config_set2.yaml` (Dataset 2)

Similar structure but for the 12-particle, 7-feature dataset:

**Key differences:**
```yaml
DECAY_CHAIN: ["P1", "P2"]       # Two decay chains
N_PARTICLES: 12
N_FEATURES: 7                   # Adds one-hot encoding for gluons
GLUON_IDS: [21]                 # Additional particle type

# Larger network architectures
GNN_BASELINE_F_O_LAYER_SIZES: [128, 96, 64]    # More layers
GNN_BASELINE_PHI_C_LAYER_SIZES: [32, 16, 8]
SIAMESE_PHI_C_LAYER_SIZES: [64]
SIAMESE_PROJ_HEAD_LAYER_SIZES: [96, 128]
DOWNSTREAM_LAYER_SIZES: [128, 64, 32, 16]

# More training epochs
BATCHSIZE: 256
EPOCHS: 30
FINETUNE_EPOCHS: 25
SIAMESE_EPOCHS: 50

# More aggressive augmentation
NUM_PARTICLES_TO_DELETE: 2

# Output suffix
DATASET_NAME: "_set2"
```

---

## Model Training

All training scripts should be run using `python3 -m` from the repository root. Models automatically save to `{module}/model_{RUN_ID}/` or `{module}/model_{RUN_ID}_set2/` directories.

### 1. GNN Baseline (Supervised Learning)

**For Dataset 1:**
```bash
python3 -m gnn_baseline.gnn
```

**For Dataset 2:**
```bash
python3 -m gnn_baseline.gnn --config set2
```

**What it does:**
- Trains a standard Graph Neural Network with supervised learning
- Architecture: GraphEmbeddings (GNN) → Dense layers → Regression output
- Uses early stopping and learning rate reduction on plateau
- Trains on non-augmented data

**Outputs:**
```
gnn_baseline/model_3/          (or model_3_set2/)
├── best_model.keras           # Best model by validation loss
├── last_model.keras           # Final epoch model
├── training_logs.csv          # Epoch-by-epoch metrics
├── results.json               # Test set performance
├── x_scaler_0.pkl            # Feature scalers
├── x_scaler_1.pkl
├── x_scaler_2.pkl
└── y_scaler.pkl
```

### 2. GNN Transformed (With Data Augmentation)

**For Dataset 1:**
```bash
python3 -m gnn_transformed.gnn_transformed
```

**For Dataset 2:**
```bash
python3 -m gnn_transformed.gnn_transformed --config set2
```

**What it does:**
- Same architecture as GNN baseline
- Trains with random particle deletion augmentation
- No early stopping; uses learning rate decay
- Tests robustness to missing particles

**Outputs:** Same structure as GNN baseline

### 3. Siamese (SimCLR Contrastive Learning)

**Step 1: Pretrain encoder with contrastive learning**

**For Dataset 1:**
```bash
python3 -m siamese.siamese
```

**For Dataset 2:**
```bash
python3 -m siamese.siamese --config set2
```

**What it does:**
- Self-supervised pretraining using SimCLR
- Creates augmented view pairs (original + transformed)
- Learns representations by maximizing agreement between views
- Saves encoder separately for downstream tasks

**Outputs:**
```
siamese/model_3/               (or model_3_set2/)
├── best_model.keras           # Full SimCLR model (encoder + projection head)
├── best_model_encoder.keras   # Encoder only (for downstream tasks)
├── training_logs.csv
├── results.json
├── train_embeddings.npz       # Precomputed embeddings for finetuning
├── val_embeddings.npz
└── test_embeddings.npz
```

**Step 2a: Finetune for downstream task (recommended)**

**For Dataset 1:**
```bash
python3 -m siamese.train_finetune
```

**For Dataset 2:**
```bash
python3 -m siamese.train_finetune --config set2
```

**What it does:**
- Loads pretrained encoder from Step 1
- Progressive finetuning:
  - Epochs 0 to `DOWNSTREAM_FREEZE_EPOCH`: Encoder trainable with low LR
  - After `DOWNSTREAM_FREEZE_EPOCH`: Encoder frozen with higher LR
- Adds downstream regression head

**Outputs:**
```
siamese/model_3_finetune/      (or model_3_finetune_set2/)
├── best_model.keras
├── last_model.keras
├── training_logs.csv
└── results.json
```

**Step 2b: Train with frozen encoder (alternative)**

**For Dataset 1:**
```bash
python3 -m siamese.train_no_finetune
```

**For Dataset 2:**
```bash
python3 -m siamese.train_no_finetune --config set2
```

**What it does:**
- Loads pretrained encoder and keeps it frozen
- Only trains downstream regression head
- Faster training; tests pure representation quality

**Outputs:** Same structure as finetune variant

### 4. Lorentz Addition (Physics Baseline)

**For Dataset 1:**
```bash
python3 -m lorentz_addition.lorentz_addition
```

**For Dataset 2:**
```bash
python3 -m lorentz_addition.lorentz_addition_set2
```

**What it does:**
- Applies physics-based approach (no ML)
- Sums 4-momentum vectors of all visible particles
- Assumes MET pseudorapidity = 0
- Serves as baseline for comparing ML models

**Outputs:**
```
lorentz_addition/
├── lorentz_addition_results.json
└── graphs/
    └── [histograms for each event type]
```

---

## Model Evaluation

All evaluation scripts should be run from the repository root using `python3 -m`. They automatically detect which models exist and evaluate them.

### 1. Same Event Type Evaluation

Evaluates models separately on each test file (event type):

**For Dataset 1:**
```bash
python3 -m model_evaluation.same_event_type
```

**For Dataset 2:**
```bash
python3 -m model_evaluation.same_event_type --config set2
```

**What it does:**
- Loads all trained models (GNN baseline, transformed, siamese variants)
- Evaluates each model on every test file individually
- Generates dual histograms comparing model predictions
- Creates accuracy plots showing performance across event types
- Saves per-file metrics to JSON

**Outputs:**
```
model_evaluation/
├── dual_histograms/                    (or dual_histograms_set2/)
│   ├── test_qX_qWY_qqqlv_X200_Y60.png
│   └── ...
├── accuracy_plots/                     (or accuracy_plots_set2/)
│   ├── mae_comparison.png
│   ├── rmse_comparison.png
│   └── r2_comparison.png
└── json/                               (or json_set2/)
    └── same_event_type_metrics.json
```

### 2. Full Dataset Evaluation

Evaluates models on the complete combined test set:

**For Dataset 1:**
```bash
python3 -m model_evaluation.full_dataset
```

**For Dataset 2:**
```bash
python3 -m model_evaluation.full_dataset --config set2
```

**What it does:**
- Loads all trained models
- Evaluates on entire test dataset (all files combined)
- Performs variance analysis by bucketing predictions
- Compares overall accuracy across models

**Outputs:**
```
model_evaluation/
├── accuracy_plots/                     (or accuracy_plots_set2/)
│   └── variance_comparison.png
└── json/                               (or json_set2/)
    └── metrics.json
```

### 3. Transformed Input Evaluation

Tests model robustness by evaluating on augmented/transformed inputs:

**For Dataset 1:**
```bash
python3 -m model_evaluation.transformed_same_event_type
python3 -m model_evaluation.transformed_full_dataset
```

**For Dataset 2:**
```bash
python3 -m model_evaluation.transformed_same_event_type --config set2
python3 -m model_evaluation.transformed_full_dataset --config set2
```

**What it does:**
- Same as regular evaluation but applies random particle deletion to test inputs
- Tests whether models maintain performance under perturbations
- Compares robustness across different model types

**Outputs:**
```
model_evaluation/
├── transformed_dual_histograms/        (or transformed_dual_histograms_set2/)
├── transformed_accuracy_plots/         (or transformed_accuracy_plots_set2/)
└── json/                               (or json_set2/)
    ├── transformed_same_event_type_metrics.json
    └── transformed_metrics.json
```

### 4. Background Data Evaluation

Tests models on background (non-SUSY) events:

**For Dataset 1:**
```bash
python3 -m model_evaluation.background_test
```

**For Dataset 2:**
```bash
python3 -m model_evaluation.background_test --config set2
```

**What it does:**
- Applies all trained models to Standard Model background events
- Creates pairwise comparison histograms between models
- Ideal models should show distinctive predictions for background vs. signal
- Helps assess whether models learned physics or just overfitted

**Outputs:**
```
model_evaluation/
└── background_plots/                   (or background_plots_set2/)
    ├── gnn_baseline_vs_gnn_transformed.png
    ├── gnn_baseline_vs_siamese_finetune.png
    └── ...
```

### Understanding Evaluation Outputs

**Metrics in JSON files:**
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `r2`: R² coefficient of determination
- `avg_relative_error`: Average relative error (|predicted - true| / true)
- `median_relative_error`: Median relative error

**Dual histograms:**
- Blue: True mass distribution
- Orange: Predicted mass distribution
- Vertical lines: Mean values
- Inset text: Performance metrics

**Accuracy plots:**
- X-axis: Different event types (or mass values)
- Y-axis: Performance metric (MAE, RMSE, or R²)
- Multiple lines: Different models for comparison

---

## Background Data Functionality

### Purpose

Background data consists of Standard Model physics events (non-SUSY processes) used to test whether models can distinguish signal from background. This is crucial for:
- Testing if models learned genuine physics patterns vs. overfitting
- Assessing generalization to out-of-distribution data
- Validating that models don't produce spurious predictions on background

### Data Structure

**Raw data:**
```
raw_data_background/
└── test_TbqqTblv.root          # Top quark pair production events
```

**Preprocessed data:**
```
processed_data_background/
└── test_TbqqTblv.npz
```

### Preprocessing Background Data

**For Dataset 1:**
```bash
python3 preprocess_data_background.py
```

**For Dataset 2:**
```bash
python3 preprocess_data_background.py --dataset set2
```

**What it does:**
- Extracts features from two separate decay chains (P1 and P2)
- Combines specific particles from each chain
- Formats data identically to signal data (same shape, same features)
- No target mass labels (background events don't contain SUSY particles)

### Evaluating on Background Data

```bash
python3 -m model_evaluation.background_test
python3 -m model_evaluation.background_test --config set2
```

**Interpretation:**
- Models trained on SUSY signal should produce different predictions for background
- Consistent predictions between signal and background may indicate overfitting
- Distribution shape differences help assess model discriminative power

**Outputs:**
- Pairwise comparison histograms between all model combinations
- Shows how different models respond to non-SUSY events
- Saved to `model_evaluation/background_plots/`

---

## Additional Notes

### Model Versioning

The `RUN_ID` parameter in config files controls model versioning:
```yaml
RUN_ID: 3
```

This creates directories like `model_3/`, `model_3_finetune/`, etc. Increment this value to train new model versions without overwriting previous runs.

### Dataset Selection

Most scripts accept `--config set2` or `--dataset set2` to use Dataset 2:
```bash
python3 -m gnn_baseline.gnn --config set2
python3 preprocess_data.py --dataset set2
```

Without this flag, Dataset 1 (config.yaml) is used by default.

### Data Scalers

Models save `StandardScaler` objects during training:
- `x_scaler_0.pkl`, `x_scaler_1.pkl`, `x_scaler_2.pkl`: Scale pt, eta, phi features
- `y_scaler.pkl`: Scale target mass values

These must be loaded during evaluation to properly scale inputs/outputs.

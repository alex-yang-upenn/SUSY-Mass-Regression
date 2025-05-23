# Predicting Mass of Supersymmetric Particles via Regression on Decay Chain

### Workflow

The majority of the relevant constants and hyperparameters are defined in `config.py`. If any of the models or training data need small adjustments, this can be done by editing config.py, then re-running the relevant training scripts.

#### Formatting Training Data
First, install ROOT and the PyROOT bindings, and copy the root files into the processed_data directory. Run `preprocess_data.py`. Currently, the data is formatted with shape (N_EVENTS, N_PARTICLES, N_FEATURES). However, the GNN model expects (N_EVENTS, N_FEATURES, N_PARTICLES). Thus, the `load_data()` helper method handles this transformation.

> Note: `preprocess_data.py` works for an arbitrary number of particles. However, it only considers a single decay chain of the TTree (with the branch prefix specified in `config.py`) and it always formats the training data according to the description at the top of the file. This file should be customized if more flexibility is needed.

#### Training Models
Run `lorentz_addition/lorentz_addiction.py` to check the data and to get a no-ML performance baseline.

Run `gnn_baseline/gnn.py` to obtain a basic GNN model trained on an unmodified dataset.

Run `siamese/siamese.py` to obtain an unsupervised contrastive learning model trained on a transformed dataset.

#### Evaluation
Run `full_dataset.py` to evaluate trained models on the full unmodified test dataset. Evaluations results are stored in `metrics.json`.

Run `same_event_type.py` to evaluate trained models on the unmodified test dataset, file by file. The resulting accuracy is graphed in `accuracy_plots/standard_inputs.png`. A small subset of the files will have evaluation results stored in `same_event_type_metrics.json` and histograms comparing model performance stored under `dual_histograms`.

Run `transformed_full_dataset.py` to evaluate trained models on the full test dataset, with transformations/distortions applied. Evaluations results are stored in `transformed_metrics.json`.

Run `transformed_same_event_type.py` to evaluate trained models on the test dataset, file by file, with transformations/distortions applied. The resulting accuracy is graphed in `accuracy_plots/transformed_inputs.png`. A small subset of the files will have evaluation results stored in `transformed_same_event_type_metrics.json` and histograms comparing model performance stored under `transformed_dual_histograms`.
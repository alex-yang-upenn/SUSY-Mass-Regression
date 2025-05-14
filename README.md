# Predicting Mass of Supersymmetric Particles via Regression on Decay Chain

### Workflow

The majority of the relevant constants and hyperparameters are defined in config.py. If any of the models or training data need small adjustments, this can be done by editing config.py, then re-running the relevant training scripts.

#### Formatting Training Data
First, install ROOT and the PyROOT bindings, and copy the root files into the processed_data directory. Run preprocess_data.py.

> Note: preprocess_data.py works for an arbitrary number of particles. However, it only considers a single decay chain of the TTree (with the branch prefix specified in config.py) and it always formats the training data according to the description at the top of the file. This file should be customized if more flexibility is needed.

#### Training Models
Run lorentz_addition/lorentz_addiction.py to check the data and to get a no-ML performance baseline.

Run gnn_baseline/gnn.py to obtain a basic GNN model trained on an unmodified dataset.

Run siamese/siamese.py to obtain an unsupervised contrastive learning model trained on a transformed dataset.


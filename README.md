# hts-constrained-embeddings

This repository contains code to reproduce the experiments in **"Forecasting Hierarchical Time Series with a Regularized Embedding Space"**

# Data

The Australian travel flow data used in experiments can be downloaded here: https://robjhyndman.com/publications/hierarchical-tourism/.

# Experiments

Run the `experiment.py` script with the following flags to reproduce:

1. **Initial Experiment**:
    1. `python experiment.py reproduce --metrics_file='data/metrics/baseline.txt' --output_path='data/preds_baseline' --serialize_path='data/models_baseline' --reconciled_path='data/reconciled_preds_baseline'`

    2. *in R*: `reconcile_all(in_dir = 'preds_baseline', out_dir = 'reconciled_preds_baseline')`

    3. `python experiment.py optimal_reconciliation --metrics_file='data/metrics/baseline.txt' --output_path='data/preds_baseline' --serialize_path='data/models_baseline' --reconciled_path='data/reconciled_preds_baseline'`

2. **Short Training Sequences**:
    1. `python experiment.py reproduce --train_size=54 --metrics_file='data/metrics/small.txt' --output_path='data/preds_small' --serialize_path='data/models_small' --reconciled_path='data/reconciled_preds_small'`

    2. *in R*: `reconcile_all(in_dir = 'preds_small', out_dir = 'reconciled_preds_small')`

    3. `python experiment.py optimal_reconciliation --train_size=54 --metrics_file='data/metrics/small.txt' --output_path='data/preds_small' --serialize_path='data/models_small' --reconciled_path='data/reconciled_preds_small'`

3. **Long Forecast Horizon**:
    1. `python experiment.py reproduce --horizon=24 --metrics_file='data/metrics/horizon.txt' --output_path='data/preds_horizon' --serialize_path='data/models_horizon' --reconciled_path='data/reconciled_preds_horizon'`

    2. *in R*: `reconcile_all(in_dir = 'preds_horizon', out_dir = 'reconciled_preds_horizon', horizon = 24)`

    3. `python experiment.py optimal_reconciliation --horizon=24 --metrics_file='data/metrics/horizon.txt' --output_path='data/preds_horizon' --serialize_path='data/models_horizon' --reconciled_path='data/reconciled_preds_horizon'`

4. **Relative Embedding Dimension: 0.5**:
`python experiment.py reproduce --embed_dim_ratio=0.5 --metrics_file='data/metrics/edim05.txt' --output_path='data/preds_edim05' --serialize_path='data/models_edim05' --reconciled_path='data/reconciled_preds_edim05'`

5. **Relative Embedding Dimension: 2**:
`python experiment.py reproduce --embed_dim_ratio=2 --metrics_file='data/metrics/edim2.txt' --output_path='data/preds_edim2' --serialize_path='data/models_edim2' --reconciled_path='data/reconciled_preds_edim2'`

7. **Relative Embedding Dimension: 4**:
`python experiment.py reproduce --embed_dim_ratio=4 --metrics_file='data/metrics/edim4.txt' --output_path='data/preds_edim4' --serialize_path='data/models_edim4' --reconciled_path='data/reconciled_preds_edim4'`

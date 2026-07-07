# WhiteMatterHomogenization

Code, processed datasets, and trained surrogate-model artifacts for histology-informed homogenization of brain white matter.

## Repository layout

- `data/case_tables/`
  - campaign input tables used to define the 1000-case Latin hypercube study
- `data/processed/`
  - public-safe processed datasets with machine-specific path columns removed
- `models/single_component/`
  - trained MLP weights and evaluation artifacts for the single-component pressure target
- `scripts/`
  - RVE generation, Abaqus campaign, dataset rebuild, training, and inference utilities

## Included public datasets

- `data/processed/ml_hgo_dataset_lhs_1000_fixed_matrix_public.csv`
  - fixed-matrix homogenization dataset with 1000 cases
- `data/processed/ml_hgo_dataset_lhs_1000_single_component_pressure_public.csv`
  - single-component pressure-fit dataset with 1000 cases
- `data/processed/ml_hgo_dataset_lhs_1000_single_component_pressure_with_interaction_public.csv`
  - same single-component dataset plus the derived stiffness-interaction feature used by the published MLP model

Absolute local path fields from the original campaign files were removed before publication.

## Trained model package

The main published surrogate package is in `models/single_component/`:

- `mlp_fixed_matrix_model_weights.npz`
- `mlp_fixed_matrix_run_config.json`
- `mlp_fixed_matrix_metrics.json`
- `mlp_fixed_matrix_metrics_summary.csv`
- `mlp_fixed_matrix_data_split.json`
- `mlp_fixed_matrix_predictions.csv`
- `mlp_fixed_matrix_training_history.csv`

The weights file stores the learned network parameters together with the input and target normalization statistics.

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run inference with the released model:

```bash
python scripts/predict_mlp_hgo_single_component.py data/processed/ml_hgo_dataset_lhs_1000_single_component_pressure_with_interaction_public.csv
```

This writes a prediction file next to the input CSV.

## Retraining

To retrain the single-component model using the public processed dataset:

```bash
python scripts/train_mlp_hgo_lhs_1000.py --dataset-mode single_component
```

Retraining outputs will be written to `models/single_component_retrained/`.

## Abaqus-dependent workflow

The full RVE-generation and homogenization workflow is retained in `scripts/`, including:

- dual-target RVE generators
- Abaqus batch runner
- fixed-matrix refit utilities
- dataset rebuild utilities
- the Fortran user subroutine for recruited-stretch axonal response

These parts require a local Abaqus installation and are not exercised by the lightweight inference workflow above.

## Notes

- The processed public CSV files are the recommended starting point for reuse because they remove machine-specific path fields from the original campaign exports.
- Large raw Abaqus outputs, intermediate run folders, and rebuilt private working datasets are intentionally excluded from version control.
- The repository intentionally excludes large raw Abaqus result files and ODB outputs.
- Processed datasets and trained-model artifacts are included for reproducibility of the surrogate-model results.

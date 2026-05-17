# Current Work Optimized Results

This folder collects the latest optimized ln-feature concentration modeling results for the application package.

## Application Model

The deployable model entry is registered as `current_work` in `application/model_registry.json`.

Model files:

- `application/models/current_work/rgb_concentration_model.pkl`
- `application/models/current_work/exp_fit_params.csv`
- `application/models/current_work/gs_amortized_posterior_model.pt`
- `application/models/current_work/best_optimized_combinations.csv`

## Result Materials

- `csv/best_optimized_combinations.csv`: selected input vectors and hyperparameters.
- `csv/optimization/`: input-vector and hyperparameter search records.
- `csv/train/`, `csv/test/`, `csv/loo/`: predictions, metrics, recovery summaries, and heatmap tables.
- `fig/train/`, `fig/test/`, `fig/loo/`: recovery mean/RSD heatmaps.

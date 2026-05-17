# Recovery and RSD Results

ROI CSV: `/home/jzh/CDs/dataset/food/roi_detailed_statistics.csv`

Prediction directory: `/home/jzh/CDs/OUTPUT/food/mate20x_model_predictions`

Formula:

- Recovery: `R(%) = predicted_concentration / added_concentration * 100`.
- For 0 concentration rows, recovery is undefined and written as blank, while detected concentration is retained.
- RSD: standard deviation of replicate recoveries divided by mean recovery.

Output files:

- `model_recovery_samples.csv`
- `model_recovery_summary.csv`
- `model_recovery_heatmap_table.csv`
- `figures/model_recovery_heatmap_rsd.png`

## Model Recovery Summary Head

| source | model | prediction_column | matrix | added_concentration_c3 | original_concentration_c2 | n | detected_c1_mean | detected_c1_sd | recovery_mean_percent | recovery_sd_percent | recovery_rsd_percent | meets_80_110_recovery | meets_rsd_lt_3 | meets_recovery_and_rsd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_prediction | exp | conc_pred | chicken | 0 | 0 | 0 | 11.4194 |  |  |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 30 | 0 | 1 | 16.5817 |  | 55.2723 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 60 | 0 | 1 | 33.1067 |  | 55.1778 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 90 | 0 | 1 | 29.1588 |  | 32.3987 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 120 | 0 | 1 | 44.0526 |  | 36.7105 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 150 | 0 | 1 | 54.8112 |  | 36.5408 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 180 | 0 | 1 | 56.3449 |  | 31.3027 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 210 | 0 | 1 | 67.8216 |  | 32.296 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 240 | 0 | 1 | 74.2925 |  | 30.9552 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 0 | 0 | 0 | 14.8898 |  |  |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 30 | 0 | 1 | 20.2354 |  | 67.4513 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 60 | 0 | 1 | 33.7495 |  | 56.2492 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 90 | 0 | 1 | 43.6324 |  | 48.4804 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 120 | 0 | 1 | 41.1573 |  | 34.2978 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 150 | 0 | 1 | 57.1232 |  | 38.0821 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 180 | 0 | 1 | 49.2592 |  | 27.3662 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 210 | 0 | 1 | 67.7596 |  | 32.2665 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 240 | 0 | 1 | 72.8395 |  | 30.3498 |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 0 | 0 | 0 | 0.0165549 |  |  |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 30 | 0 | 1 | 0.0791669 |  | 0.26389 |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 60 | 0 | 1 | 0.00281682 |  | 0.0046947 |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 90 | 0 | 1 | 0.0715309 |  | 0.0794788 |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 120 | 0 | 1 | 0.017185 |  | 0.0143208 |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 150 | 0 | 1 | 0.175445 |  | 0.116963 |  |  | False | False | False |

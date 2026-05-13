# Recovery and RSD Results

ROI CSV: `/home/jzh/CDs/dataset/test/roi_results/roi_detailed_statistics.csv`

Prediction directory: `/home/jzh/CDs/OUTPUT/results`

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
| model_prediction | exp | conc_pred | chicken | 0 | 0 | 0 | 0 |  |  |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 3 | 0 | 1 | 8.16163 |  | 272.054 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 6 | 0 | 1 | 11.7143 |  | 195.238 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 9 | 0 | 1 | 18.0516 |  | 200.573 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 18 | 0 | 1 | 24.2929 |  | 134.961 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 30 | 0 | 1 | 31.5903 |  | 105.301 |  |  | True | False | False |
| model_prediction | exp | conc_pred | chicken | 50 | 0 | 1 | 38.2156 |  | 76.4312 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 90 | 0 | 1 | 73.2146 |  | 81.3496 |  |  | True | False | False |
| model_prediction | exp | conc_pred | chicken | 110 | 0 | 1 | 63.8528 |  | 58.048 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 150 | 0 | 1 | 95.011 |  | 63.3407 |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 240 | 0 | 1 | 178.836 |  | 74.515 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 0 | 0 | 0 | 0.288058 |  |  |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 3 | 0 | 1 | 7.92158 |  | 264.053 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 6 | 0 | 1 | 14.2108 |  | 236.847 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 8 | 0 | 1 | 16.2753 |  | 203.441 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 18 | 0 | 1 | 18.5797 |  | 103.221 |  |  | True | False | False |
| model_prediction | exp | conc_pred | egg | 30 | 0 | 1 | 18.5317 |  | 61.7723 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 50 | 0 | 1 | 42.9206 |  | 85.8412 |  |  | True | False | False |
| model_prediction | exp | conc_pred | egg | 90 | 0 | 1 | 56.3153 |  | 62.5726 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 110 | 0 | 1 | 70.142 |  | 63.7655 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 150 | 0 | 1 | 101.828 |  | 67.8853 |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 240 | 0 | 1 | 143.357 |  | 59.7321 |  |  | False | False | False |
| model_prediction | rgb_ridge | conc_pred | chicken | 0 | 0 | 0 | 0.172023 |  |  |  |  | False | False | False |
| model_prediction | rgb_ridge | conc_pred | chicken | 3 | 0 | 1 | 0.940768 |  | 31.3589 |  |  | False | False | False |

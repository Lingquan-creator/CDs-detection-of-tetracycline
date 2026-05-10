# Recovery and RSD Results

ROI CSV: `/home/jzh/CDs/dataset/RAW/roi_results/roi_detailed_statistics.csv`

Prediction directory: `/home/jzh/CDs/OUTPUT/RAW/mate20x_model_predictions`

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
| model_prediction | exp | conc_pred | chicken | 0 | 0 | 0 | 6.54184 | 1.69068 |  |  |  | False | False | False |
| model_prediction | exp | conc_pred | chicken | 2 | 0 | 3 | 11.5152 | 2.22736 | 575.762 | 111.368 | 19.3428 | False | False | False |
| model_prediction | exp | conc_pred | chicken | 10 | 0 | 3 | 14.8034 | 4.69733 | 148.034 | 46.9733 | 31.7315 | False | False | False |
| model_prediction | exp | conc_pred | chicken | 20 | 0 | 3 | 20.0651 | 2.82994 | 100.326 | 14.1497 | 14.1038 | True | False | False |
| model_prediction | exp | conc_pred | chicken | 50 | 0 | 3 | 28.2243 | 4.22826 | 56.4485 | 8.45651 | 14.9809 | False | False | False |
| model_prediction | exp | conc_pred | chicken | 100 | 0 | 3 | 42.1788 | 6.97647 | 42.1788 | 6.97647 | 16.5402 | False | False | False |
| model_prediction | exp | conc_pred | egg | 0 | 0 | 0 | 5.60318 | 0.0534785 |  |  |  | False | False | False |
| model_prediction | exp | conc_pred | egg | 2 | 0 | 2 | 11.2976 | 3.44683 | 564.881 | 172.341 | 30.5093 | False | False | False |
| model_prediction | exp | conc_pred | egg | 10 | 0 | 2 | 9.74904 | 1.31728 | 97.4904 | 13.1728 | 13.5119 | True | False | False |
| model_prediction | exp | conc_pred | egg | 20 | 0 | 2 | 12.8613 | 3.56856 | 64.3063 | 17.8428 | 27.7466 | False | False | False |
| model_prediction | exp | conc_pred | egg | 50 | 0 | 2 | 17.7859 | 1.15357 | 35.5718 | 2.30715 | 6.48589 | False | False | False |
| model_prediction | exp | conc_pred | egg | 100 | 0 | 2 | 32.1803 | 0.435931 | 32.1803 | 0.435931 | 1.35465 | False | True | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 0 | 0 | 0 | 0.00712478 | 0.0073438 |  |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 2 | 0 | 3 | 0.533317 | 0.816605 | 26.6658 | 40.8302 | 153.118 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 10 | 0 | 3 | 0.128638 | 0.168132 | 1.28638 | 1.68132 | 130.702 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 20 | 0 | 3 | 0.102347 | 0.0956415 | 0.511737 | 0.478208 | 93.4479 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 50 | 0 | 3 | 0.016317 | 0.0217898 | 0.0326341 | 0.0435797 | 133.54 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 100 | 0 | 3 | 0.0127229 | 0.010219 | 0.0127229 | 0.010219 | 80.3196 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 0 | 0 | 0 | 0.0973377 | 0.132003 |  |  |  | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 2 | 0 | 2 | 3.22522 | 3.73068 | 161.261 | 186.534 | 115.672 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 10 | 0 | 2 | 1.88162 | 1.70596 | 18.8162 | 17.0596 | 90.6644 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 20 | 0 | 2 | 0.44485 | 0.0490258 | 2.22425 | 0.245129 | 11.0208 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 50 | 0 | 2 | 1.23568 | 1.28744 | 2.47137 | 2.57488 | 104.188 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 100 | 0 | 2 | 0.043944 | 0.026695 | 0.043944 | 0.026695 | 60.7478 | False | False | False |

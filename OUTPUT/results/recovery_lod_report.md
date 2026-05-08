# Recovery and LOD Results

ROI CSV: `/home/jzh/CDs/dataset/RAW/roi_results/roi_detailed_statistics.csv`

Prediction directory: `/home/jzh/CDs/OUTPUT/RAW/mate20x_model_predictions`

Formulas from `temp` images:

- Recovery: `R(%) = (c1 - c2) / c3 * 100`; this run uses `c2 = 0`.
- LOD: `LOD = 3 * sigma / S`; this run uses `abs(S)` because several signals decrease with concentration.

Output files:

- `signal_recovery_samples.csv`
- `signal_recovery_summary.csv`
- `lod_results.csv`
- `model_recovery_samples.csv`
- `model_recovery_summary.csv`

## Primary LOD Rows (G signal)

| matrix | signal | slope_S | intercept | r2 | blank_n | blank_sigma | lod | lod_formula |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chicken | G_sig | -7.25922 | 1151.46 | 0.808359 | 3 | 68.6709 | 28.3795 | LOD=3*sigma/abs(S) |
| chicken | G_sig_calc | -7.25922 | 1151.46 | 0.808359 | 3 | 68.6709 | 28.3795 | LOD=3*sigma/abs(S) |
| egg | G_sig | -6.29491 | 1254.04 | 0.888731 | 2 | 2.49149 | 1.18738 | LOD=3*sigma/abs(S) |
| egg | G_sig_calc | -6.29491 | 1254.04 | 0.888731 | 2 | 2.49149 | 1.18738 | LOD=3*sigma/abs(S) |
| milk | G_sig | -7.15854 | 983.327 | 0.701858 | 3 | 402.089 | 168.507 | LOD=3*sigma/abs(S) |
| milk | G_sig_calc | -7.15854 | 983.327 | 0.701858 | 3 | 402.089 | 168.507 | LOD=3*sigma/abs(S) |

## Model Recovery Summary Head

| source | model | prediction_column | matrix | added_concentration_c3 | original_concentration_c2 | n | detected_c1_mean | detected_c1_sd | recovery_mean_percent | recovery_sd_percent | recovery_rsd_percent | meets_80_110_recovery | meets_rsd_lt_3 | meets_recovery_and_rsd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_prediction | exp | conc_pred | chicken | 2 | 0 | 3 | 11.5152 | 2.22736 | 575.762 | 111.368 | 19.3428 | False | False | False |
| model_prediction | exp | conc_pred | chicken | 10 | 0 | 3 | 14.8034 | 4.69733 | 148.034 | 46.9733 | 31.7315 | False | False | False |
| model_prediction | exp | conc_pred | chicken | 20 | 0 | 3 | 20.0651 | 2.82994 | 100.326 | 14.1497 | 14.1038 | True | False | False |
| model_prediction | exp | conc_pred | chicken | 50 | 0 | 3 | 28.2243 | 4.22826 | 56.4485 | 8.45651 | 14.9809 | False | False | False |
| model_prediction | exp | conc_pred | chicken | 100 | 0 | 3 | 42.1788 | 6.97647 | 42.1788 | 6.97647 | 16.5402 | False | False | False |
| model_prediction | exp | conc_pred | egg | 2 | 0 | 2 | 11.2976 | 3.44683 | 564.881 | 172.341 | 30.5093 | False | False | False |
| model_prediction | exp | conc_pred | egg | 10 | 0 | 2 | 9.74904 | 1.31728 | 97.4904 | 13.1728 | 13.5119 | True | False | False |
| model_prediction | exp | conc_pred | egg | 20 | 0 | 2 | 12.8613 | 3.56856 | 64.3063 | 17.8428 | 27.7466 | False | False | False |
| model_prediction | exp | conc_pred | egg | 50 | 0 | 2 | 17.7859 | 1.15357 | 35.5718 | 2.30715 | 6.48589 | False | False | False |
| model_prediction | exp | conc_pred | egg | 100 | 0 | 2 | 32.1803 | 0.435931 | 32.1803 | 0.435931 | 1.35465 | False | True | False |
| model_prediction | exp | conc_pred | milk | 2 | 0 | 3 | 16.7637 | 8.29066 | 838.183 | 414.533 | 49.4562 | False | False | False |
| model_prediction | exp | conc_pred | milk | 10 | 0 | 3 | 16.8626 | 2.3451 | 168.626 | 23.451 | 13.9072 | False | False | False |
| model_prediction | exp | conc_pred | milk | 20 | 0 | 3 | 22.3277 | 2.85506 | 111.638 | 14.2753 | 12.7871 | False | False | False |
| model_prediction | exp | conc_pred | milk | 50 | 0 | 3 | 37.8635 | 4.30023 | 75.7271 | 8.60046 | 11.3572 | False | False | False |
| model_prediction | exp | conc_pred | milk | 100 | 0 | 3 | 65.6713 | 8.6014 | 65.6713 | 8.6014 | 13.0977 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 2 | 0 | 3 | 28.6261 | 27.4841 | 1431.31 | 1374.2 | 96.0105 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 10 | 0 | 3 | 45.2813 | 28.1584 | 452.813 | 281.584 | 62.1855 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 20 | 0 | 3 | 24.6776 | 6.55016 | 123.388 | 32.7508 | 26.543 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 50 | 0 | 3 | 13.4485 | 6.67583 | 26.8971 | 13.3517 | 49.6398 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | chicken | 100 | 0 | 3 | 34.114 | 6.69919 | 34.114 | 6.69919 | 19.6377 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 2 | 0 | 2 | 11.0794 | 0.53634 | 553.967 | 26.817 | 4.8409 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 10 | 0 | 2 | 19.9707 | 6.51839 | 199.707 | 65.1839 | 32.6398 | False | False | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 20 | 0 | 2 | 25.9511 | 0.237376 | 129.755 | 1.18688 | 0.914706 | False | True | False |
| model_prediction | gs_posterior_mean | posterior_mean | egg | 50 | 0 | 2 | 47.9755 | 3.5764 | 95.951 | 7.15281 | 7.45465 | True | False | False |

## Signal Recovery Summary Head

| source | matrix | signal | added_concentration_c3 | original_concentration_c2 | n | detected_c1_mean | detected_c1_sd | recovery_mean_percent | recovery_sd_percent | recovery_rsd_percent | meets_80_110_recovery | meets_rsd_lt_3 | meets_recovery_and_rsd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| signal_linear_calibration | chicken | R_sig | 2 | 0 | 3 | 1.08223 | 11.1154 | 54.1113 | 555.771 | 1027.09 | False | False | False |
| signal_linear_calibration | chicken | R_sig | 10 | 0 | 3 | 15.0754 | 21.5527 | 150.754 | 215.527 | 142.966 | False | False | False |
| signal_linear_calibration | chicken | R_sig | 20 | 0 | 3 | 35.8996 | 10.4089 | 179.498 | 52.0443 | 28.9944 | False | False | False |
| signal_linear_calibration | chicken | R_sig | 50 | 0 | 3 | 60.8706 | 12.0398 | 121.741 | 24.0796 | 19.7793 | False | False | False |
| signal_linear_calibration | chicken | R_sig | 100 | 0 | 3 | 90.8956 | 13.4467 | 90.8956 | 13.4467 | 14.7936 | True | False | False |
| signal_linear_calibration | chicken | G1_sig_calc | 2 | 0 | 3 | 1.72974 | 10.7858 | 86.487 | 539.291 | 623.552 | True | False | False |
| signal_linear_calibration | chicken | G1_sig_calc | 10 | 0 | 3 | 15.5989 | 20.845 | 155.989 | 208.45 | 133.631 | False | False | False |
| signal_linear_calibration | chicken | G1_sig_calc | 20 | 0 | 3 | 36.6991 | 9.70716 | 183.496 | 48.5358 | 26.4507 | False | False | False |
| signal_linear_calibration | chicken | G1_sig_calc | 50 | 0 | 3 | 61.4735 | 11.0792 | 122.947 | 22.1584 | 18.0227 | False | False | False |
| signal_linear_calibration | chicken | G1_sig_calc | 100 | 0 | 3 | 90.3689 | 12.2972 | 90.3689 | 12.2972 | 13.6078 | True | False | False |
| signal_linear_calibration | chicken | G2_sig_calc | 2 | 0 | 3 | 1.74019 | 10.7262 | 87.0094 | 536.308 | 616.379 | True | False | False |
| signal_linear_calibration | chicken | G2_sig_calc | 10 | 0 | 3 | 15.5715 | 20.8484 | 155.715 | 208.484 | 133.888 | False | False | False |
| signal_linear_calibration | chicken | G2_sig_calc | 20 | 0 | 3 | 36.6717 | 9.67429 | 183.358 | 48.3714 | 26.3808 | False | False | False |
| signal_linear_calibration | chicken | G2_sig_calc | 50 | 0 | 3 | 61.4504 | 11.0942 | 122.901 | 22.1884 | 18.0539 | False | False | False |
| signal_linear_calibration | chicken | G2_sig_calc | 100 | 0 | 3 | 90.3885 | 12.2947 | 90.3885 | 12.2947 | 13.602 | True | False | False |
| signal_linear_calibration | chicken | G_sig | 2 | 0 | 3 | 1.73497 | 10.756 | 86.7483 | 537.798 | 619.953 | True | False | False |
| signal_linear_calibration | chicken | G_sig | 10 | 0 | 3 | 15.5852 | 20.8467 | 155.852 | 208.467 | 133.759 | False | False | False |
| signal_linear_calibration | chicken | G_sig | 20 | 0 | 3 | 36.6854 | 9.69071 | 183.427 | 48.4536 | 26.4157 | False | False | False |
| signal_linear_calibration | chicken | G_sig | 50 | 0 | 3 | 61.4619 | 11.0867 | 122.924 | 22.1734 | 18.0383 | False | False | False |
| signal_linear_calibration | chicken | G_sig | 100 | 0 | 3 | 90.3787 | 12.2959 | 90.3787 | 12.2959 | 13.6049 | True | False | False |
| signal_linear_calibration | chicken | G_sig_calc | 2 | 0 | 3 | 1.73497 | 10.756 | 86.7483 | 537.798 | 619.953 | True | False | False |
| signal_linear_calibration | chicken | G_sig_calc | 10 | 0 | 3 | 15.5852 | 20.8467 | 155.852 | 208.467 | 133.759 | False | False | False |
| signal_linear_calibration | chicken | G_sig_calc | 20 | 0 | 3 | 36.6854 | 9.69071 | 183.427 | 48.4536 | 26.4157 | False | False | False |
| signal_linear_calibration | chicken | G_sig_calc | 50 | 0 | 3 | 61.4619 | 11.0867 | 122.924 | 22.1734 | 18.0383 | False | False | False |

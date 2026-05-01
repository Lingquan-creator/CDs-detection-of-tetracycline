"""
Quick example: Fit exponential model and predict concentrations.

This script demonstrates the basic workflow:
1. Fit exponential model to training data
2. Load parameters and predict on new data

Prerequisites:
- ROI statistics CSV from pic_RGB pipeline
- Filenames must contain concentration values (e.g., "12.5.dng")
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import ExpConfig
from train import run_exponential_fitting, parse_args as parse_fit_args
from predict import predict_concentration, parse_args as parse_predict_args


def example_workflow():
    """
    Example workflow for exponential fitting and prediction.
    """
    
    print("=" * 70)
    print("EXPONENTIAL FITTING EXAMPLE WORKFLOW")
    print("=" * 70)
    
    # Configuration
    cfg = ExpConfig()
    
    print("\nConfiguration:")
    print(f"  Input CSV: {cfg.roi_csv}")
    print(f"  Output directory: {cfg.output_dir}")
    print(f"  Default channel: {cfg.channel}")
    print(f"  Weight epsilon: {float(cfg.weight_eps):.6g}")
    
    # Check if input file exists
    if not cfg.roi_csv.exists():
        print(f"\nERROR: Input file not found: {cfg.roi_csv}")
        print("\nPlease ensure you have run the pic_RGB pipeline first to generate ROI statistics.")
        print("Expected path: pic_RGB/roi_results/roi_detailed_statistics.csv")
        return
    
    print("\n" + "=" * 70)
    print("STEP 1: FITTING EXPONENTIAL MODEL")
    print("=" * 70)
    print("\nCommand equivalent:")
    print(f"  python concentration_EXP/train.py")
    print()
    
    # Note: In actual use, you would run train.py directly
    # This is just a demonstration
    print("Tip: Run the fitting script with:")
    print("  python concentration_EXP/train.py --channel G_corr --weight-eps 5.0")
    
    print("\n" + "=" * 70)
    print("STEP 2: PREDICTING WITH SAVED PARAMETERS")
    print("=" * 70)
    print("\nCommand equivalent:")
    print(f"  python concentration_EXP/predict.py --roi-csv <new_data.csv>")
    print()
    
    print("Tip: Use saved parameters to predict on new data:")
    print("  python concentration_EXP/predict.py \\")
    print("    --params exp_fit_output/exp_fit_params.csv \\")
    print("    --roi-csv pic_RGB/roi_results/new_samples.csv \\")
    print("    --channel G_corr")
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"\nParameters: {cfg.params_path}")
    print(f"Fit quality: {cfg.predictions_path}")
    print(f"Concentrations: <input_csv>_pred_concentration_exp.csv")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Run fitting on your training data:
   python concentration_EXP/train.py

2. Inspect the fit quality:
   - Check R2 values in exp_fit_params.csv
   - Visualize exp_fit_predictions.csv

3. Choose the best channel (highest R2)

4. Predict on new data:
   python concentration_EXP/predict.py --roi-csv <new_data> --channel <best_channel>

For more options, see:
   python concentration_EXP/train.py --help
   python concentration_EXP/predict.py --help
    """)
    
    print("=" * 70)
    print("For detailed documentation, see: concentration_EXP/README.md")
    print("=" * 70)


if __name__ == "__main__":
    example_workflow()

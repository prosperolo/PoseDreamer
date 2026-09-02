#!/usr/bin/env python3
"""
Complete Pose Complexity Evaluation Pipeline

This script runs the complete pipeline:
1. Data mining from h5 files
2. Model training with LightGBM
3. Test set evaluation and sample analysis

Usage:
    python pipeline.py --h5_folder_path /path/to/h5/files --save_folder ./results
    python pipeline.py --h5_folder_path /path/to/h5/files --save_folder ./results --num_samples 500
"""

import fire
import numpy as np
from pathlib import Path

from posedreamer.hard_mining.data_mining import PoseComplexityDataMiner
from posedreamer.hard_mining.train_model import PoseComplexityModel
from posedreamer.hard_mining.inference import PoseComplexityPredictor, SMPLXParams


def run_pipeline(h5_folder_path: str, save_folder: str, num_samples: int = None, debug_visuals: bool = False):
    """
    Run the complete pose complexity evaluation pipeline.
    
    Args:
        h5_folder_path: Path to directory containing h5 files
        save_folder: Directory to save all outputs (dataset, model, results)
        num_samples: Maximum number of h5 files to process (optional, processes all if None)
        debug_visuals: Whether to save debug visualizations comparing h5 vs YOLO joints (TEMP)
    """
    print(f"Starting Pose Complexity Evaluation Pipeline")
    print(f"H5 Files: {h5_folder_path}")
    print(f"Output: {save_folder}")
    print(f"Max Samples: {num_samples if num_samples else 'All'}")
    
    # Create output directory
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Mining
    print("\n" + "="*60)
    print("STEP 1: DATA MINING")
    print("="*60)
    
    dataset_path = save_path / "pose_dataset"
    
    print(f"Mining pose complexity data from: {h5_folder_path}")
    if num_samples:
        print(f"Processing up to {num_samples} files...")
    
    miner = PoseComplexityDataMiner(debug_visuals=debug_visuals)
    miner.mine_dataset(
        data_path=h5_folder_path,
        output_path=str(dataset_path), 
        max_files=num_samples
    )
    
    # Check data quality
    targets = np.load(str(dataset_path) + "_targets.npy")
    valid_samples = (targets >= 0).sum()
    
    if valid_samples < 10:
        print(f"ERROR: Only {valid_samples} valid samples found. Need at least 10 for training.")
        print("Consider increasing num_samples or checking data quality.")
        return
    
    print(f"Data mining completed: {valid_samples} valid samples")
    
    # Step 2: Model Training
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    print("Training LightGBM model...")
    
    # Load training data
    X = np.load(str(dataset_path) + "_features.npy")
    y = np.load(str(dataset_path) + "_targets.npy")
    
    # Train model
    model = PoseComplexityModel()
    results = model.train(X, y, test_size=0.2, validation_size=0.1)
    
    # Save model
    model_path = save_path / "pose_complexity_model.pkl"
    model.save_model(str(model_path), {
        "training_results": results,
        "num_samples": valid_samples
    })
    
    # Print training results
    print("\nTraining Results:")
    for dataset, metrics in results.items():
        if dataset == "cv_rmse":
            print(f"  {dataset}: {metrics['mean']:.3f} ± {metrics['std']:.3f}")
        else:
            print(f"  {dataset}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.3f}")
    
    print(f"Model saved to: {model_path}")
    
    # Step 3: Test Set Evaluation
    print("\n" + "="*60)
    print("STEP 3: TEST SET EVALUATION")
    print("="*60)
    
    from posedreamer.hard_mining.evaluator import evaluate_model
    
    # Get test indices from training (if available in metadata)
    model_metadata = model.metadata if hasattr(model, 'metadata') else {}
    test_indices = None  # Use all valid samples for evaluation
    
    # Evaluate model on test set and save best/worst samples
    eval_df = evaluate_model(
        model_path=str(model_path),
        dataset_path=str(dataset_path),
        save_dir=str(save_path),
        test_indices=test_indices
    )
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Dataset: {valid_samples} valid samples")
    print(f"Model: LightGBM (Test R²: {results['test']['r2']:.3f})")
    print(f"Output: {save_path}")
    print("\nFiles created:")
    print(f"  Dataset: {dataset_path}*")
    print(f"  Model: {model_path}")
    print(f"  Evaluation: {save_path}/eval/")
    
    print(f"\nPipeline completed successfully!")


if __name__ == "__main__":
    fire.Fire(run_pipeline) 

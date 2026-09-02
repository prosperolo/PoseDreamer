import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import lightgbm as lgb


class PoseComplexityModel:
    """LightGBM model for predicting pose complexity from SMPL-X parameters."""
    
    def __init__(self):
        """Initialize the pose complexity model with optimal hyperparameters."""
        self.model_params = {
            "objective": "regression",
            "metric": "rmse", 
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "max_depth": 8,
            "n_estimators": 1000,
            "random_state": 42,
            "verbose": -1
        }
        self.model = None
        self.feature_names = None
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter valid samples and clean features."""
        # Filter out invalid samples (OKS < 0)
        valid_mask = y >= 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        print(f"Filtered dataset: {len(X_valid)}/{len(X)} samples kept")
        
        # Check for invalid features (NaN, inf)
        feature_valid_mask = np.isfinite(X_valid).all(axis=1)
        X_clean = X_valid[feature_valid_mask]
        y_clean = y_valid[feature_valid_mask]
        
        print(f"Cleaned dataset: {len(X_clean)}/{len(X_valid)} samples kept")
        
        return X_clean, y_clean
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, validation_size: float = 0.1) -> Dict[str, Any]:
        """Train the pose complexity model."""
        # Preprocess data
        X, y = self.preprocess_data(X, y)
        
        if len(X) < 10:
            raise ValueError("Insufficient training data after preprocessing")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=42
        )
        
        print(f"Dataset splits:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Create and train model
        self.model = lgb.LGBMRegressor(**self.model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Evaluate model
        results = self._evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val)
        
        return results
    
    def _evaluate_model(self, X_train, y_train, X_test, y_test, X_val, y_val) -> Dict[str, Any]:
        """Evaluate the trained model and return metrics."""
        results = {}
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        results["train"] = {
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "mae": mean_absolute_error(y_train, y_train_pred),
            "r2": r2_score(y_train, y_train_pred)
        }
        
        # Validation metrics
        y_val_pred = self.model.predict(X_val)
        results["validation"] = {
            "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "mae": mean_absolute_error(y_val, y_val_pred),
            "r2": r2_score(y_val, y_val_pred)
        }
        
        # Test metrics
        y_test_pred = self.model.predict(X_test)
        results["test"] = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "mae": mean_absolute_error(y_test, y_test_pred),
            "r2": r2_score(y_test, y_test_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                   cv=5, scoring="neg_mean_squared_error")
        results["cv_rmse"] = {
            "mean": np.sqrt(-cv_scores.mean()),
            "std": np.sqrt(cv_scores.std())
        }
        
        return results
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        return df
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict OKS scores for given SMPL-X parameters."""
        return self.model.predict(X)
    
    def save_model(self, model_path: str, metadata: Dict = None):
        """Save the trained model and metadata."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "model_type": "lightgbm",
                "model_params": self.model_params,
                "feature_names": self.feature_names,
                "metadata": metadata or {}
            }, f)
        
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.model_params = data["model_params"]
        self.feature_names = data.get("feature_names")
        
        return data.get("metadata", {})


def plot_training_results(results: Dict, save_path: str = None):
    """Plot training results and metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Metrics comparison
    datasets = ["train", "validation", "test"]
    
    # RMSE
    rmse_values = [results[ds]["rmse"] for ds in datasets]
    bars1 = ax1.bar(datasets, rmse_values)
    ax1.set_title("RMSE by Dataset")
    ax1.set_ylabel("RMSE")
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{value:.3f}", ha="center", va="bottom")
    
    # MAE
    mae_values = [results[ds]["mae"] for ds in datasets]
    bars2 = ax2.bar(datasets, mae_values, color='orange')
    ax2.set_title("MAE by Dataset")
    ax2.set_ylabel("MAE")
    for bar, value in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{value:.3f}", ha="center", va="bottom")
    
    # R²
    r2_values = [results[ds]["r2"] for ds in datasets]
    bars3 = ax3.bar(datasets, r2_values, color='green')
    ax3.set_title("R² by Dataset")
    ax3.set_ylabel("R²")
    ax3.set_ylim(0, 1)
    for bar, value in zip(bars3, r2_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{value:.3f}", ha="center", va="bottom")
    
    # Cross-validation
    cv_mean = results["cv_rmse"]["mean"]
    cv_std = results["cv_rmse"]["std"]
    ax4.bar(["CV RMSE"], [cv_mean], yerr=[cv_std], capsize=10, color='red')
    ax4.set_title("Cross-Validation RMSE")
    ax4.set_ylabel("RMSE")
    ax4.text(0, cv_mean, f"{cv_mean:.3f}±{cv_std:.3f}", 
            ha="center", va="bottom")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training results plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train pose complexity prediction model")
    parser.add_argument("dataset_path", help="Path to the mined dataset (without extension)")
    parser.add_argument("--test_size", type=float, default=0.2, 
                       help="Fraction of data for testing")
    parser.add_argument("--validation_size", type=float, default=0.1,
                       help="Fraction of training data for validation")
    parser.add_argument("--output_dir", default="./trained_models",
                       help="Directory to save trained model")
    parser.add_argument("--plot_results", action="store_true",
                       help="Plot and save training results")
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    features_path = Path(args.dataset_path + "_features.npy")
    targets_path = Path(args.dataset_path + "_targets.npy")
    metadata_path = Path(args.dataset_path + "_metadata.json")
    
    if not all(p.exists() for p in [features_path, targets_path]):
        raise FileNotFoundError("Dataset files not found. Run data mining first.")
    
    X = np.load(features_path)
    y = np.load(targets_path)
    
    with open(metadata_path, "r") as f:
        dataset_metadata = json.load(f)
    
    feature_names = dataset_metadata.get("feature_names", [])
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Valid samples: {(y >= 0).sum()}/{len(y)}")
    
    # Create and train model
    print("\nTraining LightGBM model...")
    model = PoseComplexityModel()
    model.feature_names = feature_names
    
    results = model.train(X, y, args.test_size, args.validation_size)
    
    # Print results
    print("\n=== TRAINING RESULTS ===")
    for dataset, metrics in results.items():
        if dataset == "cv_rmse":
            print(f"{dataset}: {metrics['mean']:.3f} ± {metrics['std']:.3f}")
        else:
            print(f"{dataset}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
    
    # Feature importance
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    feature_importance = model.get_feature_importance(feature_names)
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "pose_complexity_lightgbm.pkl"
    model.save_model(str(model_path), {
        "training_results": results,
        "dataset_metadata": dataset_metadata,
        "feature_importance": feature_importance.to_dict("records")
    })
    
    # Save feature importance
    importance_path = output_dir / "feature_importance_lightgbm.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")
    
    # Plot results
    if args.plot_results:
        plot_path = output_dir / "training_results_lightgbm.png"
        plot_training_results(results, str(plot_path))
    
    print(f"\nModel training completed successfully!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main() 
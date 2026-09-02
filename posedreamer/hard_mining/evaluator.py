import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import cv2

from posedreamer.hard_mining.inference import PoseComplexityPredictor


class PoseComplexityEvaluator:
    """Evaluates pose complexity model on test set and saves best/worst samples."""
    
    def __init__(self, model_path: str, dataset_path: str):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset (without extension)
        """
        self.predictor = PoseComplexityPredictor(model_path)
        self.dataset_path = dataset_path
        
        # Load dataset
        self.features = np.load(dataset_path + "_features.npy")
        self.targets = np.load(dataset_path + "_targets.npy")
        
        # Load CSV for metadata access
        csv_path = Path(dataset_path).with_suffix(".csv")
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
        
        print(f"Loaded dataset: {len(self.features)} samples")
        print(f"Valid samples: {(self.targets >= 0).sum()}")
    
    def evaluate_test_set(self, save_dir: str, test_indices: np.ndarray = None):
        """
        Evaluate model on test set and save top/bottom 10% samples.
        
        Args:
            save_dir: Directory to save evaluation results
            test_indices: Indices of test samples (if None, uses all valid samples)
        """
        eval_dir = Path(save_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter valid samples
        valid_mask = self.targets >= 0
        
        if test_indices is not None:
            # Use provided test indices
            mask = np.zeros(len(self.targets), dtype=bool)
            mask[test_indices] = True
            test_mask = valid_mask & mask
        else:
            # Use all valid samples
            test_mask = valid_mask
        
        test_features = self.features[test_mask]
        test_targets = self.targets[test_mask]
        test_df = self.df[test_mask].reset_index(drop=True)
        
        print(f"\nEvaluating on {len(test_features)} test samples...")
        
        # Predict scores
        predicted_scores = self.predictor.model.predict(test_features)
        
        # Create evaluation DataFrame
        eval_df = pd.DataFrame({
            'actual_oks': test_targets,
            'predicted_oks': predicted_scores,
            'h5_path': test_df['h5_path'],
            'image_path': test_df.get('image_path', ''),
            'densepose_path': test_df.get('densepose_path', '')
        })
        
        # Sort by predicted scores
        eval_df = eval_df.sort_values('predicted_oks', ascending=False).reset_index(drop=True)
        
        # Calculate percentiles
        n_samples = len(eval_df)
        top_10_count = max(1, n_samples // 10)
        bottom_10_count = max(1, n_samples // 10)
        
        print(f"Saving top {top_10_count} and bottom {bottom_10_count} samples...")
        
        # Get top and bottom 10%
        top_samples = eval_df.head(top_10_count)
        bottom_samples = eval_df.tail(bottom_10_count)
        
        # Save evaluation results
        self._save_samples(top_samples, eval_dir / "top_10_percent", "best")
        self._save_samples(bottom_samples, eval_dir / "bottom_10_percent", "worst")
        
        # Save full evaluation results
        eval_df.to_csv(eval_dir / "full_evaluation.csv", index=False)
        
        # Generate summary statistics
        self._generate_summary(eval_df, top_samples, bottom_samples, eval_dir)
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {eval_dir}")
        
        return eval_df
    
    def _save_samples(self, samples_df: pd.DataFrame, save_dir: Path, category: str):
        """Save sample files (h5, images, denseposes) to directory."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        sample_info = []
        
        for idx, row in samples_df.iterrows():
            sample_id = f"sample_{idx:03d}"
            
            # Copy h5 file
            h5_src = row['h5_path']
            if os.path.exists(h5_src):
                h5_dst = save_dir / f"{sample_id}.h5"
                shutil.copy2(h5_src, h5_dst)
            
            # Copy image file
            img_src = row['image_path']
            if pd.notna(img_src) and os.path.exists(img_src):
                img_ext = Path(img_src).suffix
                img_dst = save_dir / f"{sample_id}_image{img_ext}"
                shutil.copy2(img_src, img_dst)
            
            # Copy densepose file
            dp_src = row['densepose_path']
            if pd.notna(dp_src) and os.path.exists(dp_src):
                dp_ext = Path(dp_src).suffix
                dp_dst = save_dir / f"{sample_id}_densepose{dp_ext}"
                shutil.copy2(dp_src, dp_dst)
            
            # Store sample info
            sample_info.append({
                'sample_id': sample_id,
                'actual_oks': row['actual_oks'],
                'predicted_oks': row['predicted_oks'],
                'original_h5': h5_src,
                'original_image': img_src,
                'original_densepose': dp_src
            })
        
        # Save sample metadata
        with open(save_dir / "sample_info.json", 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        print(f"Saved {len(samples_df)} {category} samples to: {save_dir}")
    
    def _generate_summary(self, eval_df: pd.DataFrame, top_samples: pd.DataFrame, 
                         bottom_samples: pd.DataFrame, eval_dir: Path):
        """Generate evaluation summary statistics."""
        
        summary = {
            "evaluation_metrics": {
                "total_samples": len(eval_df),
                "r2": float(np.corrcoef(eval_df['actual_oks'], eval_df['predicted_oks'])[0, 1] ** 2)
            },
            "score_distribution": {
                "actual_oks": {
                    "mean": float(eval_df['actual_oks'].mean()),
                    "std": float(eval_df['actual_oks'].std()),
                    "min": float(eval_df['actual_oks'].min()),
                    "max": float(eval_df['actual_oks'].max())
                },
                "predicted_oks": {
                    "mean": float(eval_df['predicted_oks'].mean()),
                    "std": float(eval_df['predicted_oks'].std()),
                    "min": float(eval_df['predicted_oks'].min()),
                    "max": float(eval_df['predicted_oks'].max())
                }
            },
            "top_10_percent": {
                "count": len(top_samples),
                "avg_predicted_oks": float(top_samples['predicted_oks'].mean()),
                "avg_actual_oks": float(top_samples['actual_oks'].mean())
            },
            "bottom_10_percent": {
                "count": len(bottom_samples),
                "avg_predicted_oks": float(bottom_samples['predicted_oks'].mean()),
                "avg_actual_oks": float(bottom_samples['actual_oks'].mean())
            }
        }
        
        # Save summary
        with open(eval_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Samples: {summary['evaluation_metrics']['total_samples']}")
        print(f"R²: {summary['evaluation_metrics']['r2']:.3f}")
        
        print(f"\nTop 10% - Highest Predicted Complexity ({summary['top_10_percent']['count']} samples):")
        print(f"  Avg Predicted OKS: {summary['top_10_percent']['avg_predicted_oks']:.3f}")
        print(f"  Avg Actual OKS: {summary['top_10_percent']['avg_actual_oks']:.3f}")
        
        print(f"\nBottom 10% - Lowest Predicted Complexity ({summary['bottom_10_percent']['count']} samples):")
        print(f"  Avg Predicted OKS: {summary['bottom_10_percent']['avg_predicted_oks']:.3f}")
        print(f"  Avg Actual OKS: {summary['bottom_10_percent']['avg_actual_oks']:.3f}")
    
    def create_visual_comparison(self, eval_dir: Path, max_samples: int = 5):
        """Create visual comparison of top/bottom samples."""
        import matplotlib.pyplot as plt
        
        top_dir = eval_dir / "top_10_percent"
        bottom_dir = eval_dir / "bottom_10_percent"
        
        # Load sample info
        with open(top_dir / "sample_info.json", 'r') as f:
            top_info = json.load(f)[:max_samples]
        
        with open(bottom_dir / "sample_info.json", 'r') as f:
            bottom_info = json.load(f)[:max_samples]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, max_samples, figsize=(4*max_samples, 8))
        if max_samples == 1:
            axes = axes.reshape(2, 1)
        
        # Plot top samples
        for i, sample in enumerate(top_info):
            img_path = top_dir / f"{sample['sample_id']}_image.jpg"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(img_rgb)
                axes[0, i].set_title(f"Best #{i+1}\nPred: {sample['predicted_oks']:.3f}\nActual: {sample['actual_oks']:.3f}")
                axes[0, i].axis('off')
        
        # Plot bottom samples  
        for i, sample in enumerate(bottom_info):
            img_path = bottom_dir / f"{sample['sample_id']}_image.jpg"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[1, i].imshow(img_rgb)
                axes[1, i].set_title(f"Worst #{i+1}\nPred: {sample['predicted_oks']:.3f}\nActual: {sample['actual_oks']:.3f}")
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(eval_dir / "visual_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visual comparison saved to: {eval_dir / 'visual_comparison.png'}")


def evaluate_model(model_path: str, dataset_path: str, save_dir: str, test_indices: np.ndarray = None):
    """
    Convenience function to evaluate model and save results.
    
    Args:
        model_path: Path to trained model
        dataset_path: Path to dataset (without extension)
        save_dir: Directory to save evaluation results
        test_indices: Optional test set indices
    """
    evaluator = PoseComplexityEvaluator(model_path, dataset_path)
    eval_df = evaluator.evaluate_test_set(save_dir, test_indices)
    
    # Create visual comparison
    eval_dir = Path(save_dir) / "eval"
    evaluator.create_visual_comparison(eval_dir)
    
    return eval_df 
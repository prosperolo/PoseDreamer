import os
import json
import cv2
import numpy as np
import pandas as pd
import deepdish as dd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

from posedreamer.filtering.functional.oks import (
    KeypointLoader, OKSMetric, JOINTS_MAPPING, compute_oks
)


@dataclass
class SMPLXFeatures:
    """Container for SMPL-X parameter features."""
    body_pose: np.ndarray  # Body pose parameters (flattened)
    global_orient: np.ndarray  # Global orientation (3,)
    betas: np.ndarray  # Shape parameters (10,)
    transl: np.ndarray  # Translation (3,)
    left_hand_pose: np.ndarray  # Left hand pose (optional)
    right_hand_pose: np.ndarray  # Right hand pose (optional)
    jaw_pose: np.ndarray  # Jaw pose (optional)
    expression: np.ndarray  # Expression parameters (optional)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert SMPL-X parameters to a single feature vector."""
        features = []
        
        # Core body parameters
        features.append(self.body_pose.flatten())
        features.append(self.global_orient.flatten())
        features.append(self.betas.flatten())
        features.append(self.transl.flatten())
        
        # Optional parameters (set to zeros if missing)
        if self.left_hand_pose is not None:
            features.append(self.left_hand_pose.flatten())
        else:
            features.append(np.zeros(45))  # Standard hand pose size
            
        if self.right_hand_pose is not None:
            features.append(self.right_hand_pose.flatten())
        else:
            features.append(np.zeros(45))
            
        if self.jaw_pose is not None:
            features.append(self.jaw_pose.flatten())
        else:
            features.append(np.zeros(3))
            
        if self.expression is not None:
            features.append(self.expression.flatten())
        else:
            features.append(np.zeros(10))  # Standard expression size
        
        return np.concatenate(features)


class PoseComplexityDataMiner:
    """Mines pose complexity data from h5 files using OKS metrics."""
    
    def __init__(self, keypoint_model: str = "yolov8x-pose.pt", debug_visuals: bool = False):
        """
        Initialize the data miner.
        
        Args:
            keypoint_model: YOLO model for keypoint detection
            debug_visuals: Whether to save debug visualizations (TEMP - can be removed)
        """
        self.keypoint_loader = KeypointLoader(keypoint_model)
        self.oks_metric = OKSMetric()
        self.debug_visuals = debug_visuals
        self.debug_count = 0  # Counter for debug images
        
    def load_h5_data(self, h5_path: str) -> Optional[Dict]:
        """Load data from h5 file."""
        return dd.io.load(h5_path)
    
    def extract_smplx_features(self, h5_data: Dict) -> SMPLXFeatures:
        """Extract SMPL-X features from h5 data."""
        # Handle potential batch dimension and squeeze if needed
        def safe_squeeze(arr):
            if arr is None:
                return None
            arr = np.array(arr)
            return arr.squeeze() if arr.ndim > 1 and arr.shape[0] == 1 else arr
        
        return SMPLXFeatures(
            body_pose=safe_squeeze(h5_data.get("body_pose")),
            global_orient=safe_squeeze(h5_data.get("global_orient")),
            betas=safe_squeeze(h5_data.get("betas")),
            transl=safe_squeeze(h5_data.get("transl")),
            left_hand_pose=safe_squeeze(h5_data.get("left_hand_pose")),
            right_hand_pose=safe_squeeze(h5_data.get("right_hand_pose")),
            jaw_pose=safe_squeeze(h5_data.get("jaw_pose")),
            expression=safe_squeeze(h5_data.get("expression"))
        )
    
    def _save_debug_visual(self, image_rgb: np.ndarray, h5_joints: np.ndarray, 
                          yolo_joints: np.ndarray, oks_score: float, 
                          output_dir: str, filename: str):
        """
        TEMP DEBUG: Save visualization comparing h5 joints vs YOLO joints.
        This function can be easily removed later.
        """
        if not self.debug_visuals:
            return
            
        debug_dir = Path(output_dir) / "debug_visuals"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Image with h5 joints (ground truth)
        ax2.imshow(image_rgb)
        for i, (x, y) in enumerate(h5_joints):
            ax2.plot(x, y, 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
            ax2.text(x + 5, y - 5, str(i), color='white', fontsize=8, fontweight='bold')
        ax2.set_title('H5 Joints (Ground Truth)')
        ax2.axis('off')
        
        # Image with YOLO joints
        ax3.imshow(image_rgb)
        for i, (x, y, conf) in enumerate(yolo_joints):
            if conf > 0.3:  # Only show confident detections
                ax3.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
                ax3.text(x + 5, y - 5, str(i), color='white', fontsize=8, fontweight='bold')
        ax3.set_title('YOLO Joints (Detected)')
        ax3.axis('off')
        
        plt.suptitle(f'OKS Score: {oks_score:.3f} | File: {filename}', fontsize=14)
        plt.tight_layout()
        
        save_path = debug_dir / f"debug_{self.debug_count:04d}_{Path(filename).stem}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.debug_count += 1
    
    def compute_oks_score(self, h5_data: Dict, output_dir: str = None) -> Tuple[float, Dict]:
        """Compute OKS score for h5 data."""
        # Get image and densepose paths
        image_path = h5_data['generated-image']
        densepose_path = h5_data["densepose-render"]
        
        generated_image = cv2.imread(image_path)
        condition_image = cv2.imread(densepose_path)
        
        generated_image = cv2.resize(generated_image, condition_image.shape[:2])
        image_rgb = generated_image[..., ::-1]
        
        joints_2d = h5_data.get("joints_2d")
        smpl_joints = joints_2d[JOINTS_MAPPING]
        
        # Get YOLO keypoints for debugging
        yolo_result = self.keypoint_loader.predict_keypoints(
            image_rgb, smpl_joints, max_detections=4
        )
        
        # Compute OKS score
        oks_score = self.oks_metric.compute_oks(
            yolo_result.keypoints,  # YOLO as GT
            smpl_joints,            # SMPL as predictions  
            yolo_result.bbox_area
        )
        
        # TEMP DEBUG: Save visual comparison
        if self.debug_visuals and output_dir:
            self._save_debug_visual(
                image_rgb, smpl_joints, yolo_result.keypoints, 
                oks_score, output_dir, Path(image_path).name
            )
        
        metadata = {
            "is_crowded": not yolo_result.valid_sample,
            "num_detections": yolo_result.num_detections,
            "image_path": image_path,
            "densepose_path": densepose_path
        }
        
        return oks_score, metadata

    def process_single_file(self, h5_path: str, output_dir: str = None) -> Optional[Dict]:
        """Process a single h5 file and extract features + OKS score."""
        # Load h5 data
        h5_data = self.load_h5_data(h5_path)
        if h5_data is None:
            return None
        
        # Extract SMPL-X features
        smplx_features = self.extract_smplx_features(h5_data)
        if smplx_features is None:
            return None
        
        # Compute OKS score (with optional debug visuals)
        oks_score, metadata = self.compute_oks_score(h5_data, output_dir)
        
        # Convert to feature vector
        feature_vector = smplx_features.to_feature_vector()
        
        return {
            "h5_path": h5_path,
            "features": feature_vector,
            "oks_score": oks_score,
            "metadata": metadata
        }
    
    def mine_dataset(self, data_path: str, output_path: str, 
                    max_files: Optional[int] = None) -> None:
        """
        Mine pose complexity dataset from h5 files.
        
        Args:
            data_path: Path to directory containing h5 files
            output_path: Path to save the mined dataset
            max_files: Maximum number of files to process (None for all)
        """
        # Find all h5 files
        h5_files = list(Path(data_path).glob("*.h5"))
        if max_files:
            h5_files = sorted(h5_files)[:max_files]
        
        print(f"Found {len(h5_files)} h5 files to process")
        if self.debug_visuals:
            print(f"Debug visuals enabled - saving to {Path(output_path).parent}/debug_visuals/")
        
        # Process files
        results = []
        valid_samples = 0
        output_dir = str(Path(output_path).parent)
        
        for h5_path in tqdm(h5_files, desc="Processing h5 files"):
            result = self.process_single_file(str(h5_path), output_dir)
            if result is not None:
                results.append(result)
                if result["oks_score"] >= 0:
                    valid_samples += 1
        
        if not results:
            print("No valid results found!")
            return
        
        # Create dataset
        print(f"Processed {len(results)} files, {valid_samples} with valid OKS scores")
        
        # Prepare data for saving
        features_matrix = np.stack([r["features"] for r in results])
        oks_scores = np.array([r["oks_score"] for r in results])
        
        # Create DataFrame for analysis
        feature_names = self._get_feature_names()
        df = pd.DataFrame(features_matrix, columns=feature_names)
        df["oks_score"] = oks_scores
        df["h5_path"] = [r["h5_path"] for r in results]
        
        # Add metadata columns
        for result in results:
            metadata = result["metadata"]
            for key, value in metadata.items():
                if key not in df.columns:
                    df[key] = None
        
        for i, result in enumerate(results):
            metadata = result["metadata"]
            for key, value in metadata.items():
                df.loc[i, key] = value
        
        # Save dataset
        output_dir_path = Path(output_path).parent
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_path + ".csv"
        df.to_csv(csv_path, index=False)
        
        # Save features and targets separately for training
        np.save(output_path + "_features.npy", features_matrix)
        np.save(output_path + "_targets.npy", oks_scores)
        
        # Save metadata
        metadata_path = output_path + "_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "num_samples": len(results),
                "num_valid_samples": valid_samples,
                "feature_names": feature_names,
                "num_features": len(feature_names),
                "debug_visuals_enabled": self.debug_visuals,
                "debug_images_saved": self.debug_count if self.debug_visuals else 0,
                "oks_stats": {
                    "mean": float(np.mean(oks_scores[oks_scores >= 0])) if valid_samples > 0 else -1,
                    "std": float(np.std(oks_scores[oks_scores >= 0])) if valid_samples > 0 else -1,
                    "min": float(np.min(oks_scores[oks_scores >= 0])) if valid_samples > 0 else -1,
                    "max": float(np.max(oks_scores[oks_scores >= 0])) if valid_samples > 0 else -1
                }
            }, f, indent=2)
        
        print(f"\nDataset saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  Features: {output_path}_features.npy")
        print(f"  Targets: {output_path}_targets.npy")
        print(f"  Metadata: {metadata_path}")
        
        if self.debug_visuals:
            print(f"  Debug visuals: {output_dir_path}/debug_visuals/ ({self.debug_count} images)")
        
        if valid_samples > 0:
            valid_scores = oks_scores[oks_scores >= 0]
            print(f"\nOKS Statistics (valid samples only):")
            print(f"  Mean: {np.mean(valid_scores):.3f}")
            print(f"  Std: {np.std(valid_scores):.3f}")
            print(f"  Range: [{np.min(valid_scores):.3f}, {np.max(valid_scores):.3f}]")
    
    def _get_feature_names(self) -> List[str]:
        """Generate feature names for the dataset."""
        names = []
        
        # Body pose (21 joints * 3 rotation params)
        for i in range(63):
            names.append(f"body_pose_{i}")
        
        # Global orientation (3 params)
        for i in range(3):
            names.append(f"global_orient_{i}")
        
        # Shape parameters (10 betas)
        for i in range(10):
            names.append(f"betas_{i}")
        
        # Translation (3 params)
        for i in range(3):
            names.append(f"transl_{i}")
        
        # Left hand pose (45 params)
        for i in range(45):
            names.append(f"left_hand_pose_{i}")
        
        # Right hand pose (45 params)
        for i in range(45):
            names.append(f"right_hand_pose_{i}")
        
        # Jaw pose (3 params)
        for i in range(3):
            names.append(f"jaw_pose_{i}")
        
        # Expression (10 params)
        for i in range(10):
            names.append(f"expression_{i}")
        
        return names


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mine pose complexity data from h5 files")
    parser.add_argument("data_path", help="Path to directory containing h5 files")
    parser.add_argument("output_path", help="Path to save the mined dataset")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    parser.add_argument("--keypoint_model", default="yolov8x-pose.pt", 
                       help="YOLO model for keypoint detection")
    parser.add_argument("--debug_visuals", action="store_true",
                       help="TEMP: Save debug visualizations comparing h5 vs YOLO joints")
    
    args = parser.parse_args()
    
    miner = PoseComplexityDataMiner(args.keypoint_model, args.debug_visuals)
    miner.mine_dataset(args.data_path, args.output_path, args.max_files)


if __name__ == "__main__":
    main()

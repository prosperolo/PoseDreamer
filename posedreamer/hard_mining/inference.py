import os
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

from posedreamer.hard_mining.train_model import PoseComplexityModel


@dataclass
class SMPLXParams:
    """Container for SMPL-X parameters for inference."""
    body_pose: np.ndarray  # (21, 3) or (63,) body pose parameters
    global_orient: np.ndarray  # (3,) global orientation
    betas: np.ndarray  # (10,) shape parameters
    transl: np.ndarray  # (3,) translation
    left_hand_pose: np.ndarray = None  # (15, 3) or (45,) left hand pose
    right_hand_pose: np.ndarray = None  # (15, 3) or (45,) right hand pose
    jaw_pose: np.ndarray = None  # (3,) jaw pose
    expression: np.ndarray = None  # (10,) expression parameters
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert SMPL-X parameters to feature vector matching training format."""
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
            features.append(np.zeros(45))
            
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
            features.append(np.zeros(10))
        
        return np.concatenate(features)
    
    @classmethod
    def from_dict(cls, params_dict: Dict) -> "SMPLXParams":
        """Create SMPLXParams from dictionary (e.g., loaded from h5 file)."""
        def safe_squeeze(arr):
            if arr is None:
                return None
            arr = np.array(arr)
            return arr.squeeze() if arr.ndim > 1 and arr.shape[0] == 1 else arr
        
        return cls(
            body_pose=safe_squeeze(params_dict.get("body_pose")),
            global_orient=safe_squeeze(params_dict.get("global_orient")),
            betas=safe_squeeze(params_dict.get("betas")),
            transl=safe_squeeze(params_dict.get("transl")),
            left_hand_pose=safe_squeeze(params_dict.get("left_hand_pose")),
            right_hand_pose=safe_squeeze(params_dict.get("right_hand_pose")),
            jaw_pose=safe_squeeze(params_dict.get("jaw_pose")),
            expression=safe_squeeze(params_dict.get("expression"))
        )
    
    @classmethod
    def from_random(cls, seed: int = None) -> "SMPLXParams":
        """Generate random SMPL-X parameters for testing."""
        if seed is not None:
            np.random.seed(seed)
        
        return cls(
            body_pose=np.random.randn(63) * 0.1,  # Small random rotations
            global_orient=np.random.randn(3) * 0.1,
            betas=np.random.randn(10) * 0.5,  # Shape variations
            transl=np.random.randn(3) * 0.1,
            left_hand_pose=np.random.randn(45) * 0.05,
            right_hand_pose=np.random.randn(45) * 0.05,
            jaw_pose=np.random.randn(3) * 0.02,
            expression=np.random.randn(10) * 0.1
        )


class PoseComplexityPredictor:
    """Inference class for pose complexity prediction using LightGBM."""
    
    def __init__(self, model_path: str):
        """Initialize predictor with trained LightGBM model."""
        self.model_path = model_path
        self.model = PoseComplexityModel()
        self.metadata = self.model.load_model(model_path)
        
        print(f"Loaded LightGBM model from: {model_path}")
        if "training_results" in self.metadata:
            test_rmse = self.metadata["training_results"]["test"]["rmse"]
            test_r2 = self.metadata["training_results"]["test"]["r2"]
            print(f"Model performance - Test RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
    
    def predict_single(self, smplx_params: SMPLXParams) -> float:
        """Predict OKS score for a single set of SMPL-X parameters."""
        feature_vector = smplx_params.to_feature_vector()
        prediction = self.model.predict(feature_vector.reshape(1, -1))
        return float(prediction[0])
    
    def predict_batch(self, smplx_params_list: List[SMPLXParams]) -> np.ndarray:
        """Predict OKS scores for a batch of SMPL-X parameters."""
        feature_matrix = np.stack([params.to_feature_vector() for params in smplx_params_list])
        predictions = self.model.predict(feature_matrix)
        return predictions
    
    def predict_from_dict(self, params_dict: Dict) -> float:
        """Predict OKS score from parameter dictionary."""
        smplx_params = SMPLXParams.from_dict(params_dict)
        return self.predict_single(smplx_params)
    
    def predict_from_h5(self, h5_path: str) -> float:
        """Predict OKS score from h5 file."""
        import deepdish as dd
        h5_data = dd.io.load(h5_path)
        return self.predict_from_dict(h5_data)
    
    def analyze_complexity(self, smplx_params: SMPLXParams, 
                          detailed: bool = False) -> Dict[str, Any]:
        """Analyze pose complexity with interpretable results."""
        oks_score = self.predict_single(smplx_params)
        
        # Interpret complexity level
        if oks_score >= 0.8:
            complexity_level = "Low"
            description = "Simple, well-defined pose"
        elif oks_score >= 0.6:
            complexity_level = "Medium"
            description = "Moderately complex pose"
        elif oks_score >= 0.4:
            complexity_level = "High"
            description = "Complex pose with potential ambiguities"
        else:
            complexity_level = "Very High"
            description = "Very complex or unclear pose"
        
        results = {
            "oks_score": oks_score,
            "complexity_level": complexity_level,
            "description": description
        }
        
        if detailed:
            # Analyze parameter magnitudes
            body_pose_complexity = np.linalg.norm(smplx_params.body_pose)
            shape_complexity = np.linalg.norm(smplx_params.betas)
            
            results.update({
                "body_pose_magnitude": float(body_pose_complexity),
                "shape_magnitude": float(shape_complexity),
                "has_hand_poses": (smplx_params.left_hand_pose is not None and 
                                 smplx_params.right_hand_pose is not None),
                "has_facial_expressions": smplx_params.expression is not None
            })
        
        return results
    
    def get_feature_importance(self, top_k: int = 10) -> List[Dict]:
        """Get top-k most important features from the model."""
        if "feature_importance" not in self.metadata:
            print("Feature importance not available in model metadata")
            return []
        
        importance_data = self.metadata["feature_importance"]
        return importance_data[:top_k]


def demo_inference():
    """Demonstrate inference with random SMPL-X parameters."""
    print("=== POSE COMPLEXITY INFERENCE DEMO ===\n")
    
    # Generate demo poses
    poses = [
        SMPLXParams.from_random(seed=42),  # Random pose 1
        SMPLXParams.from_random(seed=123), # Random pose 2
        SMPLXParams.from_random(seed=456), # Random pose 3
    ]
    
    # Simple pose (neutral)
    simple_pose = SMPLXParams(
        body_pose=np.zeros(63),
        global_orient=np.zeros(3),
        betas=np.zeros(10),
        transl=np.zeros(3)
    )
    poses.append(simple_pose)
    
    # Complex pose
    complex_pose = SMPLXParams(
        body_pose=np.random.randn(63) * 0.5,
        global_orient=np.random.randn(3) * 0.3,
        betas=np.random.randn(10) * 1.0,
        transl=np.random.randn(3) * 0.2,
        left_hand_pose=np.random.randn(45) * 0.2,
        right_hand_pose=np.random.randn(45) * 0.2
    )
    poses.append(complex_pose)
    
    pose_names = ["Random Pose 1", "Random Pose 2", "Random Pose 3", "Simple Pose", "Complex Pose"]
    
    return poses, pose_names


def main():
    parser = argparse.ArgumentParser(description="Predict pose complexity from SMPL-X parameters")
    parser.add_argument("model_path", help="Path to trained LightGBM model file")
    parser.add_argument("--h5_file", help="Path to h5 file with SMPL-X parameters")
    parser.add_argument("--demo", action="store_true", help="Run demo with random poses")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of demo poses to analyze")
    
    args = parser.parse_args()
    
    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    predictor = PoseComplexityPredictor(args.model_path)
    
    if args.h5_file:
        # Predict from h5 file
        print(f"\nPredicting pose complexity for: {args.h5_file}")
        oks_score = predictor.predict_from_h5(args.h5_file)
        print(f"Predicted OKS Score: {oks_score:.3f}")
        
        # Load and analyze
        import deepdish as dd
        h5_data = dd.io.load(args.h5_file)
        smplx_params = SMPLXParams.from_dict(h5_data)
        analysis = predictor.analyze_complexity(smplx_params, detailed=args.detailed)
        
        print(f"Complexity Level: {analysis['complexity_level']}")
        print(f"Description: {analysis['description']}")
        
        if args.detailed:
            print(f"\nDetailed Analysis:")
            for key, value in analysis.items():
                if key not in ["oks_score", "complexity_level", "description"]:
                    print(f"  {key}: {value}")
    
    elif args.demo:
        # Run demo
        poses, pose_names = demo_inference()
        
        print("Analyzing pose complexity for demo poses...\n")
        
        for i, (pose, name) in enumerate(zip(poses[:args.batch_size], pose_names[:args.batch_size])):
            print(f"--- {name} ---")
            analysis = predictor.analyze_complexity(pose, detailed=args.detailed)
            
            print(f"OKS Score: {analysis['oks_score']:.3f}")
            print(f"Complexity: {analysis['complexity_level']} - {analysis['description']}")
            
            if args.detailed:
                print(f"Body Pose Magnitude: {analysis['body_pose_magnitude']:.3f}")
                print(f"Shape Magnitude: {analysis['shape_magnitude']:.3f}")
                print(f"Has Hand Poses: {analysis['has_hand_poses']}")
                print(f"Has Facial Expressions: {analysis['has_facial_expressions']}")
            
            print()
    
    else:
        print("Please specify either --h5_file or --demo")
        print("Use --help for more information")
    
    # Show feature importance
    print("=== TOP 10 MOST IMPORTANT FEATURES ===")
    importance = predictor.get_feature_importance(10)
    for i, feat in enumerate(importance, 1):
        print(f"{i:2d}. {feat['feature']}: {feat['importance']:.4f}")


if __name__ == "__main__":
    main() 
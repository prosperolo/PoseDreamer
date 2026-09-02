"""
3D Head pose similarity filter.

Filters samples based on head pose similarity between generated images and rendered SMPLX avatars.
"""

import os
import cv2
import numpy as np
import deepdish as dd
import pandas as pd
from typing import Dict, Any, Optional

from posedreamer.filtering.filters.base_filter import BaseFilter
from posedreamer.filtering.sample import Sample
from posedreamer.filtering.functional.head_pose import (
    SimpleSMPLXRenderer, HeadPoseDetector, compare_head_poses
)
from posedreamer.utils.paths import WEIGHTS_DIR


def _shard_csv_path(path: str) -> str:
    """If running under a SLURM array, suffix the CSV path with the task id so tasks don't clobber each other."""
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_task{task_id}{ext}"


class HeadPoseFilter(BaseFilter):
    """
    Filter samples based on 3D head pose similarity between generated and rendered images.
    
    This filter:
    1. Renders SMPLX avatar from ground truth parameters
    2. Detects 3D head poses in both generated and rendered images
    3. Finds best head match if multiple heads detected in generated image
    4. Computes Mean Absolute Error (MAE) across Roll, Pitch, Yaw
    5. Applies threshold (currently np.inf for data collection)
    """
    
    def __init__(self,
                 name: str = "head_pose_similarity",
                 max_mae_rpy: float = 9999.0,
                 smplx_model_path: str = str(WEIGHTS_DIR),
                 device: str = 'cpu',
                 stats_csv: Optional[str] = None):
        """
        Initialize head pose filter.

        Args:
            name: Filter name for identification
            smplx_model_path: Path to SMPLX model files (should contain SMPLX_NEUTRAL.npz)
            max_mae_rpy: Maximum allowed MAE across Roll, Pitch, Yaw (degrees)
            device: PyTorch device ('cpu' or 'cuda')
            stats_csv: If set, dump per-sample (mae_rpy, image_path, densepose_path) rows to this CSV.
                Under a SLURM array the filename is automatically suffixed with the task id.
        """
        self.max_mae_rpy = max_mae_rpy
        self.stats_csv = stats_csv
        super().__init__(name)
        self.smplx_model_path = smplx_model_path
        self.device = device

        self.renderer = SimpleSMPLXRenderer(smplx_model_path, device)
        self.detector = HeadPoseDetector()
        self.reset_stats()
    
    def reset_stats(self):
        """Reset statistics including head pose specific metrics."""
        super().reset_stats()
        self.stats.update({
            'mae_rpy_scores': [],
            'no_head_generated': 0,
            'no_head_rendered': 0,
            'multiple_heads_generated': 0,
            'invalid_files': 0,
            'renderer_errors': 0,
            'mean_mae_rpy': 0.0,
            'std_mae_rpy': 0.0,
            'min_mae_rpy': float('inf'),
            'max_mae_rpy_threshold': self.max_mae_rpy
        })
        if getattr(self, 'stats_csv', None) is not None:
            self.stats['densepose_paths'] = []
            self.stats['images_paths'] = []
    
    def validate(self, sample: Sample) -> bool:
        """
        Validate sample using 3D head pose similarity.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample passes head pose threshold
        """    
        smplx_data = dd.io.load(sample.smplx_path)

        # Perform head pose comparison
        comparison = compare_head_poses(
            sample.image_path, 
            sample.densepose_path,
            smplx_data, 
            self.renderer, 
            self.detector
        )
        
        # Track detection statistics
        if comparison.generated_pose is None:
            self.stats['no_head_generated'] += 1
            return False
        
        if comparison.rendered_pose is None:
            self.stats['no_head_rendered'] += 1
            return False
        
        if comparison.has_multiple_heads:
            self.stats['multiple_heads_generated'] += 1
        
            # refactor this to be more general

            angles = [comparison.generated_pose.roll, comparison.generated_pose.pitch, comparison.generated_pose.yaw]
            if any(abs(angle) > 60 for angle in angles):
                return True
        
        # Store valid MAE score
        if comparison.mae_rpy >= 0:
            self.stats['mae_rpy_scores'].append(comparison.mae_rpy)
            if self.stats_csv is not None:
                self.stats['densepose_paths'].append(sample.densepose_path)
                self.stats['images_paths'].append(sample.image_path)
            return comparison.mae_rpy <= self.max_mae_rpy
        else:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including head pose metrics."""
        stats = super().get_stats()
        
        # Compute head pose statistics
        if self.stats['mae_rpy_scores']:
            mae_scores = np.array(self.stats['mae_rpy_scores'])
            stats['mean_mae_rpy'] = float(np.mean(mae_scores))
            stats['std_mae_rpy'] = float(np.std(mae_scores))
            stats['min_mae_rpy'] = float(np.min(mae_scores))
            stats['max_mae_rpy_measured'] = float(np.max(mae_scores))
            stats['samples_below_threshold'] = int(np.sum(mae_scores <= self.max_mae_rpy))
        else:
            stats['mean_mae_rpy'] = 0.0
            stats['std_mae_rpy'] = 0.0
            stats['min_mae_rpy'] = float('inf')
            stats['max_mae_rpy_measured'] = 0.0
            stats['samples_below_threshold'] = 0
        
        # Add detection statistics
        stats['no_head_generated'] = self.stats['no_head_generated']
        stats['no_head_rendered'] = self.stats['no_head_rendered']
        stats['multiple_heads_generated'] = self.stats['multiple_heads_generated']
        stats['invalid_files'] = self.stats['invalid_files']
        stats['renderer_errors'] = self.stats['renderer_errors']
        stats['max_mae_rpy_threshold'] = self.stats['max_mae_rpy_threshold']
        
        return stats
    
    def print_detailed_stats(self):
        """Print detailed statistics for this filter."""
        stats = self.get_stats()
        
        print(f"\n=== {self.name} Detailed Statistics ===")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Passed: {stats['total_passed']}")
        print(f"Failed: {stats['total_failed']}")
        
        if stats['total_processed'] > 0:
            pass_rate = stats['total_passed'] / stats['total_processed'] * 100
            print(f"Pass rate: {pass_rate:.2f}%")
        
        print(f"\nHead Pose MAE Metrics:")
        print(f"Mean MAE (RPY): {stats['mean_mae_rpy']:.3f}° ± {stats['std_mae_rpy']:.3f}°")
        print(f"MAE Range: {stats['min_mae_rpy']:.3f}° - {stats['max_mae_rpy_measured']:.3f}°")
        print(f"Threshold: {stats['max_mae_rpy_threshold']:.1f}°")
        print(f"Samples below threshold: {stats['samples_below_threshold']}")
        
        print(f"\nRejection Reasons:")
        print(f"No head in generated image: {stats['no_head_generated']}")
        print(f"No head in rendered avatar: {stats['no_head_rendered']}")
        print(f"Invalid files: {stats['invalid_files']}")
        print(f"Renderer errors: {stats['renderer_errors']}")
        
        print(f"\nDetection Info:")
        print(f"Multiple heads in generated: {stats['multiple_heads_generated']}")
        
        if self.stats['mae_rpy_scores']:
            mae_scores = np.array(self.stats['mae_rpy_scores'])
            print(f"\nMAE Distribution:")
            print(f"25th percentile: {np.percentile(mae_scores, 25):.3f}°")
            print(f"Median: {np.median(mae_scores):.3f}°")
            print(f"75th percentile: {np.percentile(mae_scores, 75):.3f}°")
            print(f"90th percentile: {np.percentile(mae_scores, 90):.3f}°")
            print(f"95th percentile: {np.percentile(mae_scores, 95):.3f}°")

            if self.stats_csv is not None:
                csv_path = _shard_csv_path(self.stats_csv)
                pd.DataFrame({
                    'mae_rpy_scores': self.stats['mae_rpy_scores'],
                    'densepose_paths': self.stats['densepose_paths'],
                    'images_paths': self.stats['images_paths'],
                }).to_csv(csv_path, index=False)
                print(f"Wrote per-sample stats to {csv_path}")
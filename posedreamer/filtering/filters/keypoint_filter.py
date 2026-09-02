"""
Keypoint similarity filter using OKS (Object Keypoint Similarity) metric.

Filters samples based on keypoint similarity between generated images and SMPLX ground truth.
"""

import os

import cv2
import numpy as np
import deepdish as dd
import pandas as pd
from typing import Dict, Any, Optional

from posedreamer.filtering.filters.base_filter import BaseFilter
from posedreamer.filtering.sample import Sample
from posedreamer.filtering.functional.oks import compute_oks, KeypointLoader, OKSMetric, pick_joints_mapping


def _shard_csv_path(path: str) -> str:
    """If running under a SLURM array, suffix the CSV path with the task id so tasks don't clobber each other."""
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_task{task_id}{ext}"


class KeypointFilter(BaseFilter):
    """
    Filter samples based on keypoint similarity using OKS metric.
    
    This filter:
    1. Loads SMPLX ground truth keypoints
    2. Predicts keypoints on generated image using YOLO
    3. Computes OKS score between YOLO and SMPLX keypoints
    4. Rejects crowded scenes (>max_detections poses)
    5. Applies minimum OKS threshold
    """
    
    def __init__(self,
                 name: str = "keypoint_similarity",
                 min_oks: float = 0.5,
                 max_detections: int = 4,
                 model_name: str = "yolov8x-pose.pt",
                 stats_csv: Optional[str] = None):
        """
        Initialize keypoint filter.

        Args:
            name: Filter name for identification
            min_oks: Minimum OKS score threshold (0.0-1.0)
            max_detections: Maximum pose detections before marking as crowded
            model_name: YOLO model name or path
            stats_csv: If set, dump per-sample (score, image_path, densepose_path) rows to this CSV.
                Under a SLURM array the filename is automatically suffixed with the task id.
        """
        super().__init__(name)
        self.min_oks = min_oks
        self.max_detections = max_detections
        self.stats_csv = stats_csv

        # Initialize YOLO and OKS components
        self.keypoint_loader = KeypointLoader(model_name)
        self.oks_metric = OKSMetric()

        # Initialize additional statistics
        self.reset_stats()
    
    def reset_stats(self):
        """Reset statistics including OKS-specific metrics."""
        super().reset_stats()
        self.stats.update({
            'oks_scores': [],
            'crowded_empty_scenes': 0,
            'no_pose_detected': 0,
            'invalid_files': 0,
            'mean_oks': 0.0,
            'std_oks': 0.0
        })
        if getattr(self, 'stats_csv', None) is not None:
            self.stats['densepose_paths'] = []
            self.stats['images_paths'] = []
    
    def validate(self, sample: Sample) -> bool:
        """
        Validate sample using keypoint similarity.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample passes OKS threshold and is not crowded
        """
        try:
            # Load SMPL(X) data. SportsCap renders are SMPL-only (joints_2d
            # has 45 entries); other pipelines use SMPL-X (144+ entries).
            # pick_joints_mapping returns the right indices for either case.
            gt_path = sample.smplx_path or sample.smpl_path
            smplx_data = dd.io.load(gt_path)
            joints_2d = smplx_data["joints_2d"]
            smpl_joints = joints_2d[pick_joints_mapping(joints_2d)]
            
            # Load generated image
            image = cv2.imread(sample.image_path)
            if image is None:
                self.stats['invalid_files'] += 1
                return False
            else:
                image = image[..., ::-1]  # Convert BGR to RGB
            
            # Load densepose image for resizing reference
            densepose = cv2.imread(sample.densepose_path, cv2.IMREAD_GRAYSCALE)
            if densepose is not None and image.shape[:2] != densepose.shape[:2]:
                image = cv2.resize(image, (densepose.shape[1], densepose.shape[0]), 
                                 interpolation=cv2.INTER_LANCZOS4)
            
            # Compute OKS score
            oks_score, valid_sample, num_detections = compute_oks(
                image=image,
                smpl_joints=smpl_joints,
                keypoint_loader=self.keypoint_loader,
                oks_metric=self.oks_metric,
                max_detections=self.max_detections
            )
            
            # Track statistics
            if not valid_sample:
                self.stats['crowded_empty_scenes'] += 1
                return False
            
            if num_detections == 0:
                self.stats['no_pose_detected'] += 1
                return False
            
            if oks_score < 0:
                # Invalid OKS score (no valid keypoints)
                return False
            
            # Store valid OKS score
            self.stats['oks_scores'].append(oks_score)
            if self.stats_csv is not None:
                self.stats['densepose_paths'].append(sample.densepose_path)
                self.stats['images_paths'].append(sample.image_path)

            # Apply threshold
            return oks_score >= self.min_oks
            
        except Exception as e:
            print(f"Error processing sample {sample.filename}: {e}")
            self.stats['invalid_files'] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including OKS metrics."""
        stats = super().get_stats()
        
        # Compute OKS statistics
        if self.stats['oks_scores']:
            oks_scores = np.array(self.stats['oks_scores'])
            stats['mean_oks'] = float(np.mean(oks_scores))
            stats['std_oks'] = float(np.std(oks_scores))
            stats['min_oks_threshold'] = self.min_oks
            stats['samples_above_threshold'] = int(np.sum(oks_scores >= self.min_oks))
        else:
            stats['mean_oks'] = 0.0
            stats['std_oks'] = 0.0
            stats['min_oks_threshold'] = self.min_oks
            stats['samples_above_threshold'] = 0
        
        # Add crowding and detection statistics
        stats['crowded_empty_scenes'] = self.stats['crowded_empty_scenes']
        stats['no_pose_detected'] = self.stats['no_pose_detected']
        stats['invalid_files'] = self.stats['invalid_files']
        stats['max_detections_threshold'] = self.max_detections
        
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
        
        print(f"\nOKS Metrics:")
        print(f"Mean OKS: {stats['mean_oks']:.3f} ± {stats['std_oks']:.3f}")
        print(f"Threshold: {stats['min_oks_threshold']:.3f}")
        print(f"Samples above threshold: {stats['samples_above_threshold']}")
        
        print(f"\nRejection Reasons:")
        print(f"Crowded or Empty scenes (>{stats['max_detections_threshold']} poses): {stats['crowded_empty_scenes']}")
        print(f"No pose detected: {stats['no_pose_detected']}")
        print(f"Invalid files: {stats['invalid_files']}")
        
        if self.stats['oks_scores']:
            oks_scores = np.array(self.stats['oks_scores'])
            print(f"\nOKS Distribution:")
            print(f"Min: {np.min(oks_scores):.3f}")
            print(f"25th percentile: {np.percentile(oks_scores, 25):.3f}")
            print(f"Median: {np.median(oks_scores):.3f}")
            print(f"75th percentile: {np.percentile(oks_scores, 75):.3f}")
            print(f"Max: {np.max(oks_scores):.3f}")

            if self.stats_csv is not None:
                csv_path = _shard_csv_path(self.stats_csv)
                pd.DataFrame({
                    'oks_scores': self.stats['oks_scores'],
                    'densepose_paths': self.stats['densepose_paths'],
                    'images_paths': self.stats['images_paths'],
                }).to_csv(csv_path, index=False)
                print(f"Wrote per-sample stats to {csv_path}")

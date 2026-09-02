import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ultralytics import YOLO
from smplx.joint_names import JOINT_NAMES

# YOLO pose estimation joint names in order
YOLO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Mapping from YOLO joints to SMPLX joint indices (full 144-name list).
JOINTS_MAPPING = [JOINT_NAMES.index(joint) for joint in YOLO_JOINTS]

# Mapping from YOLO joints to SMPL output indices. smplx.SMPL.forward returns
# 45 joints: 24 standard SMPL + 21 regressed extras with face landmarks at
# 24-28. See SPIN constants.JOINT_MAP for the canonical correspondence.
SMPL_JOINTS_MAPPING = [
    24, 26, 25, 28, 27,   # nose, L-eye, R-eye, L-ear, R-ear
    16, 17, 18, 19, 20, 21,   # L-sho, R-sho, L-elb, R-elb, L-wri, R-wri
    1, 2, 4, 5, 7, 8,         # L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle
]


def pick_joints_mapping(joints_2d):
    """Return the right indices into joints_2d (45 -> SMPL, larger -> SMPLX)."""
    n = joints_2d.shape[0]
    if n <= 50:
        return SMPL_JOINTS_MAPPING
    return JOINTS_MAPPING

# COCO dataset keypoint detection sigmas for OKS computation
COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


@dataclass
class PoseDetectionResult:
    """Result of pose detection on an image."""
    keypoints: np.ndarray  # Shape: (17, 3) - (x, y, confidence)
    bbox_area: float
    num_detections: int
    valid_sample: bool = True


class KeypointLoader:
    """Handles loading and processing of keypoint data using YOLO."""
    
    def __init__(self, model_name: str = "yolov8x-pose.pt"):
        """
        Initialize the keypoint loader.
        
        Args:
            model_name: YOLO model name or path
        """
        self.pose_model = YOLO(model_name, task="pose", verbose=False)
    
    def predict_keypoints(self, image: np.ndarray, smpl_joints: Optional[np.ndarray] = None, 
                         max_detections: int = 4) -> PoseDetectionResult:
        """
        Predict keypoints on image and select best matching pose.
        
        Args:
            image: Input image (RGB format)
            smpl_joints: SMPL joints for pose matching (optional)
            max_detections: Maximum allowed detections before marking as crowded
            
        Returns:
            PoseDetectionResult with keypoints and metadata
        """
        # Convert RGB to BGR for YOLO
        yolo_results = self.pose_model(image[..., ::-1], verbose=False)
        
        all_poses = []
        all_boxes = []
        
        for result in yolo_results:
            if result.keypoints is not None:
                all_poses.append(result.keypoints.data.cpu().numpy())
                all_boxes.append(result.boxes.xywh.cpu().numpy())
        
        if not all_poses:
            return PoseDetectionResult(
                keypoints=np.zeros((17, 3)),
                bbox_area=0.0,
                num_detections=0
            )
        
        all_poses = np.concatenate(all_poses, axis=0)
        all_boxes = np.concatenate(all_boxes, axis=0)
        
        num_detections = len(all_poses)
        valid_sample = num_detections > 0 and num_detections <= max_detections
        
        # Select best pose (either by SMPL matching or highest confidence)
        if smpl_joints is not None:
            best_idx = self._select_best_pose_by_distance(all_poses, smpl_joints)
        else:
            best_idx = self._select_best_pose_by_confidence(all_poses)
        
        if best_idx is None:
            return PoseDetectionResult(
                keypoints=np.zeros((17, 3)),
                bbox_area=0.0,
                num_detections=num_detections,
                valid_sample=valid_sample
            )
        
        best_pose = all_poses[best_idx]
        bbox = all_boxes[best_idx]
        bbox_area = bbox[2] * bbox[3]
        
        return PoseDetectionResult(
            keypoints=best_pose,
            bbox_area=bbox_area,
            num_detections=num_detections,
            valid_sample=valid_sample
        )
    
    def _select_best_pose_by_distance(self, poses: np.ndarray, smpl_joints: np.ndarray) -> Optional[int]:
        """Select pose with minimum L2 distance to SMPL joints."""
        best_idx = None
        best_distance = float('inf')
        
        for i, pose in enumerate(poses):
            # Require at least 4 visible keypoints
            if (pose[:, 2] > 0.3).sum() < 4:
                continue
            
            distance = np.linalg.norm(pose[:, :2] - smpl_joints, axis=1).mean()
            if distance < best_distance:
                best_distance = distance
                best_idx = i
        
        return best_idx
    
    def _select_best_pose_by_confidence(self, poses: np.ndarray) -> Optional[int]:
        """Select pose with highest average confidence."""
        best_idx = None
        best_confidence = -1.0
        
        for i, pose in enumerate(poses):
            # Require at least 4 visible keypoints
            if (pose[:, 2] > 0.3).sum() < 4:
                continue
            
            avg_confidence = pose[:, 2].mean()
            if avg_confidence > best_confidence:
                best_confidence = avg_confidence
                best_idx = i
        
        return best_idx


class OKSMetric:
    """Computes Object Keypoint Similarity metric."""
    
    def __init__(self, sigmas: np.ndarray = COCO_SIGMAS):
        """
        Initialize OKS metric.
        
        Args:
            sigmas: Per-joint standard deviations for OKS computation
        """
        self.sigmas = sigmas
    
    def compute_oks(self, gt_joints: np.ndarray, pred_joints: np.ndarray, bbox_area: float) -> float:
        """
        Compute OKS score between ground truth and predicted keypoints.
        
        Args:
            gt_joints: Ground truth keypoints (17, 3) - (x, y, confidence)
            pred_joints: Predicted keypoints (17, 2) or (17, 3)
            bbox_area: Area of bounding box for normalization
            
        Returns:
            OKS score between 0 and 1, or -1.0 if invalid
        """
        # Handle input shapes
        if pred_joints.shape[1] == 3:
            pred_vis = pred_joints[:, 2] > 0.3
            pred_coords = pred_joints[:, :2]
        else:
            pred_vis = (pred_joints[:, 0] > 0) & (pred_joints[:, 1] > 0)
            pred_coords = pred_joints
        
        gt_vis = gt_joints[:, 2] > 0.3 if gt_joints.shape[1] == 3 else np.ones(len(gt_joints), dtype=bool)
        gt_coords = gt_joints[:, :2] if gt_joints.shape[1] >= 2 else gt_joints
        
        valid_mask = gt_vis & pred_vis
        
        if not valid_mask.any() or bbox_area <= 0:
            return -1.0
        
        # Compute distances
        d = np.linalg.norm(gt_coords - pred_coords, axis=1)
        
        # Compute OKS per joint
        k = 2 * self.sigmas
        oks_per_joint = np.exp(-d**2 / (2 * bbox_area * k**2)) * valid_mask
        
        return oks_per_joint.sum() / valid_mask.sum()


def compute_oks(image: np.ndarray, smpl_joints: np.ndarray, 
                keypoint_loader: Optional[KeypointLoader] = None,
                oks_metric: Optional[OKSMetric] = None,
                max_detections: int = 4) -> Tuple[float, bool, int]:
    """
    Convenience function to compute OKS score for an image and SMPL joints.
    
    Args:
        image: Input image (RGB format)
        smpl_joints: SMPL joint coordinates (17, 2)
        keypoint_loader: KeypointLoader instance (created if None)
        oks_metric: OKSMetric instance (created if None)
        max_detections: Maximum detections before marking as crowded
        
    Returns:
        Tuple of (oks_score, is_crowded, num_detections)
        oks_score: -1.0 if invalid, otherwise 0.0-1.0
        is_crowded: True if more than max_detections found
        num_detections: Number of poses detected
    """
    if keypoint_loader is None:
        keypoint_loader = KeypointLoader()
    
    if oks_metric is None:
        oks_metric = OKSMetric()
    
    # Predict keypoints
    result = keypoint_loader.predict_keypoints(image, smpl_joints, max_detections)
    
    if result.num_detections == 0:
        return -1.0, False, 0
    
    if not result.valid_sample:
        return -1.0, False, result.num_detections
    
    # Compute OKS score
    oks_score = oks_metric.compute_oks(
        result.keypoints,  # YOLO as GT
        smpl_joints,       # SMPL as predictions
        result.bbox_area
    )
    
    return oks_score, result.valid_sample, result.num_detections

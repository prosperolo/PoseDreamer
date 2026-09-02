"""Score generated images against their SMPL-X labels with OKS (YOLO-pose
predictions vs reprojected joints); used for DPO pair construction and
filter-threshold tuning."""
import os 
import deepdish as dd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import fire
import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from posedreamer.filtering.functional.oks import (
    KeypointLoader, OKSMetric, JOINTS_MAPPING, compute_oks, pick_joints_mapping
)


@dataclass
class KeypointComparison:
    """Data container for keypoint comparison results."""
    smpl_joints: np.ndarray
    yolo_joints: np.ndarray
    bbox_area: float
    oks_score: float = -1.0
    image: Optional[np.ndarray] = None
    metadata: Optional[dict] = None


class KeypointVisualizer:
    """Handles visualization of keypoint comparisons."""
    
    def save_comparison(self, comparison: KeypointComparison, save_path: str):
        """Save keypoint comparison visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # SMPL visualization
        smpl_image = comparison.image.copy()
        for joint_idx, joint in enumerate(comparison.smpl_joints):
            x, y = round(joint[0]), round(joint[1])
            cv2.circle(smpl_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(smpl_image, str(joint_idx), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        ax1.imshow(smpl_image[..., ::-1])
        ax1.set_title('SMPL Keypoints (Predictions)', fontsize=16)
        ax1.axis('off')
        
        # YOLO visualization
        yolo_image = comparison.image.copy()
        for joint_idx, joint in enumerate(comparison.yolo_joints):
            x, y = round(joint[0].item()), round(joint[1].item())
            cv2.circle(yolo_image, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(yolo_image, str(joint_idx), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        
        ax2.imshow(yolo_image[..., ::-1])
        ax2.set_title('YOLO Keypoints (Ground Truth)', fontsize=16)
        ax2.axis('off')
        
        fig.suptitle(f'OKS Score: {comparison.oks_score:.3f}', fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def load_comparison(metadata_path: str, keypoint_loader: KeypointLoader, oks_metric: OKSMetric) -> Optional[KeypointComparison]:
    """Load a single keypoint comparison from metadata file using functional implementation."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    smplx_path = metadata['condition_path'].replace('densepose-renders', 'smplx-gt-labels').replace('.png', '.h5')
    if not os.path.exists(smplx_path):
        # SportsCap data uses smpl-gt-labels (no x). compute_oks auto-detects
        # the joint convention from joints_2d.shape, so the same code path works.
        smplx_path = metadata['condition_path'].replace('densepose-renders', 'smpl-gt-labels').replace('.png', '.h5')
    if not os.path.exists(smplx_path):
        return None
        
    generated_image_path = metadata['generated_image_path']
    if not os.path.exists(generated_image_path):
        return None
    
    smplx_data = dd.io.load(smplx_path)
    # SportsCap data has SMPL (45) joints; the laion data has SMPL-X (144).
    # pick_joints_mapping picks the right indices by joints_2d.shape.
    smpl_joints = smplx_data["joints_2d"][pick_joints_mapping(smplx_data["joints_2d"])]
    
    generated_image = cv2.imread(generated_image_path)
    condition_image = cv2.imread(metadata['condition_path'])
    generated_image = cv2.resize(generated_image, condition_image.shape[:2])
    
    # Convert BGR to RGB for the functional implementation
    image_rgb = generated_image[..., ::-1]
    
    # max_detections is permissive here: predict_keypoints already picks the
    # best-matching pose, and for DPO we want every generated image to receive
    # a real OKS so it can be ranked — even if YOLO trips on equipment or
    # reflections in the scene. (The filter pipeline uses max_detections=4
    # because *there* we want to reject crowded scenes outright.)
    # NB: compute_oks's second return value is `valid_sample` (True = good),
    # despite being unpacked as `is_crowded` elsewhere in the codebase. Treat
    # !valid_sample (no detections, or more than max_detections found) as a fail.
    oks_score, valid_sample, num_detections = compute_oks(
        image=image_rgb,
        smpl_joints=smpl_joints,
        keypoint_loader=keypoint_loader,
        oks_metric=oks_metric,
        max_detections=20
    )

    if not valid_sample or oks_score < 0:
        return None
    
    # Get the actual YOLO keypoints for visualization
    result = keypoint_loader.predict_keypoints(image_rgb, smpl_joints, max_detections=4)
    
    return KeypointComparison(
        smpl_joints=smpl_joints,
        yolo_joints=result.keypoints,
        bbox_area=result.bbox_area,
        oks_score=oks_score,
        image=generated_image,
        metadata=metadata
    )


def compute_keypoint_similarity(metadata_folder: str, save_folder: str, num_images: Optional[int] = None, visualize: bool = False, resume: bool = True):
    """
    Add OKS metrics to metadata files and save updated JSONs using the unified functional implementation.

    Args:
        metadata_folder: Path to folder containing JSON metadata files
        save_folder: Path to folder where updated JSONs will be saved
        num_images: Number of images to process (default: all)
        visualize: Whether to save visualizations (default: False)
        resume: Skip files whose save_folder JSON already has an oks_metric (default: True)
    """
    # Use the unified functional implementation
    loader = KeypointLoader()
    metric = OKSMetric()

    Path(save_folder).mkdir(parents=True, exist_ok=True)
    if visualize:
        visualizer = KeypointVisualizer()
        vis_folder = Path(save_folder) / "visuals"
        vis_folder.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted([f for f in os.listdir(metadata_folder) if f.endswith('.json')])
    if num_images is not None:
        json_files = json_files[:num_images]
    
    if not json_files:
        print("No JSON files found in metadata folder")
        return
    
    oks_scores = []
    crowded_scenes = 0
    no_pose_detected = 0
    
    skipped = 0
    for idx, json_file in tqdm.tqdm(enumerate(json_files)):
        metadata_path = os.path.join(metadata_folder, json_file)
        save_path = Path(save_folder) / json_file

        # Resumability: if the output already has an oks_metric, skip — re-scoring
        # 71k samples on every restart is a non-starter. Failed (-1) entries are
        # also kept; remove them manually if you want to retry.
        if resume and save_path.exists():
            try:
                with open(save_path, 'r') as f:
                    existing = json.load(f)
                if 'oks_metric' in existing:
                    skipped += 1
                    score = existing['oks_metric']
                    if score >= 0:
                        oks_scores.append(score)
                    continue
            except (json.JSONDecodeError, OSError):
                pass  # fall through and re-score this one

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # Some input JSONs are empty/corrupt — skip them instead of aborting
            # the whole 71k-sample run.
            print(f"Skipping malformed input {json_file}: {e}")
            continue

        comparison = load_comparison(metadata_path, loader, metric)
        if comparison is None:
            oks_score = -1.0
            print(f"Failed to load {json_file} - setting OKS to -1")
        else:
            oks_score = comparison.oks_score
            if oks_score < 0:
                oks_score = -1.0
                print(f"No valid keypoints for {json_file} - setting OKS to -1")
            
            if visualize and comparison is not None:
                vis_path = vis_folder / f"keypoints_comparison_{idx:03d}_{json_file.replace('.json', '.png')}"
                visualizer.save_comparison(comparison, str(vis_path))
        
        metadata['oks_metric'] = oks_score
        # `save_path` was set above for the resume check; reuse it.
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if oks_score >= 0:
            oks_scores.append(oks_score)
            print(f"OKS: {oks_score:.3f} - {json_file}")
        else:
            print(f"OKS: -1.000 - {json_file}")
    
    if oks_scores:
        print(f"\n=== OKS METRICS SUMMARY ===")
        print(f"Valid samples: {len(oks_scores)}/{len(json_files)}  (skipped existing: {skipped})")
        print(f"Mean OKS: {np.mean(oks_scores):.3f}")
        print(f"Std OKS: {np.std(oks_scores):.3f}")
        print(f"Failed samples: {len(json_files) - len(oks_scores)}")
    
    print(f"\nCompleted processing. Updated metadata saved to: {save_folder}")
    if visualize:
        print(f"Visualizations saved to: {vis_folder}")


if __name__ == "__main__":
    fire.Fire(compute_keypoint_similarity)

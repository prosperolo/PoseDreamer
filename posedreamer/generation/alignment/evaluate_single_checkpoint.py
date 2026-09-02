import os
import torch
import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Dict
import tqdm
import fire
import deepdish as dd
import random

# EasyControl imports
from EasyControl.train.src.pipeline import FluxPipeline
from EasyControl.train.src.transformer_flux import FluxTransformer2DModel
from EasyControl.train.src.lora_helper import set_single_lora
from posedreamer.label_generation.caption_processor import CaptionProcessor

# OKS metric imports
from ultralytics import YOLO
from smplx.joint_names import JOINT_NAMES


# Set random seed
random.seed(33)
YOLO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

JOINTS_MAPPING = [JOINT_NAMES.index(joint) for joint in YOLO_JOINTS]
COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


class OKSMetric:
    """Computes Object Keypoint Similarity metric."""
    
    def __init__(self):
        self.pose_model = YOLO("yolov8x-pose.pt", task="pose", verbose=False)
        self.sigmas = COCO_SIGMAS
    
    def compute_oks(self, generated_image: np.ndarray, smpl_joints: np.ndarray) -> float:
        """Compute OKS score for generated image against SMPL ground truth."""
        yolo_joints, bbox_area = self._predict_keypoints(generated_image, smpl_joints)
        if yolo_joints is None:
            return -1.0
        
        return self._compute_oks_score(yolo_joints, smpl_joints, bbox_area)
    
    def _predict_keypoints(self, image: np.ndarray, smpl_joints: np.ndarray) -> tuple:
        """Predict keypoints and select best matching pose."""
        yolo_results = self.pose_model(image[..., ::-1])
        
        all_poses = []
        all_boxes = []
        for result in yolo_results:
            if result.keypoints is not None:
                all_poses.append(result.keypoints.data.cpu().numpy())
                all_boxes.append(result.boxes.xywh.cpu().numpy())
        
        if not all_poses:
            return None, 0
        
        all_poses = np.concatenate(all_poses, axis=0)
        all_boxes = np.concatenate(all_boxes, axis=0)
        
        best_idx = self._select_best_pose(all_poses, smpl_joints)
        if best_idx is None:
            return None, 0
            
        best_pose = all_poses[best_idx]
        bbox = all_boxes[best_idx]
        bbox_area = bbox[2] * bbox[3]
        
        return best_pose, bbox_area
    
    def _select_best_pose(self, poses: np.ndarray, smpl_joints: np.ndarray):
        """Select pose with minimum L2 distance to SMPL joints."""
        best_idx = None
        best_distance = float('inf')
        
        for i, pose in enumerate(poses):
            if (pose[:, 2] > 0.3).sum() < 4:
                continue
            distance = np.linalg.norm(pose[:, :2] - smpl_joints, axis=1).mean()
            if distance < best_distance:
                best_distance = distance
                best_idx = i
        
        return best_idx
    
    def _compute_oks_score(self, gt_joints: np.ndarray, pred_joints: np.ndarray, bbox_area: float) -> float:
        """Compute OKS score with YOLO as GT and SMPL as predictions."""
        gt_vis = gt_joints[:, 2] > 0.3 
        pred_vis = (pred_joints[:, 0] > 0) & (pred_joints[:, 1] > 0)
        valid_mask = gt_vis & pred_vis
        
        if not valid_mask.any():
            return -1.0
        
        d = np.linalg.norm(gt_joints[:, :2] - pred_joints[:, :2], axis=1)
        k = 2 * self.sigmas
        oks_per_joint = np.exp(-d**2 / (2 * bbox_area * k**2)) * valid_mask
        
        return oks_per_joint.sum() / valid_mask.sum()


def create_base_pipeline(base_model_path: str, control_lora_path: str, device: str = "cuda") -> FluxPipeline:
    """Create base pipeline with EasyControl LoRA."""
    pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device
    )
    set_single_lora(transformer, control_lora_path, lora_weights=[1], cond_size=512)
    pipe.transformer = transformer
    pipe.to(device)
    return pipe


def create_dpo_pipeline(base_model_path: str, control_lora_path: str, dpo_checkpoint: str, device: str = "cuda") -> FluxPipeline:
    """Create pipeline with base + DPO LoRA."""
    pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device
    )
    
    # Step 1: Apply base EasyControl LoRA
    set_single_lora(transformer, control_lora_path, lora_weights=[1], cond_size=512)
    
    # Step 2: Add DPO PEFT LoRA layer
    if dpo_checkpoint.endswith('.ckpt'):
        from peft import get_peft_model, LoraConfig
        
        peft_config = LoraConfig(
            r=128,
            lora_alpha=128.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        
        peft_model = get_peft_model(transformer, peft_config)
        checkpoint = torch.load(dpo_checkpoint, map_location='cpu')
        
        transformed_checkpoint = {
            (f"base_model.model.{key}" if not key.startswith('base_model.') else key).replace('.weight', '.default.weight') if 'lora_' in key else key: value
            for key, value in checkpoint.items()
        }
        
        peft_model.load_state_dict(transformed_checkpoint, strict=False)
        transformer = peft_model
    else:
        from peft import PeftModel
        transformer = PeftModel.from_pretrained(transformer, dpo_checkpoint)
    
    pipe.transformer = transformer
    pipe.to(device)
    return pipe


def generate_and_evaluate_batch(
    pipeline: FluxPipeline,
    densepose_files: List[Path],
    metadata_path: str,
    oks_metric: OKSMetric,
    batch_start: int,
    batch_size: int,
    save_dir: Path,
    model_name: str,
    **generation_kwargs
) -> List[float]:
    """Generate and evaluate a batch of samples."""
    batch_oks_scores = []
    caption_processor = CaptionProcessor()
    
    for i in range(batch_size):
        sample_idx = batch_start + i
        if sample_idx >= len(densepose_files):
            break
            
        densepose_file = densepose_files[sample_idx]
        
        # Load metadata
        metadata_file = Path(metadata_path) / f"{densepose_file.stem}.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Process inputs
        control_image = Image.open(densepose_file).convert('RGB')
        resized_control_image = control_image.resize((1024, 1024))
        caption = metadata.get("caption", "A person")
        caption = caption_processor.add_ethnic_labels(caption)
        caption = f"A ultra high DSLR image showing the following content: {caption}. Maximum detail, detailed face, perfect hands, perfection"
        
        # Generate
        seed = 1234 + sample_idx
        result = pipeline(
            caption,
            spatial_images=[resized_control_image],
            subject_images=[],
            generator=torch.Generator("cpu").manual_seed(seed),
            **generation_kwargs
        ).images[0]
        
        # Save generated image
        output_path = save_dir / f"{model_name}_{sample_idx:03d}.jpg"
        result = result.resize(control_image.size)
        result.save(output_path)
        
        # Save corresponding densepose control image
        densepose_output_path = save_dir / f"{model_name}_{sample_idx:03d}_densepose.png"
        control_image.save(densepose_output_path)
        
        # Compute OKS
        oks_score = -1.0
        # Get SMPL keypoints
        smplx_path = str(densepose_file).replace('densepose-renders', 'smplx-gt-labels').replace('.png', '.h5')
        if os.path.exists(smplx_path):
            smplx_data = dd.io.load(smplx_path)
            smpl_joints = smplx_data["joints_2d"][JOINTS_MAPPING]
            
            # Compute OKS
            generated_image = np.array(result)
            oks_score = oks_metric.compute_oks(generated_image, smpl_joints)
            print(f"OKS score: {oks_score}")
                
     
        batch_oks_scores.append(oks_score)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return batch_oks_scores


def evaluate_single_checkpoint(
    base_model_path: str,
    control_lora_path: str,
    dpo_checkpoint: str,
    densepose_path: str,
    metadata_path: str,
    save_folder: str,
    num_samples: int = 100,
    batch_size: int = 10,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 25,
    max_sequence_length: int = 512,
    cond_size: int = 512
):
    """
    Evaluate single DPO checkpoint with 200 generations vs baseline.
    
    Args:
        base_model_path: Path to base FLUX model
        control_lora_path: Path to EasyControl LoRA
        dpo_checkpoint: Path to DPO checkpoint
        densepose_path: Path to densepose images folder
        metadata_path: Path to metadata folder
        save_folder: Output folder
        num_samples: Total samples to generate (default: 200)
        batch_size: Report OKS every N samples (default: 50)
        height: Image height (default: 1024)
        width: Image width (default: 1024)
        guidance_scale: CFG scale (default: 3.5)
        num_inference_steps: Number of steps (default: 25)
        max_sequence_length: Max sequence length (default: 512)
        cond_size: Condition size (default: 512)
    """
    generation_kwargs = {
        "height": height, "width": width, "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps, "max_sequence_length": max_sequence_length,
        "cond_size": cond_size
    }
    
    # Setup
    save_dir = Path(save_folder)
    base_dir = save_dir / "baseline"
    dpo_dir = save_dir / "dpo"
    base_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)
    
    # Get densepose files
    densepose_files = list(Path(densepose_path).glob("*.png"))
    densepose_files = random.sample(densepose_files, min(num_samples, len(densepose_files)))
    
    print(f"🎯 Evaluating {len(densepose_files)} samples in batches of {batch_size}")
    print(f"📁 Results will be saved to: {save_folder}")
    
    # Initialize OKS metric
    oks_metric = OKSMetric()
    
    # Track results
    baseline_oks_all = []
    dpo_oks_all = []
    
    # Process in batches
    num_batches = (len(densepose_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(densepose_files))
        actual_batch_size = batch_end - batch_start
        
        print(f"\n🔄 Batch {batch_idx + 1}/{num_batches} (samples {batch_start + 1}-{batch_end})")
        
        # Generate baseline samples
        print("🎨 Generating baseline samples...")
        baseline_pipeline = create_base_pipeline(base_model_path, control_lora_path)
        
        baseline_batch_oks = generate_and_evaluate_batch(
            baseline_pipeline, densepose_files, metadata_path, oks_metric,
            batch_start, actual_batch_size, base_dir, "baseline", **generation_kwargs
        )
        
        del baseline_pipeline
        torch.cuda.empty_cache()
        
        # Generate DPO samples  
        print("🎨 Generating DPO samples...")
        dpo_pipeline = create_dpo_pipeline(base_model_path, control_lora_path, dpo_checkpoint)
        
        dpo_batch_oks = generate_and_evaluate_batch(
            dpo_pipeline, densepose_files, metadata_path, oks_metric,
            batch_start, actual_batch_size, dpo_dir, "dpo", **generation_kwargs
        )
        
        del dpo_pipeline
        torch.cuda.empty_cache()
        
        # Update cumulative results
        baseline_oks_all.extend(baseline_batch_oks)
        dpo_oks_all.extend(dpo_batch_oks)
        
        # Compute batch statistics
        valid_baseline = [s for s in baseline_batch_oks if s >= 0]
        valid_dpo = [s for s in dpo_batch_oks if s >= 0]
        
        baseline_mean = np.mean(valid_baseline) if valid_baseline else -1
        dpo_mean = np.mean(valid_dpo) if valid_dpo else -1
        
        # Compute cumulative statistics
        valid_baseline_all = [s for s in baseline_oks_all if s >= 0]
        valid_dpo_all = [s for s in dpo_oks_all if s >= 0]
        
        baseline_cumulative = np.mean(valid_baseline_all) if valid_baseline_all else -1
        dpo_cumulative = np.mean(valid_dpo_all) if valid_dpo_all else -1
        
        # Print batch results
        print(f"📊 Batch {batch_idx + 1} Results:")
        print(f"   Baseline OKS: {baseline_mean:.3f} (valid: {len(valid_baseline)}/{actual_batch_size})")
        print(f"   DPO OKS:      {dpo_mean:.3f} (valid: {len(valid_dpo)}/{actual_batch_size})")
        if baseline_mean >= 0 and dpo_mean >= 0:
            improvement = dpo_mean - baseline_mean
            print(f"   Improvement:  {improvement:+.3f}")
        
        print(f"📈 Cumulative Results (samples 1-{batch_end}):")
        print(f"   Baseline OKS: {baseline_cumulative:.3f} (valid: {len(valid_baseline_all)}/{batch_end})")
        print(f"   DPO OKS:      {dpo_cumulative:.3f} (valid: {len(valid_dpo_all)}/{batch_end})")
        if baseline_cumulative >= 0 and dpo_cumulative >= 0:
            cumulative_improvement = dpo_cumulative - baseline_cumulative
            print(f"   Improvement:  {cumulative_improvement:+.3f}")
    
    # Final summary
    valid_baseline_final = [s for s in baseline_oks_all if s >= 0]
    valid_dpo_final = [s for s in dpo_oks_all if s >= 0]
    
    print(f"\n🎉 Final Results ({len(densepose_files)} samples):")
    print(f"   Baseline OKS: {np.mean(valid_baseline_final):.3f} ± {np.std(valid_baseline_final):.3f}")
    print(f"   DPO OKS:      {np.mean(valid_dpo_final):.3f} ± {np.std(valid_dpo_final):.3f}")
    
    if valid_baseline_final and valid_dpo_final:
        final_improvement = np.mean(valid_dpo_final) - np.mean(valid_baseline_final)
        print(f"   Improvement:  {final_improvement:+.3f}")
        
        # Statistical significance (basic t-test approximation)
        diff_scores = [dpo - base for dpo, base in zip(valid_dpo_final, valid_baseline_final) if dpo >= 0 and base >= 0]
        if diff_scores:
            print(f"   Per-sample Δ: {np.mean(diff_scores):+.3f} ± {np.std(diff_scores):.3f}")
    
    # Save detailed results
    results = {
        "baseline_oks": baseline_oks_all,
        "dpo_oks": dpo_oks_all,
        "baseline_mean": np.mean(valid_baseline_final) if valid_baseline_final else -1,
        "dpo_mean": np.mean(valid_dpo_final) if valid_dpo_final else -1,
        "baseline_std": np.std(valid_baseline_final) if valid_baseline_final else -1,
        "dpo_std": np.std(valid_dpo_final) if valid_dpo_final else -1,
        "valid_baseline": len(valid_baseline_final),
        "valid_dpo": len(valid_dpo_final),
        "total_samples": len(densepose_files)
    }
    
    results_path = save_dir / "detailed_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_path}")


if __name__ == "__main__":
    fire.Fire(evaluate_single_checkpoint) 
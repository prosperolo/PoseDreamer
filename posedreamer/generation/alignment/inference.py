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
import matplotlib.pyplot as plt
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
random.seed(42)
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
    """Create fresh pipeline with base + DPO LoRA."""
    pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device
    )
    
    # Step 1: Apply base EasyControl LoRA (same as training)
    set_single_lora(transformer, control_lora_path, lora_weights=[1], cond_size=512)
    print("✅ Applied base EasyControl LoRA")
    
    # Step 2: Add DPO PEFT LoRA layer on top (same as training)
    if dpo_checkpoint.endswith('.ckpt'):
        from peft import get_peft_model, LoraConfig
        
        # Create PEFT config (should match trainer.py)
        peft_config = LoraConfig(
            r=128,
            lora_alpha=128.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        
        # Add PEFT LoRA layer
        peft_model = get_peft_model(transformer, peft_config)
        
        # Load DPO checkpoint weights into PEFT layer
        checkpoint = torch.load(dpo_checkpoint, map_location='cpu')
        
        # Transform checkpoint keys to PEFT format
        transformed_checkpoint = {
            (f"base_model.model.{key}" if not key.startswith('base_model.') else key).replace('.weight', '.default.weight') if 'lora_' in key else key: value
            for key, value in checkpoint.items()
        }
        
        # Load the checkpoint
        peft_model.load_state_dict(transformed_checkpoint, strict=False)
        print(f"✅ Applied DPO LoRA checkpoint: {Path(dpo_checkpoint).name}")
        
        transformer = peft_model
    else:
        # PEFT adapter folder
        from peft import PeftModel
        transformer = PeftModel.from_pretrained(transformer, dpo_checkpoint)
    
    pipe.transformer = transformer
    pipe.to(device)
    return pipe


def generate_sample(pipeline: FluxPipeline, control_image: Image.Image, caption: str, seed: int, **kwargs) -> Image.Image:
    """Generate single sample."""
    result = pipeline(
        caption,
        spatial_images=[control_image],
        subject_images=[],
        generator=torch.Generator("cpu").manual_seed(seed),
        **kwargs
    ).images[0]
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def generate_base_samples(
    base_model_path: str,
    control_lora_path: str,
    densepose_path: str,
    metadata_path: str,
    save_dir: str,
    num_samples: int = 10,
    **generation_kwargs
) -> List[Dict]:
    """Generate base samples once."""
    base_dir = Path(save_dir) / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we need to generate or can skip
    densepose_files = list(Path(densepose_path).glob("*.png"))
    densepose_files = random.sample(densepose_files, num_samples)
    base_samples = []
    pipeline = None
    
    print(f"🎨 Checking/Generating {num_samples} base samples...")
    for i, densepose_file in enumerate(tqdm.tqdm(densepose_files)):
        output_path = base_dir / f"base_{i:03d}.jpg"
        
        # Check if already exists
        if output_path.exists():
            # Load metadata for existing sample
            metadata_file = Path(metadata_path) / f"{densepose_file.stem}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                control_image = Image.open(densepose_file).convert('RGB')
                caption_processor = CaptionProcessor()
                caption = metadata.get("caption", "A person")
                caption = caption_processor.add_ethnic_labels(caption)
                caption = f"A ultra high DSLR image showing the following content: {caption}. Maximum detail, detailed face, perfect hands, perfection"
                seed = 42 + i
                
                base_samples.append({
                    "index": i,
                    "control_image": control_image,
                    "caption": caption,
                    "seed": seed,
                    "base_output": str(output_path),
                    "densepose_file": densepose_file
                })
                print(f"✅ Using existing base sample {i:03d}")
                continue
        
        # Generate new sample
        if pipeline is None:
            pipeline = create_base_pipeline(base_model_path, control_lora_path)
            caption_processor = CaptionProcessor()
        
        # Load metadata
        metadata_file = Path(metadata_path) / f"{densepose_file.stem}.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Process inputs
        control_image = Image.open(densepose_file).convert('RGB')
        caption = metadata.get("caption", "A person")
        caption = caption_processor.add_ethnic_labels(caption)
        caption = f"A ultra high DSLR image showing the following content: {caption}. Maximum detail, detailed face, perfect hands, perfection"
        
        # Generate
        seed = 42 + i
        result = generate_sample(pipeline, control_image, caption, seed, **generation_kwargs)
        
        # Save
        result.save(output_path)
        print(f"🎨 Generated base sample {i:03d}")
        
        base_samples.append({
            "index": i,
            "control_image": control_image,
            "caption": caption,
            "seed": seed,
            "base_output": str(output_path),
            "densepose_file": densepose_file
        })
    
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()
    
    return base_samples


def generate_dpo_samples(
    base_model_path: str,
    control_lora_path: str,
    dpo_checkpoint: str,
    base_samples: List[Dict],
    save_dir: str,
    checkpoint_name: str,
    **generation_kwargs
) -> List[str]:
    """Generate DPO samples with fresh pipeline."""
    dpo_dir = Path(save_dir) / "dpo" / checkpoint_name
    dpo_dir.mkdir(parents=True, exist_ok=True)
    
    dpo_outputs = []
    pipeline = None
    
    print(f"🎨 Checking/Generating DPO samples for {checkpoint_name}...")
    for sample in tqdm.tqdm(base_samples):
        output_path = dpo_dir / f"dpo_{checkpoint_name}_{sample['index']:03d}.jpg"
        
        # Check if already exists
        if output_path.exists():
            dpo_outputs.append(str(output_path))
            print(f"✅ Using existing DPO sample {sample['index']:03d}")
            continue
        
        # Generate new sample
        if pipeline is None:
            pipeline = create_dpo_pipeline(base_model_path, control_lora_path, dpo_checkpoint)
        
        result = generate_sample(
            pipeline, 
            sample["control_image"], 
            sample["caption"], 
            sample["seed"], 
            **generation_kwargs
        )
        
        result.save(output_path)
        dpo_outputs.append(str(output_path))
        print(f"🎨 Generated DPO sample {sample['index']:03d}")
    
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()
    
    return dpo_outputs


def overlay_images(densepose, rgb):
    """Overlay densepose control on generated image."""
    alpha = 0.5
    return cv2.addWeighted(rgb, 1 - alpha, densepose, alpha, 0)


def visualize_keypoints(image: np.ndarray, yolo_joints: np.ndarray, smpl_joints: np.ndarray, title: str = "") -> np.ndarray:
    """Create keypoint visualization with both YOLO and SMPL keypoints."""
    vis_image = image.copy()
    
    # Draw SMPL keypoints in green
    for joint_idx, joint in enumerate(smpl_joints):
        if joint[0] > 0 and joint[1] > 0:  # Valid joint
            x, y = int(joint[0]), int(joint[1])
            cv2.circle(vis_image, (x, y), 6, (0, 255, 0), -1)  # Green for SMPL
            cv2.putText(vis_image, str(joint_idx), (x + 8, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw YOLO keypoints in red
    for joint_idx, joint in enumerate(yolo_joints):
        if joint[2] > 0.3:  # Confident keypoint
            x, y = int(joint[0]), int(joint[1])
            cv2.circle(vis_image, (x, y), 4, (255, 0, 0), -1)  # Red for YOLO
    
    # Add legend
    cv2.putText(vis_image, "Green: SMPL (GT), Red: YOLO (Pred)", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return vis_image


def create_individual_comparison_with_oks(
    base_sample: Dict, 
    dpo_output: str, 
    checkpoint_name: str, 
    save_dir: str,
    oks_metric: OKSMetric
):
    """Create individual comparison with OKS metrics."""
    comparison_dir = Path(save_dir) / "comparisons" / checkpoint_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    sample_idx = base_sample["index"]
    
    # Load images
    control_image = base_sample["control_image"]
    control_np = np.array(control_image)
    base_img = np.array(Image.open(base_sample["base_output"]))
    dpo_img = np.array(Image.open(dpo_output))
    
    # Load SMPL ground truth keypoints
    smpl_joints = None
    oks_base = -1.0
    oks_dpo = -1.0
    
    try:
        # Get SMPL keypoints from the densepose file
        densepose_file = base_sample["densepose_file"]
        smplx_path = str(densepose_file).replace('densepose-renders', 'smplx-gt-labels').replace('.png', '.h5')
        
        if os.path.exists(smplx_path):
            smplx_data = dd.io.load(smplx_path)
            smpl_joints = smplx_data["joints_2d"][JOINTS_MAPPING]
            
            # Compute OKS scores
            oks_base = oks_metric.compute_oks(base_img, smpl_joints)
            oks_dpo = oks_metric.compute_oks(dpo_img, smpl_joints)
            
            print(f"Sample {sample_idx}: Base OKS={oks_base:.3f}, DPO OKS={oks_dpo:.3f}")
    except Exception as e:
        print(f"Could not compute OKS for sample {sample_idx}: {e}")
    
    # Create overlays
    base_img = cv2.resize(base_img, (control_np.shape[1], control_np.shape[0]))
    dpo_img = cv2.resize(dpo_img, (control_np.shape[1], control_np.shape[0]))
    base_overlay = overlay_images(control_np, base_img)
    dpo_overlay = overlay_images(control_np, dpo_img)
    
    # Create keypoint visualizations if available
    if smpl_joints is not None and oks_base > 0:
        # Get YOLO predictions for visualizations
        yolo_base, _ = oks_metric._predict_keypoints(base_img, smpl_joints)
        yolo_dpo, _ = oks_metric._predict_keypoints(dpo_img, smpl_joints)
        
        if yolo_base is not None:
            base_keypoints = visualize_keypoints(base_img, yolo_base, smpl_joints)
        else:
            base_keypoints = base_img
            
        if yolo_dpo is not None:
            dpo_keypoints = visualize_keypoints(dpo_img, yolo_dpo, smpl_joints)
        else:
            dpo_keypoints = dpo_img
    else:
        base_keypoints = base_img
        dpo_keypoints = dpo_img
    
    # Create rows: Base and DPO
    base_row = np.hstack([control_np, base_img, base_overlay, base_keypoints])
    dpo_row = np.hstack([control_np, dpo_img, dpo_overlay, dpo_keypoints])
    
    stacked = np.vstack([base_row, dpo_row])
    
    # Add headers
    col_headers = ["Control", "Generated", "Overlay", "Keypoints"]
    width_per_col = stacked.shape[1] // 4
    height_per_row = stacked.shape[0] // 2
    
    for j, header in enumerate(col_headers):
        x_pos = j * width_per_col + 10
        y_pos = 25
        cv2.rectangle(stacked, (x_pos-5, y_pos-20), (x_pos + len(header)*15, y_pos+5), (0, 0, 0), -1)
        cv2.putText(stacked, header, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add row labels with OKS scores
    base_label = f"Base (OKS: {oks_base:.3f})" if oks_base >= 0 else "Base (OKS: N/A)"
    dpo_label = f"DPO-{checkpoint_name} (OKS: {oks_dpo:.3f})" if oks_dpo >= 0 else f"DPO-{checkpoint_name} (OKS: N/A)"
    
    labels = [base_label, dpo_label]
    
    for j, label in enumerate(labels):
        y_pos = j * height_per_row + 60
        cv2.rectangle(stacked, (5, y_pos - 25), (len(label) * 12 + 10, y_pos + 5), (0, 0, 0), -1)
        cv2.putText(stacked, label, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Save individual comparison
    comparison_path = comparison_dir / f"image_{sample_idx:03d}_{checkpoint_name}_stacked.jpg"
    Image.fromarray(stacked).save(comparison_path)
    
    return {"base_oks": oks_base, "dpo_oks": oks_dpo, "comparison_path": str(comparison_path)}


def validate_dpo_alignment(
    base_model_path: str,
    control_lora_path: str,
    dpo_checkpoints_folder: str,
    densepose_path: str,
    metadata_path: str,
    save_folder: str,
    num_samples: int = 10,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 25,
    max_sequence_length: int = 512,
    cond_size: int = 512
):
    """Validate DPO alignment with OKS metrics."""
    generation_kwargs = {
        "height": height, "width": width, "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps, "max_sequence_length": max_sequence_length, "cond_size": cond_size
    }
    
    # Initialize OKS metric
    oks_metric = OKSMetric()
    
    # 1. Generate base samples (once)
    print("STEP 1: Generating Base Samples")
    base_samples = generate_base_samples(
        base_model_path, control_lora_path, densepose_path, metadata_path, save_folder, num_samples, **generation_kwargs
    )
    
    # 2. Find DPO checkpoints
    checkpoint_files = sorted(list(Path(dpo_checkpoints_folder).glob("*.ckpt")) + list(Path(dpo_checkpoints_folder).glob("*/adapter_config.json")))
    print(f"\nSTEP 2: Processing {len(checkpoint_files)} DPO checkpoints")
    
    # 3. Process each checkpoint individually
    results_summary = {}
    
    for checkpoint_file in checkpoint_files:
        if checkpoint_file.name == "adapter_config.json":
            checkpoint_path = str(checkpoint_file.parent)
            checkpoint_name = checkpoint_file.parent.name
        else:
            checkpoint_path = str(checkpoint_file)
            checkpoint_name = checkpoint_file.stem
        
        print(f"\n🔄 Processing checkpoint: {checkpoint_name}")
        
        # Generate DPO samples
        dpo_outputs = generate_dpo_samples(
            base_model_path, control_lora_path, checkpoint_path, base_samples, save_folder, checkpoint_name, **generation_kwargs
        )
        
        # Create individual comparisons with OKS
        print(f"📊 Creating comparisons with OKS metrics for {checkpoint_name}")
        checkpoint_results = []
        
        for base_sample, dpo_output in zip(base_samples, dpo_outputs):
            result = create_individual_comparison_with_oks(
                base_sample, dpo_output, checkpoint_name, save_folder, oks_metric
            )
            checkpoint_results.append(result)
        
        # Calculate summary statistics
        base_oks_scores = [r["base_oks"] for r in checkpoint_results if r["base_oks"] >= 0]
        dpo_oks_scores = [r["dpo_oks"] for r in checkpoint_results if r["dpo_oks"] >= 0]
        
        results_summary[checkpoint_name] = {
            "base_oks_mean": np.mean(base_oks_scores) if base_oks_scores else -1,
            "dpo_oks_mean": np.mean(dpo_oks_scores) if dpo_oks_scores else -1,
            "valid_samples": len(dpo_oks_scores),
            "total_samples": len(checkpoint_results)
        }
        
        print(f"📈 {checkpoint_name} Results:")
        print(f"   Base OKS: {results_summary[checkpoint_name]['base_oks_mean']:.3f}")
        print(f"   DPO OKS:  {results_summary[checkpoint_name]['dpo_oks_mean']:.3f}")
        print(f"   Valid samples: {results_summary[checkpoint_name]['valid_samples']}/{results_summary[checkpoint_name]['total_samples']}")
    
    # 4. Save summary results
    summary_path = Path(save_folder) / "oks_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n🎉 Complete! Results in {save_folder}")
    print(f"📊 OKS Summary saved to: {summary_path}")
    
    # Print final summary
    print("\n=== FINAL OKS SUMMARY ===")
    for checkpoint_name, results in results_summary.items():
        improvement = results["dpo_oks_mean"] - results["base_oks_mean"] if results["base_oks_mean"] >= 0 and results["dpo_oks_mean"] >= 0 else 0
        print(f"{checkpoint_name}:")
        print(f"  Base: {results['base_oks_mean']:.3f} → DPO: {results['dpo_oks_mean']:.3f} (Δ: {improvement:+.3f})")


if __name__ == "__main__":
    fire.Fire(validate_dpo_alignment)

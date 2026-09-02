import os
import math
import torch
import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Optional, List
import tqdm
import fire
import deepdish as dd
import random
from huggingface_hub import hf_hub_download

from EasyControl.train.src.pipeline import FluxPipeline
from EasyControl.train.src.transformer_flux import FluxTransformer2DModel
from EasyControl.train.src.lora_helper import set_single_lora, set_multi_lora
from posedreamer.label_generation.caption_processor import CaptionProcessor
from smplx.joint_names import JOINT_NAMES


random.seed(42)

# YOLO COCO pose joints in order
YOLO_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
JOINTS_MAPPING = [JOINT_NAMES.index(joint) for joint in YOLO_JOINTS]


def draw_bodypose(canvas: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Draw OpenPose-style skeleton on canvas."""
    stickwidth = 8
    
    # OpenPose skeleton connections (1-indexed)
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5],
        [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16],
        [16, 18],
    ]
    
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]
    
    visible_keypoints = []
    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]
        
        # Check if keypoints are valid (both have positive coordinates)
        if keypoint1[0] <= 0 or keypoint1[1] <= 0 or keypoint2[0] <= 0 or keypoint2[1] <= 0:
            continue
        
        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
        visible_keypoints.append((Y[0], X[0], k1_index))
        visible_keypoints.append((Y[1], X[1], k2_index))
    
    # Draw circles for visible keypoints
    visible_keypoints = set(visible_keypoints)
    for keypoint in visible_keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.circle(canvas, (x, y), stickwidth, colors[keypoint[2] - 1], thickness=-1)
    
    return canvas


def smplx_to_openpose(joints_2d: np.ndarray) -> np.ndarray:
    """
    Convert SMPLX COCO joints to OpenPose format (18 keypoints).
    
    COCO (17 keypoints) -> OpenPose (18 keypoints)
    OpenPose adds a neck point between shoulders.
    """
    # Extract COCO keypoints from SMPLX
    coco_joints = joints_2d[JOINTS_MAPPING]  # Shape: (17, 2 or 3)
    
    # Remap to OpenPose format (18 keypoints)
    # OpenPose order: nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist,
    #                 r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear
    
    def median_point(point_a, point_b):
        """Calculate midpoint between two keypoints."""
        if len(point_a) == 3:
            return np.array([(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2, 1.0])
        return np.array([(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2])
    
    # Create neck point (median of shoulders)
    neck = median_point(coco_joints[5], coco_joints[6])  # left_shoulder, right_shoulder
    
    # Remap to OpenPose format
    openpose_joints = np.array([
        coco_joints[0],   # nose
        neck,             # neck (computed)
        coco_joints[6],   # right_shoulder
        coco_joints[8],   # right_elbow
        coco_joints[10],  # right_wrist
        coco_joints[5],   # left_shoulder
        coco_joints[7],   # left_elbow
        coco_joints[9],   # left_wrist
        coco_joints[12],  # right_hip
        coco_joints[14],  # right_knee
        coco_joints[16],  # right_ankle
        coco_joints[11],  # left_hip
        coco_joints[13],  # left_knee
        coco_joints[15],  # left_ankle
        coco_joints[2],   # right_eye
        coco_joints[1],   # left_eye
        coco_joints[4],   # right_ear
        coco_joints[3],   # left_ear
    ])
    
    return openpose_joints


def create_pose_control_image(joints_2d: np.ndarray, height: int, width: int) -> Image.Image:
    """Create OpenPose-style control image from SMPLX 2D joints."""
    # Convert SMPLX to OpenPose format
    openpose_joints = smplx_to_openpose(joints_2d)
    
    # Create black canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw pose skeleton
    canvas = draw_bodypose(canvas, openpose_joints)
    
    # Convert BGR to RGB
    return Image.fromarray(canvas[:, :, ::-1])


def create_pipeline(
    base_model_path: str,
    control_lora_paths: List[str],
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> FluxPipeline:
    """Create pipeline with control LoRAs and optional checkpoint."""
    pipe = FluxPipeline.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device
    )
    
    # Apply control LoRAs
    if len(control_lora_paths) == 1:
        set_single_lora(transformer, control_lora_paths[0], lora_weights=[1], cond_size=512)
        print(f"✅ Applied single control LoRA: {Path(control_lora_paths[0]).name}")
    else:
        lora_weights = [[1] for _ in control_lora_paths]
        set_multi_lora(transformer, control_lora_paths, lora_weights=lora_weights, cond_size=512)
        print(f"✅ Applied {len(control_lora_paths)} control LoRAs:")
        for path in control_lora_paths:
            print(f"   - {Path(path).name}")
    
    # Apply checkpoint if provided
    if checkpoint_path:
        if checkpoint_path.endswith('.ckpt'):
            from peft import get_peft_model, LoraConfig
            
            peft_config = LoraConfig(
                r=128,
                lora_alpha=128.0,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
            
            peft_model = get_peft_model(transformer, peft_config)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            transformed_checkpoint = {
                (f"base_model.model.{key}" if not key.startswith('base_model.') else key).replace('.weight', '.default.weight') if 'lora_' in key else key: value
                for key, value in checkpoint.items()
            }
            
            peft_model.load_state_dict(transformed_checkpoint, strict=False)
            print(f"✅ Applied checkpoint: {Path(checkpoint_path).name}")
            transformer = peft_model
        else:
            from peft import PeftModel
            transformer = PeftModel.from_pretrained(transformer, checkpoint_path)
            print(f"✅ Applied PEFT adapter: {Path(checkpoint_path).name}")
    
    pipe.transformer = transformer
    pipe.to(device)
    return pipe


def load_smplx_annotation(densepose_file: Path) -> Optional[dict]:
    """Load SMPLX annotation and return as dict."""
    smplx_path = str(densepose_file).replace('densepose-renders', 'smplx-gt-labels').replace('.png', '.h5')
    
    if not os.path.exists(smplx_path):
        return None
    
    try:
        smplx_data = dd.io.load(smplx_path)
        return {key: smplx_data[key] for key in smplx_data.keys()}
    except Exception as e:
        print(f"Error loading SMPLX annotation: {e}")
        return None


def download_pose_lora(cache_dir: str = "./checkpoints") -> str:
    """Download pose control LoRA from HuggingFace."""
    os.makedirs(cache_dir, exist_ok=True)
    
    pose_lora_path = os.path.join(cache_dir, "models", "pose.safetensors")
    if os.path.exists(pose_lora_path):
        print(f"✅ Pose LoRA already downloaded: {pose_lora_path}")
        return pose_lora_path
    
    print("📥 Downloading pose control LoRA from HuggingFace...")
    pose_lora_path = hf_hub_download(
        repo_id="Xiaojiu-Z/EasyControl",
        filename="models/pose.safetensors",
        local_dir=cache_dir
    )
    print(f"✅ Downloaded pose LoRA to: {pose_lora_path}")
    return pose_lora_path


def generate_samples(
    base_model_path: str,
    control_lora_path: str,
    densepose_path: str,
    metadata_path: str,
    save_folder: str,
    checkpoint_path: Optional[str] = None,
    use_pose_control: bool = False,
    num_samples: int = 10,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 25,
    max_sequence_length: int = 512,
):
    """
    Generate samples with optional multi-control (densepose + pose).
    
    Args:
        base_model_path: Path to base FLUX model
        control_lora_path: Path to densepose control LoRA
        densepose_path: Path to densepose control images
        metadata_path: Path to metadata JSON files
        save_folder: Output directory for generated images
        checkpoint_path: Optional checkpoint to apply on top of base LoRA
        use_pose_control: Whether to add pose control (multi-control)
        num_samples: Number of samples to generate
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of diffusion steps
        max_sequence_length: Maximum prompt length
    
    Note: Height and width are automatically extracted from control images
    """
    save_dir = Path(save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = Path(checkpoint_path).stem if checkpoint_path else "base"
    caption_processor = CaptionProcessor()
    
    # Get random densepose files
    densepose_files = list(Path(densepose_path).glob("*.png"))
    densepose_files = random.sample(densepose_files, min(num_samples, len(densepose_files)))
    
    print(f"\n📋 Generation Configuration:")
    print(f"  Base model: {Path(base_model_path).name}")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Pose control: {'ENABLED' if use_pose_control else 'DISABLED'}")
    print(f"  Samples: {len(densepose_files)}")
    print(f"  Output: {save_dir}")
    
    # Download pose LoRA if needed
    pose_lora_path = None
    if use_pose_control:
        pose_lora_path = download_pose_lora(cache_dir=str(save_dir / "checkpoints"))
    
    # Prepare control LoRA paths
    control_lora_paths = [control_lora_path]
    if use_pose_control:
        control_lora_paths.append(pose_lora_path)
    
    # Generate samples WITHOUT pose control
    print(f"\n{'='*60}")
    print("🎨 PART 1: Generating WITHOUT pose control")
    print(f"{'='*60}")
    
    output_dir_no_pose = save_dir / checkpoint_name / "no_pose_control"
    output_dir_no_pose.mkdir(parents=True, exist_ok=True)
    
    pipeline_no_pose = create_pipeline(base_model_path, [control_lora_path], checkpoint_path)
    
    smplx_keys_printed = False
    samples_info = []
    
    for i, densepose_file in enumerate(tqdm.tqdm(densepose_files, desc="Without pose")):
        metadata_file = Path(metadata_path) / f"{densepose_file.stem}.json"
        if not metadata_file.exists():
            print(f"⚠️  Skipping {densepose_file.stem}: No metadata found")
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load control and metadata
        control_image = Image.open(densepose_file).convert('RGB')
        width, height = control_image.size  # Extract dimensions from control image
        
        caption = metadata.get("caption", "A person")
        caption = caption_processor.add_ethnic_labels(caption)
        caption = f"A ultra high DSLR image showing the following content: {caption}. Maximum detail, detailed face, perfect hands, perfection"
        
        # Load SMPLX annotation
        smplx_data = load_smplx_annotation(densepose_file)
        if smplx_data is not None and not smplx_keys_printed:
            print(f"\n📋 SMPLX Annotation Keys for sample {densepose_file.stem}:")
            for key in smplx_data.keys():
                data = smplx_data[key]
                if isinstance(data, np.ndarray):
                    print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")
                else:
                    print(f"  - {key}: type={type(data)}")
            smplx_keys_printed = True
        
        # Generate without pose
        seed = 42 + i
        result = pipeline_no_pose(
            caption,
            spatial_images=[control_image],
            subject_images=[],
            generator=torch.Generator("cpu").manual_seed(seed),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            cond_size=512
        ).images[0]
        
        output_path = output_dir_no_pose / f"sample_{i:03d}.jpg"
        result.save(output_path)
        
        # Save metadata
        sample_info = {
            "densepose_file": str(densepose_file),
            "caption": caption,
            "seed": seed,
            "has_smplx": smplx_data is not None,
            "output_no_pose": str(output_path),
            "width": width,
            "height": height
        }
        
        # Create pose control if needed for comparison
        if use_pose_control and smplx_data is not None:
            joints_2d = smplx_data.get("joints_2d")
            if joints_2d is not None:
                pose_control = create_pose_control_image(joints_2d, height, width)
                pose_path = output_dir_no_pose / f"sample_{i:03d}_pose_control.jpg"
                pose_control.save(pose_path)
                sample_info["pose_control_path"] = str(pose_path)
        
        samples_info.append(sample_info)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Cleanup first pipeline
    del pipeline_no_pose
    torch.cuda.empty_cache()
    
    print(f"\n✅ Generated {len(samples_info)} samples WITHOUT pose control")
    
    # Generate samples WITH pose control if enabled
    if use_pose_control:
        print(f"\n{'='*60}")
        print("🎨 PART 2: Generating WITH pose control (multi-control)")
        print(f"{'='*60}")
        
        output_dir_with_pose = save_dir / checkpoint_name / "with_pose_control"
        output_dir_with_pose.mkdir(parents=True, exist_ok=True)
        
        pipeline_with_pose = create_pipeline(base_model_path, control_lora_paths, checkpoint_path)
        
        for i, sample_info in enumerate(tqdm.tqdm(samples_info, desc="With pose")):
            densepose_file = Path(sample_info["densepose_file"])
            
            # Load control image
            control_image = Image.open(densepose_file).convert('RGB')
            width, height = control_image.size  # Extract dimensions from control image
            
            # Load pose control
            smplx_data = load_smplx_annotation(densepose_file)
            if smplx_data is None or "joints_2d" not in smplx_data:
                print(f"⚠️  Skipping {densepose_file.stem}: No 2D joints available")
                continue
            
            joints_2d = smplx_data["joints_2d"]
            pose_control = create_pose_control_image(joints_2d, height, width)
            
            # Generate with multi-control (densepose + pose)
            result = pipeline_with_pose(
                sample_info["caption"],
                spatial_images=[control_image, pose_control],
                subject_images=[],
                generator=torch.Generator("cpu").manual_seed(sample_info["seed"]),
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                cond_size=512
            ).images[0]
            
            output_path = output_dir_with_pose / f"sample_{i:03d}.jpg"
            result.save(output_path)
            sample_info["output_with_pose"] = str(output_path)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del pipeline_with_pose
        torch.cuda.empty_cache()
        
        print(f"\n✅ Generated {len(samples_info)} samples WITH pose control")
    
    # Save final metadata
    metadata_output_path = save_dir / checkpoint_name / "samples_metadata.json"
    with open(metadata_output_path, 'w') as f:
        json.dump(samples_info, f, indent=2)
    
    # Create comparison visualizations if pose control was used
    if use_pose_control:
        print(f"\n{'='*60}")
        print("🎨 Creating Comparison Visualizations")
        print(f"{'='*60}")
        
        comparison_dir = save_dir / checkpoint_name / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        valid_samples = [s for s in samples_info if "output_with_pose" in s and "output_no_pose" in s]
        
        for idx, sample in enumerate(valid_samples):
            # Load all images
            control_img = Image.open(sample["densepose_file"]).convert('RGB')
            no_pose_img = Image.open(sample["output_no_pose"]).convert('RGB')
            with_pose_img = Image.open(sample["output_with_pose"]).convert('RGB')
            
            # Load pose control if available
            if "pose_control_path" in sample and Path(sample["pose_control_path"]).exists():
                pose_img = Image.open(sample["pose_control_path"]).convert('RGB')
            else:
                pose_img = Image.new('RGB', control_img.size, color='black')
            
            # Get dimensions (use output image size as reference)
            img_width, img_height = no_pose_img.size
            
            # Resize all to match output dimensions
            control_img = control_img.resize((img_width, img_height), Image.LANCZOS)
            pose_img = pose_img.resize((img_width, img_height), Image.LANCZOS)
            with_pose_img = with_pose_img.resize((img_width, img_height), Image.LANCZOS)
            
            # Create horizontal stack: [control | pose | no_pose | with_pose]
            comparison = Image.new('RGB', (img_width * 4, img_height))
            comparison.paste(control_img, (0, 0))
            comparison.paste(pose_img, (img_width, 0))
            comparison.paste(no_pose_img, (img_width * 2, 0))
            comparison.paste(with_pose_img, (img_width * 3, 0))
            
            # Save individual comparison
            output_path = comparison_dir / f"comparison_{idx:03d}.jpg"
            comparison.save(output_path, quality=95)
        
        print(f"✅ Created {len(valid_samples)} comparison images in {comparison_dir}")
    
    print(f"\n{'='*60}")
    print("✅ Generation Complete!")
    print(f"{'='*60}")
    print(f"📁 Output directory: {save_dir / checkpoint_name}")
    print(f"📊 Total samples: {len(samples_info)}")
    if use_pose_control:
        print(f"   - Without pose: {output_dir_no_pose}")
        print(f"   - With pose: {output_dir_with_pose}")
        print(f"   - Comparisons: {comparison_dir}")
    print(f"📋 Metadata: {metadata_output_path}")


if __name__ == "__main__":
    fire.Fire(generate_samples)

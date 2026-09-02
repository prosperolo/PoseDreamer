"""
Functional 3D head pose comparison implementation.

Compares head poses between generated images and rendered SMPLX avatars.
"""

import os
from typing import Optional, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import trimesh
import pyrender
import smplx
from head_detector import HeadDetector


@dataclass
class HeadPose:
    """3D head pose representation."""
    roll: float
    pitch: float
    yaw: float
    confidence: float = 1.0


@dataclass
class HeadPoseComparison:
    """Result of head pose comparison between generated and rendered images."""
    generated_pose: Optional[HeadPose]
    rendered_pose: Optional[HeadPose]
    mae_rpy: float = -1.0  # Mean Absolute Error across Roll, Pitch, Yaw
    has_multiple_heads: bool = False
    match_confidence: float = 0.0


class SimpleSMPLXRenderer:
    """Simple SMPLX renderer for head pose validation."""
    
    def __init__(self, smplx_model_path: str, device: str = 'cpu'):
        self.device = device
        
        # Load SMPL-X model
        self.smplx_model = smplx.create(
            os.path.join(smplx_model_path, 'SMPLX_NEUTRAL.npz'),
            model_type='smplx', 
            gender='neutral', 
            num_betas=10,
            use_face_contour=False,
            flat_hand_mean=False,
            num_expression_coeffs=10,
            ext='npz', 
            use_pca=False
        ).to(device)
        
        self.faces = self.smplx_model.faces
        
    def generate_vertices(self, avatar_params: dict) -> np.ndarray:
        """Generate vertices from SMPL-X parameters."""
        with torch.no_grad():
            # Convert to torch tensors with correct shapes
            betas = torch.from_numpy(avatar_params['betas']).float()
            expression = torch.from_numpy(avatar_params['expression']).float()
            global_orient = torch.from_numpy(avatar_params['global_orient']).float()
            jaw_pose = torch.from_numpy(avatar_params['jaw_pose']).float()
            leye_pose = torch.from_numpy(avatar_params['leye_pose']).float()
            reye_pose = torch.from_numpy(avatar_params['reye_pose']).float()
            
            # Reshape poses
            body_pose = torch.from_numpy(avatar_params['body_pose']).float().reshape(1, 63)
            left_hand_pose = torch.from_numpy(avatar_params['left_hand_pose']).float().reshape(1, 45)
            right_hand_pose = torch.from_numpy(avatar_params['right_hand_pose']).float().reshape(1, 45)
            
            smplx_output = self.smplx_model(
                betas=betas,
                expression=expression,
                global_orient=global_orient,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
            )
            
            vertices = smplx_output.vertices[0].cpu().numpy()
            
        return vertices
        
    def render_overlay(self, vertices: np.ndarray, background_image: np.ndarray, 
                      focal: np.ndarray, princpt: np.ndarray, transl: np.ndarray, 
                      blur_strength: int = 75) -> np.ndarray:
        """Render SMPL-X mesh overlay on blurred background image."""
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        # Apply blur to background
        blurred_bg = cv2.GaussianBlur(background_image, (blur_strength, blur_strength), 0)
        image_rgb = cv2.cvtColor(blurred_bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Apply translation to vertices
        vertices_translated = vertices + transl
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices_translated, self.faces, process=False)
        
        # Material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=(0.7, 0.7, 0.7, 0.8)
        )
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # Create camera with actual parameters
        fx, fy = focal[0], focal[1]
        cx, cy = princpt[0], princpt[1]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        
        # Create scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        
        # Camera pose
        cam_pose = np.array([
            [1.0, 0, 0, 0],
            [0, -1, 0, 0], 
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        scene.add(camera, pose=cam_pose)
        scene.add(mesh_pyrender, 'mesh')
        
        # Add lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=cam_pose)
        
        # Create renderer
        renderer = pyrender.OffscreenRenderer(
            viewport_width=background_image.shape[1],
            viewport_height=background_image.shape[0],
            point_size=1.0
        )
        
        # Render
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        
        # Create mask and blend
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        
        # Blend rendered mesh with background image
        vis_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image_rgb
        
        # Cleanup
        renderer.delete()
        
        return (vis_img * 255).astype(np.uint8)


class HeadPoseDetector:
    """Wrapper for head pose detection using VGG head detector."""
    
    def __init__(self):
        if HeadDetector is None:
            raise ImportError("head_detector package not available. Please install it.")
        self.detector = HeadDetector()
    
    def detect_head_poses(self, image_path: str) -> List[HeadPose]:
        """Detect head poses in an image."""
        prediction = self.detector(image_path)
        
        head_poses = []
        if hasattr(prediction, 'heads') and prediction.heads:
            for head in prediction.heads:
                if hasattr(head, 'head_pose'):
                    pose = HeadPose(
                        roll=head.head_pose.roll,
                        pitch=head.head_pose.pitch,
                        yaw=head.head_pose.yaw,
                        confidence=getattr(head, 'confidence', 1.0)
                    )
                    head_poses.append(pose)
        
        return head_poses


def load_avatar_params(smplx_data: dict) -> dict:
    """Extract SMPL-X parameters from loaded data."""
    return {
        'global_orient': smplx_data['global_orient'],
        'body_pose': smplx_data['body_pose'],
        'left_hand_pose': smplx_data['left_hand_pose'],
        'right_hand_pose': smplx_data['right_hand_pose'],
        'jaw_pose': smplx_data['jaw_pose'],
        'betas': smplx_data['betas'],
        'leye_pose': smplx_data['leye_pose'],
        'reye_pose': smplx_data['reye_pose'],
        'expression': smplx_data.get('expression', np.zeros((1, 10)))
    }


def _angle_diff_deg(a: float, b: float) -> float:
    """Shortest absolute difference between two angles in degrees, accounting for ±180° wrap."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def compute_head_pose_mae(pose1: HeadPose, pose2: HeadPose) -> float:
    """Compute Mean Absolute Error between two head poses (Roll/Pitch/Yaw, in degrees)."""
    roll_diff = _angle_diff_deg(pose1.roll, pose2.roll)
    pitch_diff = _angle_diff_deg(pose1.pitch, pose2.pitch)
    yaw_diff = _angle_diff_deg(pose1.yaw, pose2.yaw)

    return (roll_diff + pitch_diff + yaw_diff) / 3.0


def find_best_head_match(generated_poses: List[HeadPose], rendered_pose: HeadPose) -> Tuple[HeadPose, float]:
    """Find best matching head pose from multiple detected poses."""
    if not generated_poses:
        return None, float('inf')
    
    best_pose = None
    best_mae = float('inf')
    
    for pose in generated_poses:
        mae = compute_head_pose_mae(pose, rendered_pose)
        if mae < best_mae:
            best_mae = mae
            best_pose = pose
    
    return best_pose, best_mae


def compare_head_poses(generated_image_path: str, densepose_path: str, smplx_data: dict,
                      renderer: SimpleSMPLXRenderer, 
                      detector: HeadPoseDetector) -> HeadPoseComparison:
    """
    Compare head poses between generated image and rendered SMPLX avatar.
    
    Args:
        generated_image_path: Path to generated image
        smplx_data: SMPLX parameters and camera data
        renderer: SMPLX renderer instance
        detector: Head pose detector instance
        
    Returns:
        HeadPoseComparison with results
    """
    try:
        # 1. Generate SMPL-X mesh and render avatar
        avatar_params = load_avatar_params(smplx_data)
        vertices = renderer.generate_vertices(avatar_params)
        
        # Load generated image
        generated_image = cv2.imread(generated_image_path)
        densepose = cv2.imread(densepose_path, cv2.IMREAD_GRAYSCALE)
        if densepose is not None and generated_image.shape[:2] != densepose.shape[:2]:
            generated_image = cv2.resize(generated_image, (densepose.shape[1], densepose.shape[0]), 
                                interpolation=cv2.INTER_LANCZOS4)
        if generated_image is None:
            return HeadPoseComparison(None, None)
        
        # Get camera parameters and render overlay
        focal = smplx_data['focal'].reshape(2,)
        princpt = smplx_data['princpt'].reshape(2,)
        transl = smplx_data['transl']
        
        rendered_avatar = renderer.render_overlay(vertices, generated_image, focal, princpt, transl)
        
        # 2. Detect head poses in generated image
        generated_poses = detector.detect_head_poses(generated_image_path)
        
        # 3. Detect head pose in rendered avatar
        rendered_poses = detector.detect_head_poses(rendered_avatar)
        
        # Handle no detections
        if not generated_poses and not rendered_poses:
            return HeadPoseComparison(None, None)
        
        if not rendered_poses:
            # No head detected in rendered avatar
            return HeadPoseComparison(
                generated_pose=generated_poses[0] if generated_poses else None,
                rendered_pose=None
            )
        
        if not generated_poses:
            # No head detected in generated image
            return HeadPoseComparison(
                generated_pose=None,
                rendered_pose=rendered_poses[0]
            )
        
        # 4. Find best match if multiple heads detected in generated image
        rendered_pose = rendered_poses[0]  # Assume single head in rendered avatar
        has_multiple_heads = len(generated_poses) > 1
        
        if has_multiple_heads:
            best_generated_pose, mae_rpy = find_best_head_match(generated_poses, rendered_pose)
            match_confidence = 1.0 / (1.0 + mae_rpy)  # Inverse relationship with error
        else:
            best_generated_pose = generated_poses[0]
            mae_rpy = compute_head_pose_mae(best_generated_pose, rendered_pose)
            match_confidence = 1.0
        
        return HeadPoseComparison(
            generated_pose=best_generated_pose,
            rendered_pose=rendered_pose,
            mae_rpy=mae_rpy,
            has_multiple_heads=has_multiple_heads,
            match_confidence=match_confidence
        )
        
    except Exception as e:
        print(f"Error in head pose comparison: {e}")
        return HeadPoseComparison(None, None)

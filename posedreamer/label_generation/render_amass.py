"""Render AMASS motion-capture frames into SMPL-X control images and .h5
labels, with a random yaw per frame (the AMASS pose source of paper 3.1)."""
import argparse
import logging
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
from os import environ
import json
import cv2
import numpy as np
import deepdish as dd
import smplx
import torch
import random
from torch import multiprocessing
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import copy 
from tqdm import tqdm
from smplx.lbs import batch_rodrigues
from PIL import Image
from multiprocessing import Pool

from posedreamer.rendering.smplx_dataset import SMPLXRenderDataset, collate_batch
from posedreamer.utils.misc import recursive_to, recursive_numpy

from posedreamer.rendering.renderer_pyrd import Renderer
from posedreamer.utils.paths import WEIGHTS_DIR

MAX_TASKS = 16

def get_smpl_vertex_embedding(device=torch.device("cpu")):
        # embed_url = "https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy"
        embed_path = str(WEIGHTS_DIR / "mds_d=256.npy")
        embed_map, _ = np.load(embed_path, allow_pickle=True)  
        embed_map = torch.tensor(embed_map).float()[:, 0]
        embed_map -= embed_map.min()
        embed_map /= embed_map.max()
        return embed_map.cpu().numpy()


def project_smpl_keypoints(pts, t, f, pp):
    pts = pts + t[:, None]
    pts2d = pts[..., :2] / pts[..., 2:]
    return pts2d * f[:, None] + pp[:, None]

dir_path = os.path.dirname(os.path.realpath(__file__))
smplx_model = smplx.create(
        str(WEIGHTS_DIR / "SMPLX_NEUTRAL.npz"),
        model_type="smplx",
        gender="neutral", use_face_contour=False,
        num_betas=10, flat_hand_mean=False,
        num_expression_coeffs=10,
        ext="npz", use_pca=False
    )

smpl_model = smplx.create(
    str(WEIGHTS_DIR),
    gender='neutral',
    ext='pkl',
    num_betas=10,
    use_pca=False) 

vertex2marker = torch.tensor(
    [
        7780, 5032, 8043, 5292, 8080, 5334, 7497, 4761,  # 0 - 7
        7107, 4371, 6996, 4252, 7210, 4474, 6323, 3562,  # 8 - 15
        6150, 3389, 8320, 5626, 7142, 4406, 6651, 3903,  # 16 - 23
        8527, 5780, 6443, 3682, 6050, 3287, 6265, 3504,  # 24 - 31
        7176, 5465, 6844, 4100, 7129, 4393, 8339, 5646,  # 32 - 39
        8635, 8847, 8245, 5525, 6434, 3673, 8376, 5682,  # 40 - 47
        6870, 4126, 8466, 5772, 6124, 3363, 6232, 3471,  # 48 - 55
        5969, 3207, 6405, 3644, 8412, 5601, 5519, 3804
    ])

def crop_smallest_square_nonblack(image, threshold=0):
    arr = np.array(image)
    # mask of any channel > threshold (i.e., not black)
    mask = (arr > threshold).any(axis=2)

    if not mask.any():
        raise ValueError("No non-black pixels found.")

    # bounding box of non-black pixels
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1  # end is exclusive
    x0, x1 = xs.min(), xs.max() + 1

    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    side = side + side*0.1

    # center the square on the bbox
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0_sq = cy - side // 2
    x0_sq = cx - side // 2
    y1_sq = y0_sq + side
    x1_sq = x0_sq + side

    # clamp square to image bounds while keeping the same size
    H, W = mask.shape
    if y0_sq < 0:
        y1_sq -= y0_sq; y0_sq = 0
    if x0_sq < 0:
        x1_sq -= x0_sq; x0_sq = 0
    if y1_sq > H:
        y0_sq -= (y1_sq - H); y1_sq = H
    if x1_sq > W:
        x0_sq -= (x1_sq - W); x1_sq = W

    # final integer bounds
    y0_sq, x0_sq = max(0, int(y0_sq)), max(0, int(x0_sq))
    y1_sq, x1_sq = min(H, int(y1_sq)), min(W, int(x1_sq))

    cropped = image.crop((x0_sq, y0_sq, x1_sq, y1_sq))
    return cropped, (x0_sq, y0_sq, x1_sq, y1_sq), (x0 - x0_sq, y0 - y0_sq, w, h)

faces = torch.from_numpy(smplx_model.faces).to(dtype=torch.int32).contiguous()

def _get_start_end_index(images):
        if "SLURM_ARRAY_TASK_ID" not in environ:
            return 0, len(images)
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
        num_in_one_bucket = len(images) // MAX_TASKS
        return task_id * num_in_one_bucket, min(len(images), (task_id + 1) * num_in_one_bucket)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    subfolders = os.listdir(input_folder)
    all_files = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            subsubfolders = os.listdir(subfolder_path)
            for subsubfolder in subsubfolders:
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                if os.path.isdir(subsubfolder_path):
                    files = os.listdir(subsubfolder_path)
                    for file in files:
                        if file.endswith(".npz"):
                            all_files.append(os.path.join(subsubfolder_path, file))
    print(f"Found {len(all_files)} files in {input_folder}")
    
    all_files = sorted(all_files)
    start, end = _get_start_end_index(all_files)
    print(f"Reading subset from {start} to {end}, total: {len(all_files)}")
    indices = list(range(start, end))

    for i in tqdm(indices):
        try:
            process_file(all_files[i], output_folder)
        except Exception as e:
            print(f"Skipping {all_files[i]} with exception: {e}")

def process_file(file_path, output_folder):
    file_content = np.load(file_path, allow_pickle=True)
    betas = file_content["betas"]
    body_pose = file_content["pose_body"]#[:, 3:66]
    global_orient = file_content["root_orient"]#[:, :3]
    transl = file_content["trans"]
    jaw_pose = file_content["pose_jaw"]
    eye_pose = file_content["pose_eye"]
    hand_pose = file_content["pose_hand"]
    
    idxs = random.sample(range(body_pose.shape[0]), 15)
    params_list = []
    for it, i in enumerate(idxs):
        params = {
            "betas": betas[None, :],
            "body_pose": body_pose[i:i+1],
            "global_orient": global_orient[i:i+1],
            "transl": transl[i:i+1],
            "jaw_pose": jaw_pose[i:i+1],
            "leye_pose": eye_pose[i:i+1][:, :3],
            "reye_pose": eye_pose[i:i+1][:, 3:],
            "left_hand_pose": hand_pose[i][:45].reshape(15, 3),
            "right_hand_pose": hand_pose[i][45:].reshape(15, 3),
        }
        amplitude = 2
        params["betas"] = (np.random.rand(10) - 0.5) * amplitude
        params["betas"] = np.expand_dims(params["betas"], axis=0)

        filename = file_path.split("/")
        last_three = filename[-3:]
        last_three[-1] = last_three[-1].replace(".npz", "") + f"_{it:02d}"
        base_path = "_".join(last_three)

        other_params = {
            "image_w": 1024, 
            "image_h": 1024,
            "focal": np.array([[890.0, 890.0]]),
            "base_path": os.path.basename(file_path).replace(".npz", "") + f"_{it:02d}",
            "princpt": np.array([[512.0, 512.0]])
        }

        params_list.append((params, other_params))
        
    outputs_densepose_path = os.path.join(output_folder, "densepose-renders-amass-v6")
    outputs_smplx_path = os.path.join(output_folder, "smplx-gt-labels-amass-v6")

    for n, params in enumerate(params_list):
        
        try:
            smplx_params, other_params = params
            base_path = other_params["base_path"]
            output_path = os.path.join(outputs_densepose_path, f"{base_path}.png")
            if os.path.exists(output_path):
                continue
            for key in smplx_params:
                smplx_params[key] = torch.tensor(smplx_params[key], dtype=torch.float32, device=torch.device("cpu"))
            smplx_params["global_orient"] = torch.tensor(np.array([[np.pi, 0, 0]])).float()
            R_global = batch_rodrigues(smplx_params["global_orient"].reshape(-1, 3))  # (B, 3, 3)
            t_global = smplx_params["transl"]

            # Augment with a random yaw so each AMASS frame is seen from a
            # random horizontal viewing angle
            random_rot = torch.tensor([[0.0, np.random.randn()*np.pi, 0.0]], dtype=torch.float32).repeat(R_global.shape[0], 1)
            R_random = batch_rodrigues(random_rot)
            R_total = R_random @ R_global

            total_aa = torch.from_numpy(
                R.from_matrix(R_total.detach().cpu().numpy()).as_rotvec()
            ).float()

            smplx_params["global_orient"] = total_aa

            # smplx_params["transl"] = np.zeros_like(smplx_params["transl"])
            smplx_params["transl"] += np.array([0.0, 0.0, 3.0])

            with torch.no_grad():
                smplx_output = smplx_model(
                    betas=smplx_params["betas"],
                    global_orient=smplx_params["global_orient"],
                    body_pose=smplx_params["body_pose"],
                    # transl=smplx_params["transl"]
                    jaw_pose=smplx_params["jaw_pose"],
                    leye_pose=smplx_params["leye_pose"],
                    reye_pose=smplx_params["reye_pose"],
                    left_hand_pose=smplx_params["left_hand_pose"],
                    right_hand_pose=smplx_params["right_hand_pose"],
                )

            # render densepose
            # converst SMPL-X verts to SMPL

            verts = smplx_output.vertices + smplx_params["transl"][:, None]
            
            renderer = Renderer(focal_length=other_params["focal"][0][0], principal_point=other_params["princpt"][0], 
                                img_w=other_params["image_w"], img_h=other_params["image_h"], faces=faces.cpu().numpy(), colormap="smplx")
            densepose_render = renderer.render_front_view(verts.cpu().numpy()) 
            densepose_render = densepose_render[:, :, ::-1]
            renderer.delete()

            densepose_render, bbox, bbox_to_save = crop_smallest_square_nonblack(Image.fromarray(densepose_render))
            densepose_render = np.array(densepose_render)
            x0, y0, x1, y1 = bbox

            max_size = max(x1 - x0, y1 - y0)
            if max_size < 50:
                continue  # skip too small

            px0, py0, w, h = bbox_to_save

            principal_point = other_params["princpt"] - np.array([[x0, y0]])
            img_h = y1 - y0
            img_w = x1 - x0

            smplx_params["focal"] = other_params["focal"]
            smplx_params["princpt"] = principal_point
            smplx_params["bbox"] = bbox_to_save
            smplx_params["img_shape"] = np.array([img_w, img_h])

            # prepare also 3d/2d joints and markers
            joints_3d = smplx_output.joints
            markers_3d = smplx_output.vertices[:, vertex2marker]
            joints_2d = project_smpl_keypoints(
                joints_3d, smplx_params["transl"], other_params["focal"], smplx_params["princpt"])
            markers_2d = project_smpl_keypoints(
                markers_3d, smplx_params["transl"], other_params["focal"], smplx_params["princpt"])

            output = recursive_numpy(smplx_params)
            output["joints_3d"] = joints_3d.cpu().numpy()[0]
            output["markers_3d"] = markers_3d.cpu().numpy()[0]
            output["joints_2d"] = joints_2d.cpu().numpy()[0]
            output["markers_2d"] = markers_2d.cpu().numpy()[0]

            base_path = other_params["base_path"]
            output_path = os.path.join(outputs_smplx_path, f"{base_path}.h5")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dd.io.save(output_path, output)

            output_path = os.path.join(outputs_densepose_path, f"{base_path}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, densepose_render)

        except Exception as e:
            logger.info(f"Process error -- {str(e)}")


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="DenseposeRCNN prediction script")
        parser.add_argument("--input_data_root", type=str, required=True,
                            help="Root directory of input images")
        parser.add_argument("--out_data_root", type=str, required=True,
                            help="Root directory of output files")
        args = parser.parse_args()
        return args

    print("starting logger and rest")
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    args = parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("logger and args ready")

    process_folder(args.input_data_root, args.out_data_root)

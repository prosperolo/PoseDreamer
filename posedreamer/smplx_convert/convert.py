"""
Convert an AMASS SMPL-X npz into SMPL parameters (root_orient, body_pose, betas).
Combines the logic of:
  - transfer_model/write_obj.py  (SMPL-X forward pass -> mesh vertices)
  - transfer_model/__main__.py   (mesh vertices -> SMPL fitting)
  - transfer_model/merge_output.py (aggregate per-frame results)
without writing intermediate mesh files to disk.
"""

import os
import os.path as osp
import argparse
import glob
import traceback

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf

import smplx
from smplx import build_layer
from smplx.joint_names import Body

from transfer_model.config.defaults import conf as default_conf
from transfer_model.transfer_model import run_fitting
from transfer_model.utils import read_deformation_transfer
import deepdish as dd
from posedreamer.utils.paths import REPO_ROOT, WEIGHTS_DIR


# ---------------------------------------------------------------------------
# Step 1  –  SMPL-X forward pass  (adapted from write_obj.py)
# ---------------------------------------------------------------------------

def smplx_forward_all(model_folder, motion_file, model_type="smplx",
                      num_betas=10, gender="neutral"):
    """Load an AMASS npz and run the SMPL-X model to obtain per-frame vertices.

    Returns
    -------
    vertices : Tensor [N, V, 3]
    faces    : Tensor [F, 3]  (int32, shared across frames)
    gender   : str
    """
    motion = dd.io.load(motion_file) #np.load(motion_file, allow_pickle=True)

    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        num_betas=num_betas,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=False,
        use_face_contour=False,
        ext="npz",
    )

    betas = torch.tensor(motion["betas"]).float()
    global_orient = torch.tensor(motion["global_orient"]).float()
    body_pose = torch.tensor(motion["body_pose"]).reshape(1,-1).float()
    jaw_pose = torch.tensor(motion["jaw_pose"]).float()
    left_hand_pose = torch.tensor(motion["left_hand_pose"]).float()
    right_hand_pose = torch.tensor(motion["right_hand_pose"]).float()
    leye_pose = torch.tensor(motion["leye_pose"]).float()
    reye_pose = torch.tensor(motion["reye_pose"]).float()
    transl = torch.tensor(motion["transl"]).float()

    with torch.no_grad():
        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            transl=transl,
        )
    
    vertices = output.vertices.detach().cpu()  # [N, V, 3]
    faces = torch.tensor(model.faces.astype(np.int32))
    return vertices, faces, gender


# ---------------------------------------------------------------------------
# Step 2  –  SMPL fitting  (adapted from __main__.py)
# ---------------------------------------------------------------------------

def fit_smpl(vertices, faces, exp_cfg_path, batch_size=1):
    """Fit SMPL parameters to each frame's SMPL-X mesh vertices.

    Returns
    -------
    all_global_orient : list of Tensors  (rotation matrices from run_fitting)
    all_body_pose     : list of Tensors
    all_betas         : list of Tensors
    all_transl        : list of Tensors  (pelvis / root translation [B, 3])
    """
    cfg = default_conf.copy()
    cfg.merge_with(OmegaConf.load(exp_cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    body_model = build_layer(cfg.body_model.folder, **cfg.body_model)
    body_model = body_model.to(device=device)

    def_matrix = read_deformation_transfer(
        cfg.get("deformation_transfer_path", ""), device=device)

    mask_ids = None
    mask_ids_fname = osp.expandvars(cfg.mask_ids_fname)
    if osp.exists(mask_ids_fname):
        mask_ids = torch.from_numpy(np.load(mask_ids_fname)).to(device=device)

    N = vertices.shape[0]
    all_global_orient, all_body_pose, all_betas, all_transl = [], [], [], []

    for start in tqdm(range(0, N, batch_size), desc="Fitting SMPL"):
        end = min(start + batch_size, N)
        B = end - start
        batch = {
            "vertices": vertices[start:end].to(device),
            "faces": faces.unsqueeze(0).expand(B, -1, -1).to(device),
            "paths": [f"{i:04d}" for i in range(start, end)],
        }
        var_dict = run_fitting(cfg, batch, body_model, def_matrix, mask_ids)

        all_global_orient.append(var_dict["global_orient"].detach().cpu())
        all_body_pose.append(var_dict["body_pose"].detach().cpu())
        all_betas.append(var_dict["betas"].detach().cpu())
        all_transl.append(var_dict["joints"][:, 0, :].detach().cpu())

    return all_global_orient, all_body_pose, all_betas, all_transl


# ---------------------------------------------------------------------------
# Step 3  –  Aggregate & save  (adapted from merge_output.py)
# ---------------------------------------------------------------------------

def rotmats_to_aa(tensors):
    """Stack list of per-frame rotation-matrix tensors and convert to
    axis-angle.  Each tensor has shape [1, J, 3, 3].
    Returns numpy array of shape [N, J*3].
    """
    x = torch.cat(tensors, dim=0).numpy()   # [N, J, 3, 3]
    N = x.shape[0]
    aa = R.from_matrix(x.reshape(-1, 3, 3)).as_rotvec()
    return aa.reshape(N, -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_one_file(motion_file, args):
    """Convert a single AMASS npz file to SMPL format."""
    # 1) SMPL-X forward → mesh vertices
    vertices, faces, gender = smplx_forward_all(
        args.smplx_model_folder, motion_file,
        model_type=args.smplx_model_type)
    N = vertices.shape[0]
    return vertices, faces, gender, N


def convert_batch_files(file_data_list, args):
    """Convert a batch of AMASS npz files together to better utilize GPU.
    
    Args:
        file_data_list: List of tuples (motion_file, vertices, faces, gender, N)
        args: command line arguments
    """
    if not file_data_list:
        return
    
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=""), level="INFO", colorize=True)
    
    # Concatenate all vertices from all files
    all_vertices = []
    file_frame_counts = []
    faces = None
    
    for motion_file, vertices, file_faces, gender, N in file_data_list:
        all_vertices.append(vertices)
        file_frame_counts.append(N)
        if faces is None:
            faces = file_faces
    
    concatenated_vertices = torch.cat(all_vertices, dim=0)
    total_frames = concatenated_vertices.shape[0]
    
    print(f"Processing batch of {len(file_data_list)} files with {total_frames} total frames")
    
    # 2) Fit SMPL to all frames together (batched)
    all_orient, all_pose, all_betas, all_transl = fit_smpl(
        concatenated_vertices, faces, args.exp_cfg,
        batch_size=args.batch_size)
    
    # 3) Aggregate: rot-mats → axis-angle
    root_orient_all = rotmats_to_aa(all_orient)  # [total_frames, 3]
    body_pose_all = rotmats_to_aa(all_pose)       # [total_frames, 69]
    betas_all = torch.cat(all_betas, 0).numpy()   # [total_frames, num_betas]
    transl_all = torch.cat(all_transl, 0).numpy() # [total_frames, 3]
    
    # 4) Split results back to individual files and save
    start_idx = 0
    for idx, (motion_file, _, _, gender, N) in enumerate(file_data_list):
        end_idx = start_idx + N
        
        root_orient = root_orient_all[start_idx:end_idx]
        body_pose = body_pose_all[start_idx:end_idx]
        betas = betas_all[start_idx:end_idx]
        transl = transl_all[start_idx:end_idx]
        
        smpl_dir = os.path.join(args.root_dir, "smpl")
        rel_path = os.path.relpath(motion_file, os.path.join(args.root_dir, "smplx"))
        output_path = os.path.join(smpl_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dd.io.save(output_path, {
            "root_orient": root_orient,
            "body_pose": body_pose,
            "betas": betas,
            "transl": transl,
            "gender": gender,
        })
        print(f"[{idx+1}/{len(file_data_list)}] Saved {N} frames → {output_path}")
        
        start_idx = end_idx

def main():
    parser = argparse.ArgumentParser(
        description="Convert AMASS SMPL-X npz → SMPL params (parallel with locks)")
    parser.add_argument("--smplx-model-folder", default=str(WEIGHTS_DIR), type=str,
                        help="Folder containing SMPL-X model files")
    parser.add_argument("--root-dir", required=True,
                        type=str, help="Root directory containing AMASS npz files")
    parser.add_argument("--smplx-model-type", default="smplx", type=str)
    parser.add_argument("--exp-cfg", default=str(REPO_ROOT / "smplx/config_files/smplx2smpl.yaml"),
                        type=str, help="SMPL fitting config yaml")
    parser.add_argument("--pattern", default="**/*.h5", type=str,
                        help="Glob pattern to find npz files")
    parser.add_argument("--batch-size", default=96, type=int,
                        help="Number of frames to fit in parallel")
    parser.add_argument("--files-per-batch", default=8, type=int,
                        help="Number of files to process together in one batch")
    parser.add_argument("--debug", action="store_true",
                        help="Reprocess files even if output exists")
    args = parser.parse_args()

    # Find all h5 files recursively under the smplx subfolder
    smplx_dir = os.path.join(args.root_dir, "smplx")
    all_npz = glob.glob(os.path.join(smplx_dir, args.pattern), recursive=True)
    motion_files = all_npz
    
    print(f"Found {len(motion_files)} npz files to process")
    
    LOCK_NAME = ".convert.lock"
    
    file_batch = []
    locked_files = []
    
    for motion_file in tqdm(sorted(motion_files)):
        smpl_dir = os.path.join(args.root_dir, "smpl")
        rel_path = os.path.relpath(motion_file, os.path.join(args.root_dir, "smplx"))
        output_path = os.path.join(smpl_dir, rel_path)

        # Skip if output already exists
        if os.path.exists(output_path) and not args.debug:
            continue
        
        # Try to acquire lock
        lock_path = motion_file + LOCK_NAME
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            locked_files.append(lock_path)
        except FileExistsError:
            # Another process is working on this file
            continue
        except OSError as e:
            print(f"Error acquiring lock {lock_path}: {e}")
            continue
        
        # Load file data
        try:
            vertices, faces, gender, N = convert_one_file(motion_file, args)
            file_batch.append((motion_file, vertices, faces, gender, N))
            
            # Process batch when it reaches the desired size
            if len(file_batch) >= args.files_per_batch:
                try:
                    convert_batch_files(file_batch, args)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    traceback.print_exc()
                finally:
                    # Release locks for processed batch
                    for lock in locked_files:
                        if os.path.exists(lock):
                            os.remove(lock)
                    file_batch = []
                    locked_files = []
        except Exception as e:
            print(f"Error loading {motion_file}: {e}")
            traceback.print_exc()
            # Release lock for failed file
            if os.path.exists(lock_path):
                os.remove(lock_path)
            if lock_path in locked_files:
                locked_files.remove(lock_path)
    
    # Process remaining files in the last batch
    if file_batch:
        try:
            convert_batch_files(file_batch, args)
        except Exception as e:
            print(f"Error processing final batch: {e}")
            traceback.print_exc()
        finally:
            # Release remaining locks
            for lock in locked_files:
                if os.path.exists(lock):
                    os.remove(lock)


if __name__ == "__main__":
    main()

"""Render SMPL-X parameter files (.h5) into mesh-to-RGB control images with
the pyrender renderer, and save the matching SMPL-X labels with 2D/3D
joints and markers."""
import argparse
import logging
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import deepdish as dd
import numpy as np
import smplx
import torch
from torch import multiprocessing
from torch.utils.data import DataLoader

from posedreamer.rendering.smplx_dataset import SMPLXRenderDataset, collate_batch
from posedreamer.utils.misc import recursive_to, recursive_numpy

from posedreamer.rendering.renderer_pyrd import Renderer
import time
from posedreamer.utils.paths import WEIGHTS_DIR
print("done with imports")

np.object = object


def parse_args():
    parser = argparse.ArgumentParser(description="DenseposeRCNN prediction script")
    parser.add_argument("--input_data_root", type=str, required=True,
                        help="Root directory of input images")
    parser.add_argument("--out_data_root", type=str, required=True,
                        help="Root directory of output files")
    parser.add_argument("--use_tokenhmr_pose_params", action="store_true",
                        help="Copy body pose params from token-hmr")
    parser.add_argument("--resample_shape_params", action="store_true",
                        help="Randomly resample shape params from the standard normal dist.")
    parser.add_argument("--extreme_aspect_ratio_limit", type=float, default=None,
                        help="If render exceeds this limit pad smaller side to less extreme random aspect ratio")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Processing batch size")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Number of processes working in parallel")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="Logging frequency on the workers")
    args = parser.parse_args()
    return args

print("starting logger and rest")
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
args = parse_args()
dir_path = os.path.dirname(os.path.realpath(__file__))
print("logger and args ready")

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


def process_chunk(process_id: int):
    print("getting renderer and smpl ready")
    num_gpus = torch.cuda.device_count()
    device_id = process_id % num_gpus
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)

    outputs_densepose_path = os.path.join(args.out_data_root, "densepose-renders")
    outputs_smplx_path = os.path.join(args.out_data_root, "smplx-gt-labels")

    smplx_model = smplx.create(
        str(WEIGHTS_DIR / "SMPLX_NEUTRAL.npz"),
        model_type="smplx",
        gender="neutral", use_face_contour=False,
        num_betas=10, flat_hand_mean=False,
        num_expression_coeffs=10,
        ext="npz", use_pca=False
    ).to(device)


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
        ]).to(device)

    faces = smplx_model.faces

    print("loaded everything else")

    start = time.time()
    dataset = SMPLXRenderDataset(
        densepose_root=os.path.join(args.input_data_root, "densepose/"),
        smplerx_root=os.path.join(args.input_data_root, "smplest-x_crops/"),
        tokenhmr_root=os.path.join(args.input_data_root, "camerahmr_crops/"),
        use_tokenhmr_pose_params=args.use_tokenhmr_pose_params,
        resample_shape_params=args.resample_shape_params,
        extreme_aspect_ratio_limit=args.extreme_aspect_ratio_limit,
        process_id=process_id,
        num_processes=args.num_processes
    )
    print(f"Dataset loaded in {time.time() - start:.2f} seconds")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch, shuffle=True)

    logger.info(f"Starting process: {process_id}")

    for n, batch in enumerate(dataloader):
        try:
            try: 
                batch_device = recursive_to(batch, device)
                smplx_params = torch.utils.data.default_collate(batch_device["smplx_params"])
            except Exception as e:
                print("SMPLX params collate error:", e)
                continue
            out_paths = [os.path.join(outputs_densepose_path,  f"{x}.png") for x in batch["base_path"]]
            if all(os.path.exists(x) for x in out_paths):
                print(f"Process {process_id}: skipping {n} / {len(dataloader)}")
                continue

            batch_device = recursive_to(batch, device)
            smplx_params = torch.utils.data.default_collate(batch_device["smplx_params"])
            for key in smplx_params:
                smplx_params[key] = smplx_params[key].squeeze(1).float()
            with torch.no_grad():
                smplx_output = smplx_model(
                    betas=smplx_params["betas"],
                    expression=smplx_params["expression"],
                    global_orient=smplx_params["global_orient"],
                    body_pose=smplx_params["body_pose"],
                    jaw_pose=smplx_params["jaw_pose"],
                    leye_pose=smplx_params["leye_pose"],
                    reye_pose=smplx_params["reye_pose"],
                    left_hand_pose=smplx_params["left_hand_pose"],
                    right_hand_pose=smplx_params["right_hand_pose"],
                )

            # render densepose directly on the SMPL-X mesh (one sample at a time;
            # the SMPL-X colormap is loaded inside Renderer)
            verts = (smplx_output.vertices + smplx_params["transl"][:, None]).cpu().numpy()
            focal = smplx_params["focal"].cpu().numpy()
            princpt = smplx_params["princpt"].cpu().numpy()
            densepose_render = []
            for i in range(len(verts)):
                renderer = Renderer(focal_length=focal[i][0], principal_point=princpt[i],
                                    img_w=int(batch["image_w"][i]), img_h=int(batch["image_h"][i]),
                                    faces=faces, colormap="smplx")
                render = renderer.render_front_view(verts[i:i + 1])
                renderer.delete()
                densepose_render.append(render[:, :, ::-1])

            # prepare also 3d/2d joints and markers
            joints_3d = smplx_output.joints
            markers_3d = smplx_output.vertices[:, vertex2marker]
            joints_2d = project_smpl_keypoints(
                joints_3d, smplx_params["transl"], smplx_params["focal"], smplx_params["princpt"])
            markers_2d = project_smpl_keypoints(
                markers_3d, smplx_params["transl"], smplx_params["focal"], smplx_params["princpt"])

            # save smplx info for training
            for j3d, j2d, m3d, m2d, smplx_p, base_path in zip(
                    joints_3d.cpu().numpy(), joints_2d.cpu().numpy(),
                    markers_3d.cpu().numpy(), markers_2d.cpu().numpy(),
                    batch["smplx_params"], batch["base_path"]):
                output = recursive_numpy(smplx_p)
                output["joints_3d"] = j3d
                output["markers_3d"] = m3d
                output["joints_2d"] = j2d
                output["markers_2d"] = m2d

                output_path = os.path.join(outputs_smplx_path, f"{base_path}.h5")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                dd.io.save(output_path, output)

            # save rendered densepose
            for (H, W, original_dp, dp, base_path) in zip(
                    batch["image_h"], batch["image_w"], batch["original_densepose"],
                    densepose_render, batch["base_path"]):
                dp = dp[:H, :W]

                output_path = os.path.join(outputs_densepose_path, f"{base_path}.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, dp)

            if n % args.log_freq == 0:
                logger.info(f"Process {process_id}: {n} / {len(dataloader)}")

        except Exception as e:
            logger.info(f"Process {process_id}: error at idx={n} -- {str(e)}")


if __name__ == "__main__":

    if args.num_processes > 1:
        multiprocessing.start_processes(
            process_chunk,
            args=(),
            nprocs=args.num_processes,
            join=True
        )
    else:
        process_chunk(0)

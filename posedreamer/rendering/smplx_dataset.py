"""Dataset that pairs DensePose crops with their fitted SMPL-X parameters
(SMPLer-X / TokenHMR) for batch rendering."""
import glob
import os

import cv2
import deepdish as dd
import numpy as np
import torch


class SMPLXRenderDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            densepose_root, smplerx_root, tokenhmr_root,
            use_tokenhmr_pose_params=False, resample_shape_params=False,
            extreme_aspect_ratio_limit=None,
            process_id=0, num_processes=1,
    ):
        self.densepose_root = densepose_root
        self.smplerx_root = smplerx_root
        self.tokenhmr_root = tokenhmr_root
        self.use_tokenhmr_pose_params = use_tokenhmr_pose_params
        self.resample_shape_params = resample_shape_params
        self.extreme_aspect_ratio_limit = extreme_aspect_ratio_limit

        self.tokenhmr_paths = os.listdir(self.tokenhmr_root)
        self.tokenhmr_paths = [os.path.join(self.tokenhmr_root, x) for x in self.tokenhmr_paths]

        densepose_paths = [tokenhmr_path.replace(self.tokenhmr_root, self.densepose_root).replace(".hd5", ".png") for tokenhmr_path in self.tokenhmr_paths]
        self.tokenhmr_paths = [x for i, x in enumerate(self.tokenhmr_paths) if os.path.exists(densepose_paths[i])]

        # Shard the sample list across parallel render processes
        if num_processes > 1:
            self.tokenhmr_paths = self.tokenhmr_paths[process_id::num_processes]

    def __len__(self):
        return len(self.tokenhmr_paths)
    
    def batch_rotmat_to_rodrigues(self, R_batch):
        """
        Convert a batch of rotation matrices (N, 3, 3) to Rodrigues vectors (N, 3)
        using OpenCV.
        """
        rodrigues_batch = []
        for R in R_batch:
            rvec, _ = cv2.Rodrigues(R)
            rodrigues_batch.append(rvec.flatten())
        return np.stack(rodrigues_batch, axis=0)
    
    def get_empty_dict(self):
        return {
            "original_densepose": None,
            "smplx_params": None,
            "image_h": None,
            "image_w": None,
            "base_path": None
        }

    def __getitem__(self, idx):
        tokenhmr_path = self.tokenhmr_paths[idx]
        smplerx_path = tokenhmr_path.replace(self.tokenhmr_root, self.smplerx_root)
        densepose_path = tokenhmr_path.replace(self.tokenhmr_root, self.densepose_root).replace(".hd5", ".png")

        densepose = cv2.imread(densepose_path, 0)
        H, W = densepose.shape
        try:
            smplerx_params = dd.io.load(smplerx_path)
        except Exception as e:
            return self.get_empty_dict()
        if 0 not in smplerx_params.keys():
            return self.get_empty_dict()
        multiple = False
        if 1 in smplerx_params.keys():
            multiple = True
        cam_params = smplerx_params[0]["meta"]
        smplerx_params = smplerx_params[0]["smplx_pred"]
        smplerx_params["focal"] = np.array(cam_params["focal"])
        smplerx_params["princpt"] = np.array(cam_params["princpt"])
        smplerx_params["bbox"] = np.array(cam_params["bbox"])

        if self.use_tokenhmr_pose_params:
            try:
                tokenhmr_params = dd.io.load(tokenhmr_path)
                if tokenhmr_params["body_pose"].shape[0] != 1:
                    multiple = True
                tokenhmr_body_pose = tokenhmr_params["body_pose"][0]
                smplerx_body_pose = smplerx_params["body_pose"]
                tokenhmr_body_pose = self.batch_rotmat_to_rodrigues(tokenhmr_body_pose.cpu().numpy())
                # use all pose params except for hands from Token-HMR
                smplerx_params["body_pose"] = tokenhmr_body_pose[:smplerx_body_pose.shape[0]]
            except Exception as e:
                return self.get_empty_dict()

        if self.resample_shape_params:
            amplitude = 2
            smplerx_params["betas"] = (np.random.rand(smplerx_params["betas"].shape[1]) - 0.5) * amplitude
            smplerx_params["betas"] = np.expand_dims(smplerx_params["betas"], axis=0)

        if self.extreme_aspect_ratio_limit is not None:
            if (H > W) and (H / W > self.extreme_aspect_ratio_limit):
                aspect_ratio = np.random.uniform(1., self.extreme_aspect_ratio_limit)
                H_pad, W_pad = H, round(H / aspect_ratio)
                pad_left = np.random.randint(0, W_pad - W + 1)
                pad_right = W_pad - W - pad_left
                pad_top, pad_bot = 0, 0
            elif (W > H) and (W / H > self.extreme_aspect_ratio_limit):
                aspect_ratio = np.random.uniform(1., self.extreme_aspect_ratio_limit)
                H_pad, W_pad = round(W / aspect_ratio), W
                pad_top = np.random.randint(0, H_pad - H + 1)
                pad_bot = H_pad - H - pad_top
                pad_left, pad_right = 0, 0
            else:
                H_pad, W_pad = H, W
                pad_left, pad_right = 0, 0
                pad_top, pad_bot = 0, 0

            # update principal point
            smplerx_params["princpt"] += np.array([pad_left, pad_top], dtype=np.float32)
            # update bbox
            smplerx_params["bbox"] += np.array([pad_left, pad_top, pad_left, pad_top], dtype=np.float32)
            # save padding
            smplerx_params["pad"] = np.array([pad_left, pad_top, pad_right, pad_bot], dtype=np.float32)
        else:
            H_pad, W_pad = H, W
        smplerx_params = {k: torch.from_numpy(v) for k, v in smplerx_params.items()}

        return {
            "original_densepose": densepose,
            "smplx_params": smplerx_params,
            "image_h": H_pad,
            "image_w": W_pad,
            "base_path": densepose_path.replace(self.densepose_root, "").replace(".png", "")
        }


def collate_batch(batch):
    return dict(
        smplx_params=[x["smplx_params"] for x in batch],
        original_densepose=[x["original_densepose"] for x in batch],
        image_h=[x["image_h"] for x in batch],
        image_w=[x["image_w"] for x in batch],
        base_path=[x["base_path"] for x in batch],
    )

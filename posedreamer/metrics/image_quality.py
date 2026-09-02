"""Inception Score / FID over a folder of generated images (paper 4.4)."""
import random
import numpy as np
import torch
import cv2
from fire import Fire
from glob import glob
from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


def array_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.transpose((2, 0, 1))).unsqueeze(0)


def compute_image_quality(data_pattern: str, real_folder: str, num_samples: int = 1000):
    inception = InceptionScore()
    fid = FrechetInceptionDistance(feature=64)
    files = glob(data_pattern)
    if len(files) > num_samples:
        files = random.sample(files, num_samples)

    real_files = glob(f"{real_folder}/*")
    real_files = real_files[:num_samples]

    for file, real_file in tqdm(zip(files, real_files)):
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        img_tensor = array_to_tensor(image)
        inception.update(img_tensor)
        real_image = cv2.cvtColor(cv2.imread(real_file), cv2.COLOR_BGR2RGB)
        fid.update(array_to_tensor(real_image), real=True)
        fid.update(img_tensor, real=False)
    print(f'Inception score: {inception.compute()}')
    print(f'FID score: {fid.compute()}')


if __name__ == "__main__":
    Fire(compute_image_quality)

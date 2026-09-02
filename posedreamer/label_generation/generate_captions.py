"""Caption the LAION person crops produced by process_laion.py with a VLM."""
import os
import json
from fire import Fire
import numpy as np
import cv2
from posedreamer.label_generation.image_captioning import ImageCaptioner
import random
from tqdm import tqdm


def process_image(image: np.ndarray, metadata_path, captioner):
    generated_metadata = {
        "caption": captioner.generate_caption(image[..., ::-1]),
    }
    with open(metadata_path, "w") as file:
        json.dump(generated_metadata, file)


def dataset(dataset_folder):
    images_folder = os.path.join(dataset_folder, "image_crops")
    metadata_folder = os.path.join(dataset_folder, "metadata")
    os.makedirs(metadata_folder, exist_ok=True)
    image_captioner = ImageCaptioner("gemma-3-4b-it")
    images_list = os.listdir(images_folder)
    random.shuffle(images_list)
    for filename in tqdm(images_list):
        image_path = os.path.join(images_folder, filename)
        metadata_path = os.path.join(metadata_folder, filename.replace(".jpg", ".json"))
        if not os.path.isfile(metadata_path):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            process_image(image, metadata_path, image_captioner)
        else:
            print(f"Skipping {filename}, already generated")


if __name__ == "__main__":
    Fire(dataset)

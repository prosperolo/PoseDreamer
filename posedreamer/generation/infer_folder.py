"""Batch inference over a folder of control renders with FLUX + the spatial
control LoRA; saves side-by-side comparison strips."""
import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora
from tqdm import tqdm
import random
import numpy as np
import os
import cv2
import json


def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()


def overlay_images(densepose, rgb):
    alpha = 0.5
    return cv2.addWeighted(rgb, 1 - alpha, densepose, alpha, 0)


def main(args):
    # Initialize model
    device = "cuda"
    base_path = "black-forest-labs/FLUX.1-dev"  # HF model id, or path to a local FLUX.1-dev checkout
    pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
    transformer = FluxTransformer2DModel.from_pretrained(
        base_path, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16, 
        device=device
    )
    pipe.transformer = transformer
    pipe.to(device)

    set_single_lora(pipe.transformer, args.weights_path, lora_weights=[1], cond_size=512)

    
    cse_folder = args.cse_folder_path
    images_folder = args.images_folder_path

    control_folder = os.path.join(cse_folder, "densepose-renders")
    crops_folder = os.path.join(images_folder, "image_crops")
    metadata_folder = os.path.join(images_folder, "metadata")
    save_folder = os.path.join(cse_folder, "generated-images")

    os.makedirs(save_folder, exist_ok=True)

    control_images = os.listdir(control_folder)
    for i in tqdm(control_images):
        image_path = os.path.join(crops_folder, i.replace(".png", ".jpg"))
        control_path = os.path.join(control_folder, i)
        metadata_path = os.path.join(metadata_folder, i.replace(".png", ".json"))
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        prompt = metadata["caption"]

        control_image = Image.open(control_path)
        control_image = control_image.resize((args.width, args.height))

        result = pipe(
            prompt,
            height=args.height,
            width=args.width,
            guidance_scale=3.5,
            num_inference_steps=25,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            spatial_images=[control_image],
            subject_images=[],
            cond_size=512,
        ).images[0]

        # Clear cache after generation
        clear_cache(pipe.transformer)

        image = Image.open(image_path).convert('RGB')
        image = image.resize((args.width, args.height))
        image = np.array(image)
        control_image = np.array(control_image)
        result = np.array(result)
        overlay = overlay_images(control_image, result)
        gt_overlay = overlay_images(control_image, image)

        result = np.concatenate([image, gt_overlay, control_image, result, overlay], axis=1)
        result = Image.fromarray(result)
        result.save(os.path.join(save_folder, i.replace(".png", ".jpg")))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with FLUX.1-dev model")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--cse_folder_path", type=str)
    parser.add_argument("--images_folder_path", type=str)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_images_per_prompt", type=int, default=20)
    args = parser.parse_args()

    main(args)

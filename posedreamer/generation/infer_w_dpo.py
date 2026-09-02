"""Large-scale dataset generation with the control LoRA + DPO LoRA stacked
(optionally + realism LoRA) — the final generation step of the pipeline."""
import os 
from os import environ
import torch
from PIL import Image
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora
from posedreamer.label_generation.caption_processor import CaptionProcessor
import tqdm
import random
import numpy as np
import os
import cv2
import json
from fire import Fire
from pathlib import Path


MAX_TASKS = int(os.environ.get("MAX_TASKS", 16))





def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()


def overlay_images(densepose, rgb):
    alpha = 0.5
    return cv2.addWeighted(rgb, 1 - alpha, densepose, alpha, 0)


class DataGenerator:
    def __init__(self, weights_path, dpo_checkpoint, use_realism_lora=True):
        self.pipeline = self.get_pipeline(weights_path, dpo_checkpoint, use_realism_lora)
        self.caption_processor = CaptionProcessor()

    def get_pipeline(self, control_lora_path, dpo_checkpoint, use_realism_lora=True):
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
        # Apply the spatial (mesh-to-RGB) control LoRA
        set_single_lora(transformer, control_lora_path, lora_weights=[1], cond_size=512)
        print(f"✅ Applied control LoRA: {Path(control_lora_path).name}")

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
        print(f"✅ Applied DPO LoRA checkpoint: {dpo_checkpoint}")
        
        transformer = peft_model
        pipe.transformer = transformer
        pipe.to(device)
        
        if use_realism_lora:
            pipe.load_lora_weights("XLabs-AI/flux-RealismLora", adapter_name="realism_lora")
        return pipe

    def _get_start_end_index(self, images):
        if "SLURM_ARRAY_TASK_ID" not in environ:
            return 0, len(images)
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
        num_in_one_bucket = len(images) // MAX_TASKS
        return task_id * num_in_one_bucket, min(len(images), (task_id + 1) * num_in_one_bucket)

    def read_annotations(self, annotation_files: str):
        annotations = []
        for anno_file in annotation_files:
            assert os.path.exists(anno_file), f"Annotation file {anno_file} does not exist"
            with open(anno_file, "r") as file:
                annotations += json.load(file)["caption"]
        return annotations

    def generate(self, densepose_path: str, metadata_path: str, save_dir: str, save_dict_dir: str, height=1024, width=1024, seed=42):
        print(f"Generating images from {densepose_path} to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dict_dir, exist_ok=True)
        filenames = os.listdir(densepose_path)
        filenames = sorted(filenames)[::-1]
        densepose_filenames = [os.path.join(densepose_path, filename) for filename in filenames]
        metadata_filenames = [filename.replace(densepose_path, metadata_path).replace(".png", ".json") for filename in densepose_filenames]
        print(f"Found {len(densepose_filenames)} annotations")
        start, end = self._get_start_end_index(densepose_filenames)
        print(f"Reading subset from {start} to {end}, total: {len(densepose_filenames)}")

        indices = list(range(start, end))
        random.shuffle(indices)
        for index in tqdm.tqdm(indices):
            try:
                annotation = metadata_filenames[index]
                filepath = densepose_filenames[index]
                image_path = os.path.join(filepath)
                with open(annotation, "r") as file:
                    annotation = json.load(file)["caption"]
                control_image = Image.open(image_path).convert('RGB')
                caption = self.caption_processor.add_ethnic_labels(annotation)
                caption = f"A ultra high DSLR image showing the following content: {caption}. Maximum detail, detailed face, perfect hands, perfection"
                guidance_scale=3.5
                num_inference_steps=25
                max_sequence_length=512
                cond_size=512
                for i in range(1):
                    filename_no_ext = filenames[index].split(".png")[0]
                    random_seed = torch.randint(0, 1000000, (1,)).item()
                    out_path = os.path.join(save_dir, filename_no_ext + f"_{i}.jpg")
                    out_dict_path = os.path.join(save_dict_dir, filename_no_ext + f"_{i}.json")
                    save_dict = {
                        "condition_path": filepath, 
                        "caption": caption,
                        "seed": 42,
                        "generated_image_path": out_path,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_inference_steps,
                        "max_sequence_length": max_sequence_length,
                        "cond_size": cond_size,
                        "height": height,
                        "width": width
                    }
                    if os.path.exists(out_path):
                        continue

                    result = self.pipeline(
                        caption,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=max_sequence_length,
                        generator=torch.Generator("cpu").manual_seed(random_seed),
                        spatial_images=[control_image],
                        subject_images=[],
                        cond_size=cond_size,
                    ).images[0]
                    # Clear the control-token cache after each generation
                    clear_cache(self.pipeline.transformer)
                    result.save(out_path)
                    with open(out_dict_path, "w") as f:
                        json.dump(save_dict, f)
            except Exception as e:
                print(f"Error: {e}")
                pass


def dataset(weights_path: str, dpo_checkpoint: str, densepose_path: str, metadata_path: str,
            save_dir: str, save_dict_dir: str, height=1024, width=1024, use_realism_lora=True):
    os.makedirs(save_dir, exist_ok=True)
    generator = DataGenerator(weights_path, dpo_checkpoint, use_realism_lora=use_realism_lora)
    generator.generate(densepose_path, metadata_path, save_dir, save_dict_dir, height=height, width=width)

if __name__ == "__main__":
    Fire(dataset)
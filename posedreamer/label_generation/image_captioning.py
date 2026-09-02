"""VLM wrappers (BLIP / BLIP-2 / Gemma-3) used to caption images."""
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration, Gemma3ForConditionalGeneration
from PIL import Image

CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',    # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large',  # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',               # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',       # 15.77GB
    'git-large-coco': 'microsoft/git-large-coco',            # 1.58GB
    'fuse-cap': 'noamrot/FuseCap',                           # 990MB
    'gemma-3-4b-it': 'google/gemma-3-4b-it'
}


class ImageCaptioner:
    def __init__(self, model: str = "blip2-2.7b", device: str = "cuda"):
        self.model = model
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        model_path = CAPTION_MODELS[model]
        if model.startswith('git-'):
            caption_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        elif model.startswith('blip2-'):
            caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        elif model.startswith('gemma-'):
            caption_model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        else:
            caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        self.caption_processor = AutoProcessor.from_pretrained(model_path)
        self.caption_model = caption_model.eval().to(device)

    def generate_caption(self, image: np.ndarray) -> str:
        if self.model.startswith('gemma-'):
            return self.generate_caption_gemma(image)
        else:
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            if self.model.startswith('blip2-'):
                inputs = inputs.to(self.dtype)
            tokens = self.caption_model.generate(**inputs, max_new_tokens=100)
            return self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    
    def generate_caption_gemma(self, image: np.ndarray) -> str:
        text = f"Generate a detailed description for the image. Please output just the description and no additional information, and be concise."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image", "image": Image.fromarray(image)}
                ]
            }
        ]
        inputs = self.caption_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        tokens = self.caption_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        tokens = tokens[0][input_len:]
        return self.caption_processor.decode(tokens, skip_special_tokens=True).strip()
    
    def pil_resize_image(self, image):
        width, height = image.size
        new_height = 256
        new_width = int(width * new_height / height)
        return image.resize((new_width, new_height))



import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field
import os

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.training_utils import compute_density_for_timestep_sampling
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from EasyControl.train.src.transformer_flux import FluxTransformer2DModel
from EasyControl.train.src.pipeline import FluxPipeline, resize_position_encoding
from EasyControl.train.src.lora_helper import set_single_lora

from posedreamer.generation.alignment.loss import DPOFlowMatchingLoss, flow_matching_target
from posedreamer.generation.alignment.checkpoint import save_dpo_lora_safetensors

logger = logging.getLogger(__name__)


@dataclass
class LoraTrainConfig:
    rank: int = 128
    alpha: float = 128.0
    target_modules: List[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: Optional[str] = None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        class_name = CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        class_name = T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    text_encoder = class_name.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    return text_encoder


class DPOEasyControlTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for DPO finetuning of EasyControl.
    Uses reference model approach with MSE error comparison.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        control_lora_path: str,
        learning_rate: float = 1e-5,
        beta: float = 500,
        cond_size: int = 512,
        gradient_checkpointing: bool = True,
        **kwargs
    ): 
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.control_lora_path = control_lora_path
        self.lora_config = LoraTrainConfig()
        self.gradient_checkpointing = gradient_checkpointing
        self.save_hyperparameters()
        
        self.load_models()
        self.add_dpo_lora()
        
        # Loss function
        self.dpo_loss = DPOFlowMatchingLoss(beta=beta)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.cond_size = cond_size
        self.vae_scale_factor = 16
        self.height_cond = 2 * (cond_size // self.vae_scale_factor)
        self.width_cond = 2 * (cond_size // self.vae_scale_factor)
        self.offset = 64
        
        self.weighting_scheme = "logit_normal"
        self.logit_mean = 0.0
        self.logit_std = 1.0
        self.mode_scale = 1.29
        self.guidance_scale = 3.5
    
    def add_dpo_lora(self):
        """Add DPO LoRA to the transformer."""
        # Enable gradient checkpointing for memory efficiency
        if self.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing for memory optimization")
        
        peft_config = LoraConfig(
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.alpha,
            target_modules=self.lora_config.target_modules,
        )
        self.peft_model = get_peft_model(self.transformer, peft_config)
        self.transformer = self.peft_model
        self.peft_model.train()
    
    def load_models(self):
        """Load all required model components."""
        # Load tokenizers
        self.tokenizer_one = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")
        self.tokenizer_two = T5TokenizerFast.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer_2")
        
        # Load models
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")
        self.transformer = FluxTransformer2DModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="transformer")
        
        set_single_lora(self.transformer, self.control_lora_path, lora_weights=[1], cond_size=512)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        
        self.text_encoder_one = import_model_class_from_model_name_or_path(self.pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder_two = import_model_class_from_model_name_or_path(self.pretrained_model_name_or_path, subfolder="text_encoder_2")
        
        # Freeze non-trainable components
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.transformer.requires_grad_(False)
    
    def encode_text(self, captions: List[str]) -> tuple:
        """Encode text captions using both text encoders."""
        text_inputs = self.tokenizer_one(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs.input_ids

        text_inputs = self.tokenizer_two(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs.input_ids
        
        # Encode with both text encoders (reimplemented encode_token_ids)
        batch_size = text_input_ids_1.shape[0]
        device = self.device
        
        # CLIP encoding
        clip_outputs = self.text_encoder_one(text_input_ids_1.to(device), output_hidden_states=False)
        pooled_prompt_embeds = clip_outputs.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.text_encoder_one.dtype, device=device)
        
        # T5 encoding
        t5_outputs = self.text_encoder_two(text_input_ids_2.to(device))
        prompt_embeds = t5_outputs[0]  # last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_two.dtype, device=device)
        
        # Create text_ids (position encodings for text)
        _, seq_len, _ = prompt_embeds.shape
        text_ids = torch.zeros(seq_len, 3).to(device=device, dtype=prompt_embeds.dtype)
        
        return prompt_embeds, pooled_prompt_embeds, text_ids
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with VAE."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents
    
    def sample_noise_and_timesteps(self, images: torch.Tensor, batch_size: int) -> tuple:
        """Sample noise and timesteps ONCE for all DPO forward passes."""
        # Calculate VAE latent dimensions
        vae_channels = 16
        latent_height = 2 * (images.shape[-2] // self.vae_scale_factor)
        latent_width = 2 * (images.shape[-1] // self.vae_scale_factor)
        
        # Sample noise with correct VAE latent shape
        noise = torch.randn(
            batch_size, vae_channels, latent_height, latent_width,
            device=self.device, dtype=torch.bfloat16
        )
        
        # Sample timesteps using EasyControl's logic
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=self.device)
        
        return noise, timesteps
    
    def forward_model(
        self, 
        images: torch.Tensor, 
        conditions: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> tuple:
        """
        Forward pass with PRE-ENCODED text and PRE-SAMPLED noise/timesteps.
        Optimized for DPO: encode text once, use for both winner/loser.
        """
        batch_size = images.shape[0]
        
        # Encode images and conditions
        model_input = self.encode_images(images)
        cond_input = self.encode_images(conditions)
        
        # Prepare position encodings
        height_ = 2 * (images.shape[-2] // self.vae_scale_factor)
        width_ = 2 * (images.shape[-1] // self.vae_scale_factor)
        
        latent_image_ids, cond_latent_image_ids = resize_position_encoding(
            batch_size, height_, width_, self.height_cond, self.width_cond,
            self.device, model_input.dtype
        )
        
        # Add noise according to flow matching with PROVIDED noise/timesteps
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        
        # Pack latents
        packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=batch_size,
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )
        
        packed_cond_input = FluxPipeline._pack_latents(
            cond_input,
            batch_size=batch_size,
            num_channels_latents=cond_input.shape[1],
            height=cond_input.shape[2],
            width=cond_input.shape[3],
        )
        
        # Prepare final latent IDs
        latent_image_ids = torch.concat([latent_image_ids, cond_latent_image_ids], dim=-2)
        
        # Handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([self.guidance_scale], device=self.device)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None
        
        # Forward through transformer
        model_pred = self.transformer(
            hidden_states=packed_noisy_model_input,
            cond_hidden_states=packed_cond_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        
        # Unpack latents
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=images.shape[-2],
            width=images.shape[-1],
            vae_scale_factor=self.vae_scale_factor,
        )
        
        target = flow_matching_target(noise, model_input)
        return model_pred, target
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """Get sigmas for timesteps (from EasyControl)."""
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def compute_dpo_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute DPO loss using reference model approach."""
        # Extract data
        x_w, x_l = batch["x_w"], batch["x_l"]
        cond = batch["cond"]
        captions = batch["captions"]
        
        noise, timesteps = self.sample_noise_and_timesteps(x_w, x_w.shape[0])
        
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_text(captions)
        
        assert self.peft_model.base_model.peft_config, "PEFT adapters not properly initialized"
        
        self.peft_model.enable_adapter_layers()
        model_pred_w, target_w = self.forward_model(
            x_w, cond, prompt_embeds, pooled_prompt_embeds, text_ids, noise, timesteps
        )
        model_pred_l, target_l = self.forward_model(
            x_l, cond, prompt_embeds, pooled_prompt_embeds, text_ids, noise, timesteps
        )
        
        self.peft_model.disable_adapter_layers()
        with torch.no_grad():
            ref_pred_w, _ = self.forward_model(
                x_w, cond, prompt_embeds, pooled_prompt_embeds, text_ids, noise, timesteps
            )
            ref_pred_l, _ = self.forward_model(
                x_l, cond, prompt_embeds, pooled_prompt_embeds, text_ids, noise, timesteps
            )
        self.peft_model.enable_adapter_layers()
        
        if torch.equal(model_pred_w, ref_pred_w):
            logger.warning("⚠️ Model and reference predictions are identical! Check adapter setup.")
        
        loss_dict = self.dpo_loss(
            model_pred_winner=model_pred_w,
            model_pred_loser=model_pred_l,
            ref_pred_winner=ref_pred_w,
            ref_pred_loser=ref_pred_l,
            target_winner=target_w,
            target_loser=target_l
        )
        
        # Add debug metrics
        loss_dict["noise_std"] = noise.std().item()
        loss_dict["timestep_mean"] = timesteps.float().mean().item()
        
        return loss_dict

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with memory optimizations."""
        # Clear cache at the beginning of each step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        results = self.compute_dpo_loss(batch)
        
        # Log metrics
        self.log("train_loss", results["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", results["accuracy"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_win_diff", results["win_diff"], on_step=True, on_epoch=True)
        self.log("train_lose_diff", results["lose_diff"], on_step=True, on_epoch=True)
        self.log("train_model_win_err", results["model_win_err"], on_step=True, on_epoch=True)
        self.log("train_model_lose_err", results["model_lose_err"], on_step=True, on_epoch=True)
        self.log("noise_std", results["noise_std"], on_step=True, on_epoch=True)
        self.log("timestep_mean", results["timestep_mean"], on_step=True, on_epoch=True)
        self.log("winner_metrics", batch["winner_metrics"].mean(), on_step=True, on_epoch=True)
        self.log("loser_metrics", batch["loser_metrics"].mean(), on_step=True, on_epoch=True)
        
        # Clear cache after computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results["loss"]

    def on_save_checkpoint(self, checkpoint):
        """Save DPO LoRA checkpoint in diffusers-compatible format."""
        checkpoint.clear()
        
        # Save LoRA as safetensors for inference
        save_dpo_lora_safetensors(
            peft_model=self.peft_model,
            checkpoint_dir=os.path.join(self.trainer.default_root_dir, "safetensor_checkpoints"), 
            global_step=self.global_step,
        )
        lora_state_dict = get_peft_model_state_dict(self.peft_model)
        checkpoint.update(lora_state_dict)
        
    def configure_optimizers(self):
        """Configure optimizer."""
        # Only optimize LoRA parameters
        params_to_optimize = [p for p in self.transformer.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.learning_rate,
        )
        
        return optimizer

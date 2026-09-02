from pathlib import Path
import logging
from safetensors.torch import save_file
from peft import get_peft_model_state_dict

logger = logging.getLogger(__name__)


def save_dpo_lora_safetensors(peft_model, checkpoint_dir: str, global_step: int) -> str:
    """
    Save DPO LoRA weights as safetensors for diffusers inference.
    
    Args:
        peft_model: PEFT model with LoRA adapters
        checkpoint_dir: Base directory to save checkpoints
        global_step: Current training step
    
    Returns:
        Path to saved safetensors file
    """
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir) / f"dpo_lora_step_{global_step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Get PEFT state dict
    lora_state_dict = get_peft_model_state_dict(peft_model)
    
    # Convert PEFT keys to diffusers format
    diffusers_state_dict = {}
    for key, tensor in lora_state_dict.items():
        # Convert: base_model.model.transformer_blocks.X... -> transformer.transformer_blocks.X...
        new_key = key
        if new_key.startswith("base_model.model."):
            new_key = new_key[17:]  # Remove "base_model.model."
        if not new_key.startswith("transformer."):
            new_key = f"transformer.{new_key}"
        
        # Convert: .default.weight -> .weight
        new_key = new_key.replace(".default.weight", ".weight")
        new_key = new_key.replace(".default.bias", ".bias")
        
        diffusers_state_dict[new_key] = tensor
    
    # Save as safetensors
    safetensors_file = checkpoint_path / "pytorch_lora_weights.safetensors"
    save_file(diffusers_state_dict, str(safetensors_file))
    
    logger.info(f"Saved DPO LoRA checkpoint: {safetensors_file}")
    
    return str(safetensors_file)

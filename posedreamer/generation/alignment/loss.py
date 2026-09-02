import torch
import torch.nn.functional as F
from typing import Dict 


def dpo_loss_with_reference(
    model_pred_winner: torch.Tensor,
    model_pred_loser: torch.Tensor,
    ref_pred_winner: torch.Tensor,
    ref_pred_loser: torch.Tensor,
    target_winner: torch.Tensor,
    target_loser: torch.Tensor,
    beta: float = 500.0
) -> Dict[str, torch.Tensor]:
    """
    Compute DPO loss using reference model comparison approach.
    
    Args:
        model_pred_winner: Model predictions for winner samples [B, C, H, W]
        model_pred_loser: Model predictions for loser samples [B, C, H, W]
        ref_pred_winner: Reference model predictions for winner samples [B, C, H, W]
        ref_pred_loser: Reference model predictions for loser samples [B, C, H, W]
        target_winner: Ground truth targets for winner samples [B, C, H, W]
        target_loser: Ground truth targets for loser samples [B, C, H, W]
        beta: Regularization parameter
    
    Returns:
        Dictionary containing loss and metrics
    """
    model_win_err = (model_pred_winner - target_winner).pow(2).mean(dim=[1, 2, 3])
    model_lose_err = (model_pred_loser - target_loser).pow(2).mean(dim=[1, 2, 3])
    
    ref_win_err = (ref_pred_winner - target_winner).pow(2).mean(dim=[1, 2, 3])
    ref_lose_err = (ref_pred_loser - target_loser).pow(2).mean(dim=[1, 2, 3])
    
    win_diff = model_win_err - ref_win_err
    lose_diff = model_lose_err - ref_lose_err
    
    inside_term = -0.5 * beta * (win_diff - lose_diff)
    loss = -F.logsigmoid(inside_term).mean()
    
    accuracy = (inside_term > 0).float().mean()
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "win_diff": win_diff.mean(),
        "lose_diff": lose_diff.mean(),
        "model_win_err": model_win_err.mean(),
        "model_lose_err": model_lose_err.mean(),
        "ref_win_err": ref_win_err.mean(),
        "ref_lose_err": ref_lose_err.mean(),
        "inside_term": inside_term.mean()
    }
    
    return metrics


def flow_matching_target(noise: torch.Tensor, clean_image: torch.Tensor) -> torch.Tensor:
    """
    Compute flow matching target (velocity).
    
    Args:
        noise: Random noise tensor
        clean_image: Clean image tensor
        
    Returns:
        Flow matching target (velocity)
    """
    return noise - clean_image


class DPOFlowMatchingLoss:
    """
    DPO loss for flow matching models with reference model comparison.
    """
    
    def __init__(self, beta: float = 500.0):
        self.beta = beta
    
    def __call__(
        self,
        model_pred_winner: torch.Tensor,
        model_pred_loser: torch.Tensor,
        ref_pred_winner: torch.Tensor,
        ref_pred_loser: torch.Tensor,
        target_winner: torch.Tensor,
        target_loser: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute DPO loss."""
        return dpo_loss_with_reference(
            model_pred_winner=model_pred_winner,
            model_pred_loser=model_pred_loser,
            ref_pred_winner=ref_pred_winner,
            ref_pred_loser=ref_pred_loser,
            target_winner=target_winner,
            target_loser=target_loser,
            beta=self.beta
        )

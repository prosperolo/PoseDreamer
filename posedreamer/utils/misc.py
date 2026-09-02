import torch


def recursive_to(x, target):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x: Batch of data.
        target: Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x

def recursive_numpy(x):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x: Batch of data.
        target: Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_numpy(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, list):
        return [recursive_numpy(i) for i in x]
    else:
        return x
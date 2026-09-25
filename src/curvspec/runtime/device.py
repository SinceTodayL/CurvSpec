import random

import numpy as np
import torch


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = not cuda_deterministic


def to_device(data, device):
    if isinstance(data, list):
        return [to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    if isinstance(data, torch.Tensor):
        return data.contiguous().to(device, non_blocking=device.type == "cuda")
    return data

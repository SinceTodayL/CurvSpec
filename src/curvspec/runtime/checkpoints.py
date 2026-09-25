from collections import OrderedDict

import torch


def _strip_module_prefix(state_dict):
    if not state_dict or not all(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict((key[7:], value) for key, value in state_dict.items())


def load_model_state(model, state_dict):
    return model.load_state_dict(_strip_module_prefix(state_dict))


def normalize_ema_state(state_dict):
    return _strip_module_prefix(state_dict) if state_dict is not None else None


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_checkpoint(model, optimizer, config, path, epoch, model_val, ema_shadow=None, criterion_state=None):
    checkpoint = {
        "config": config,
        "epoch": epoch,
        "model_val": [float(value) for value in model_val],
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if ema_shadow is not None:
        checkpoint["ema_shadow"] = ema_shadow
    if criterion_state is not None:
        checkpoint["criterion_state"] = criterion_state
    torch.save(checkpoint, path)

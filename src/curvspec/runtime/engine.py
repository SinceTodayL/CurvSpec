import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from curvspec.corpus import build_evaluation_loaders, build_training_loaders
from curvspec.matching import build_objective
from curvspec.protocol import build_evaluator
from curvspec.representation import build_model
from curvspec.runtime.checkpoints import (
    load_checkpoint,
    load_model_state,
    normalize_ema_state,
    save_checkpoint,
)
from curvspec.runtime.device import resolve_device, set_seed, to_device
from curvspec.runtime.logging import create_logger
from curvspec.runtime.meters import AverageMeter
from curvspec.runtime.optimization import EMA
from curvspec.runtime.optimizer_factory import build_optimizer


def extract_curvatures(model):
    curvatures = {}
    with torch.no_grad():
        model.curv.data = torch.clamp(model.curv.data, **model._curv_minmax)
        curvatures["global_curv"] = model.curv.exp().item()
        for index, manifold in enumerate(model.clip_manifolds):
            manifold.k.data = torch.clamp(
                manifold.k.data,
                min=model.config.curv_init / 10,
                max=model.config.curv_init * 10,
            )
            curvatures[f"clip_curv_{index + 1}"] = manifold.k.item()
        for index, manifold in enumerate(model.frame_manifolds):
            manifold.k.data = torch.clamp(
                manifold.k.data,
                min=model.config.curv_init / 10,
                max=model.config.curv_init * 10,
            )
            curvatures[f"frame_curv_{index + 1}"] = manifold.k.item()
    return curvatures


def plot_curvature_evolution(curvature_history, save_path):
    if not curvature_history:
        return
    figure = plt.figure(figsize=(16, 10))
    epochs = list(range(len(curvature_history)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(curvature_history[0])))
    for key, color in zip(curvature_history[0], colors):
        values = [epoch_curvatures[key] for epoch_curvatures in curvature_history]
        if key == "global_curv":
            plt.plot(epochs, values, label=key, linewidth=3, color="black", linestyle="--")
        elif "clip" in key:
            plt.plot(epochs, values, label=key, linewidth=2.5, color=color)
        else:
            plt.plot(epochs, values, label=key, linewidth=2.5, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Curvature Value")
    plt.title("Multi-Curvature Learning Evolution During Training")
    plt.legend(loc="best", ncol=2)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _loss_meters(config):
    if not config.get("log_loss_components", False):
        return None
    return {
        "loss_sim": AverageMeter(),
        "loss_div": AverageMeter(),
        "loss_hyp": AverageMeter(),
        "loss_curv_reg": AverageMeter(),
        "loss_cov": AverageMeter(),
        "loss_exclusive": AverageMeter(),
    }


def train_one_epoch(epoch, train_loader, model, criterion, config, optimizer, device, ema=None, ema_step=0):
    criterion.cfg["use_hard_negative"] = epoch >= config["hard_negative_start_epoch"]
    loss_meter = AverageMeter()
    component_meters = _loss_meters(config)
    gate_meters = None
    if config.get("use_adaptive_fusion", False):
        gate_meters = {"gate_clip": AverageMeter(), "gate_frame": AverageMeter()}

    model.train()
    train_bar = tqdm(train_loader, desc=f"epoch {epoch}", total=len(train_loader), unit="batch", dynamic_ncols=True)
    for batch_index, batch in enumerate(train_bar):
        batch = to_device(batch, device)
        optimizer.zero_grad()
        model_output = model(batch)
        loss_output = criterion(model_output, batch)

        if component_meters is not None:
            loss_value, loss_items = loss_output
            for key, meter in component_meters.items():
                value = loss_items[key]
                meter.update(value.detach().item() if torch.is_tensor(value) else value)
        else:
            loss_value = loss_output

        loss_value.backward()
        optimizer.step()
        if ema is not None:
            ema(model, ema_step)
            ema_step += 1

        loss_meter.update(loss_value.detach().item())
        if gate_meters is not None and model.last_fusion_gate is not None:
            gate_mean = model.last_fusion_gate.mean().detach().item()
            gate_meters["gate_clip"].update(gate_mean)
            gate_meters["gate_frame"].update(1.0 - gate_mean)
        train_bar.set_description(
            f"exp: {config['model_name']} epoch:{epoch:2d} iter:{batch_index:3d} loss:{loss_value.detach().item():.4f}"
        )

    components = None if component_meters is None else {key: meter.avg for key, meter in component_meters.items()}
    if components is not None and gate_meters is not None:
        components.update({key: meter.avg for key, meter in gate_meters.items()})
    return loss_meter.avg, ema_step, components


def _evaluate_with_ema(model, evaluator, context_loader, query_loader, ema):
    if ema is not None:
        ema.assign(model)
    try:
        return evaluator(model, context_loader, query_loader)
    finally:
        if ema is not None:
            ema.resume(model)


def _log_metrics(logger, prefix, values):
    logger.info(
        "%s R@1: %.1f R@5: %.1f R@10: %.1f R@100: %.1f Rsum: %.1f",
        prefix,
        *[float(value) for value in values],
    )


def validate_one_epoch(
    epoch,
    context_loader,
    query_loader,
    model,
    evaluator,
    config,
    optimizer,
    best_value,
    average_loss,
    logger,
    ema=None,
    loss_components=None,
    criterion=None,
):
    validation_value = _evaluate_with_ema(model, evaluator, context_loader, query_loader, ema)
    improved = float(validation_value[4]) > float(best_value[4])
    if improved:
        best_value = [float(value) for value in validation_value]
        ema_shadow = None if ema is None else {key: value.detach().cpu() for key, value in ema.shadow.items()}
        criterion_state = None
        if config.get("learnable_loss_weights", False) and criterion is not None:
            criterion_state = criterion.state_dict()
        save_checkpoint(
            model,
            optimizer,
            config,
            os.path.join(config["model_root"], "best.ckpt"),
            epoch,
            best_value,
            ema_shadow,
            criterion_state,
        )

    curvatures = extract_curvatures(model)
    logger.info("Epoch: %d Average Loss: %.4f", epoch, average_loss)
    if loss_components is not None:
        logger.info("Loss Components: %s", loss_components)
    _log_metrics(logger, "Validation", validation_value)
    _log_metrics(logger, "Best", best_value)
    logger.info("Curvatures: %s", curvatures)
    return validation_value, best_value, not improved, curvatures


def _initialize_ema(model, config, checkpoint):
    if not config.get("use_ema", False):
        return None
    ema = EMA(config.get("ema_decay", 0.999))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            ema.register(name, parameter.data)
    shadow = normalize_ema_state(checkpoint.get("ema_shadow") if checkpoint else None)
    if shadow is not None:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in shadow:
                ema.shadow[name] = shadow[name].to(parameter.device)
    return ema


def _prepare_output(config):
    Path(config["model_root"]).mkdir(parents=True, exist_ok=True)
    Path(config["ckpt_path"]).mkdir(parents=True, exist_ok=True)
    with open(Path(config["model_root"]) / "hyperparams.yaml", "w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=True)


def train(config, resume=None):
    _prepare_output(config)
    logger = create_logger(config["model_root"])
    set_seed(config["seed"])
    device = resolve_device()
    logger.info("Device: %s", device)

    train_loader, context_loader, query_loader = build_training_loaders(config)
    model = build_model(config)
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint is not None:
        load_model_state(model, checkpoint["state_dict"])
    model = model.to(device)

    criterion = build_objective(config).to(device)
    if checkpoint is not None and checkpoint.get("criterion_state") is not None:
        criterion.load_state_dict(checkpoint["criterion_state"])
    evaluator = build_evaluator(config)
    extra_parameters = list(criterion.named_parameters()) if config.get("learnable_loss_weights", False) else None
    optimizer = build_optimizer(config, model, train_loader, extra_parameters)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    ema = _initialize_ema(model, config, checkpoint)

    current_epoch = int(checkpoint["epoch"]) if checkpoint is not None else -1
    best_value = [float(value) for value in checkpoint.get("model_val", [0, 0, 0, 0, 0])] if checkpoint else [0.0] * 5
    ema_step = (current_epoch + 1) * len(train_loader) if current_epoch >= 0 else 0
    early_stop_count = 0
    curvature_history = []

    for epoch in range(current_epoch + 1, config["n_epoch"]):
        epoch_start = time.time()
        average_loss, ema_step, loss_components = train_one_epoch(
            epoch,
            train_loader,
            model,
            criterion,
            config,
            optimizer,
            device,
            ema,
            ema_step,
        )
        with torch.no_grad():
            _, best_value, failed_to_improve, curvatures = validate_one_epoch(
                epoch,
                context_loader,
                query_loader,
                model,
                evaluator,
                config,
                optimizer,
                best_value,
                average_loss,
                logger,
                ema,
                loss_components,
                criterion,
            )
        curvature_history.append(curvatures)
        logger.info("Epoch Time: %.2fs", time.time() - epoch_start)

        early_stop_count = early_stop_count + 1 if failed_to_improve else 0
        if config["max_es_cnt"] != -1 and early_stop_count > config["max_es_cnt"]:
            logger.info("Early Stop")
            break

    if curvature_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(config["model_root"], f"{timestamp}_curvature.png")
        plot_curvature_evolution(curvature_history, plot_path)
        logger.info("Curvature plot: %s", plot_path)


def evaluate(config, checkpoint_path):
    _prepare_output(config)
    logger = create_logger(config["model_root"], "evaluation.log")
    set_seed(config["seed"])
    device = resolve_device()
    context_loader, query_loader = build_evaluation_loaders(config)
    model = build_model(config)
    checkpoint = load_checkpoint(checkpoint_path)
    load_model_state(model, checkpoint["state_dict"])
    model = model.to(device)
    evaluator = build_evaluator(config)
    ema = _initialize_ema(model, config, checkpoint)
    with torch.no_grad():
        values = _evaluate_with_ema(model, evaluator, context_loader, query_loader, ema)
    _log_metrics(logger, "Test", values)
    return values

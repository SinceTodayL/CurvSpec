from curvspec.runtime.optimization import BertAdam


def build_optimizer(config, model, train_loader, extra_parameters=None):
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    parameter_groups = [
        {
            "params": [parameter for name, parameter in named_parameters if not any(term in name for term in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [parameter for name, parameter in named_parameters if any(term in name for term in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if extra_parameters:
        parameter_groups.append(
            {
                "params": [parameter for _, parameter in extra_parameters if parameter.requires_grad],
                "weight_decay": 0.0,
            }
        )

    return BertAdam(
        parameter_groups,
        lr=config["lr"],
        weight_decay=config["wd"],
        warmup=config["lr_warmup_proportion"],
        t_total=len(train_loader) * config["n_epoch"],
        schedule="warmup_linear",
    )

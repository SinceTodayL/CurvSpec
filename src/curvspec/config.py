from pathlib import Path

import yaml


DATASET_ALIASES = {
    "act": "activitynet",
    "cha": "charades",
    "tvr": "tvr",
}


class AttributeDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def load_config(dataset, project_root, data_root=None, output_root=None):
    project_root = Path(project_root).resolve()
    config_path = project_root / "experiments" / f"{dataset}.yaml"
    if dataset not in DATASET_ALIASES or not config_path.exists():
        choices = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {choices}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    dataset_name = DATASET_ALIASES[dataset]
    resolved_data_root = Path(data_root).expanduser().resolve() if data_root else project_root / "dataset" / "netdisk"
    resolved_output_root = Path(output_root).expanduser().resolve() if output_root else project_root / "outputs"

    config.update(
        dataset_alias=dataset,
        dataset_name=dataset_name,
        root=str(project_root),
        data_root=str(resolved_data_root),
        model_root=str(resolved_output_root / dataset_name / config["model_name"]),
    )
    config["ckpt_path"] = str(Path(config["model_root"]) / "checkpoints")
    config["num_workers"] = 1 if config.get("no_core_driver", False) else config["num_workers"]
    config["pin_memory"] = not config.get("no_pin_memory", False)
    return config

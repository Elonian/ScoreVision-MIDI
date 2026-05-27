__all__ = [
    "GrandStaffDataset",
    "ctc_collate",
    "deep_update",
    "get_metrics",
    "load_or_create_vocabulary",
    "load_yaml_config",
    "log_environment",
    "resolve_path",
    "run_training",
    "setup_logging",
]


def __getattr__(name):
    if name in {"deep_update", "load_yaml_config", "resolve_path"}:
        from utils.config import deep_update, load_yaml_config, resolve_path

        return {
            "deep_update": deep_update,
            "load_yaml_config": load_yaml_config,
            "resolve_path": resolve_path,
        }[name]

    if name in {"GrandStaffDataset", "ctc_collate"}:
        from utils.data import GrandStaffDataset, ctc_collate

        return {"GrandStaffDataset": GrandStaffDataset, "ctc_collate": ctc_collate}[name]

    if name in {"log_environment", "setup_logging"}:
        from utils.logging import log_environment, setup_logging

        return {"log_environment": log_environment, "setup_logging": setup_logging}[name]

    if name == "get_metrics":
        from utils.metrics import get_metrics

        return get_metrics

    if name == "run_training":
        from utils.training import run_training

        return run_training

    if name == "load_or_create_vocabulary":
        from utils.vocabulary import load_or_create_vocabulary

        return load_or_create_vocabulary

    raise AttributeError(f"module 'utils' has no attribute {name!r}")

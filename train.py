"""Entry point for launching training via PyTorch Lightning's CLI."""

from __future__ import annotations

import sys
from pathlib import Path

from importlib import import_module
from typing import Any, Mapping, Optional

from jsonargparse import Namespace
from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers.logger import Logger

from datasets import ShakespeareDataModule
from models import CharDecoderTransformer

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"


class TrainCLI(LightningCLI):
    """Custom LightningCLI wiring models, data modules, and trainer config."""

    def __init__(self, **kwargs) -> None:
        parser_kwargs = kwargs.pop("parser_kwargs", {})
        default_config_files = parser_kwargs.get("default_config_files", [])
        default_config_files = [
            str(_DEFAULT_CONFIG),
            *default_config_files,
        ]
        parser_kwargs["default_config_files"] = default_config_files
        super().__init__(
            CharDecoderTransformer,
            ShakespeareDataModule,
            seed_everything_default=42,
            save_config_callback=None,
            run=True,
            subclass_mode_model=True,
            subclass_mode_data=True,
            parser_kwargs=parser_kwargs,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_subclass_arguments(Logger, "logger", required=False)
        parser.link_arguments("logger", "trainer.logger", apply_on="instantiate")
    def before_instantiate_classes(self) -> None:
        active_cfg = _get_active_config(self.config)
        metadata = _infer_datamodule_metadata(active_cfg)
        if metadata is None:
            return
        model_cfg = getattr(active_cfg, "model", None)
        if model_cfg is None:
            return
        vocab_size = metadata.get("vocab_size")
        if vocab_size is not None:
            if hasattr(model_cfg, "init_args"):
                setattr(model_cfg.init_args, "vocab_size", vocab_size)
            elif isinstance(model_cfg, dict):
                model_cfg.setdefault("init_args", {})["vocab_size"] = vocab_size

        sequence_length = metadata.get("sequence_length")
        target_max_seq = None
        if hasattr(model_cfg, "init_args"):
            target_max_seq = getattr(model_cfg.init_args, "max_sequence_length", None)
        elif isinstance(model_cfg, dict):
            target_max_seq = model_cfg.get("init_args", {}).get("max_sequence_length")
        if sequence_length is not None and target_max_seq is None:
            if hasattr(model_cfg, "init_args"):
                setattr(model_cfg.init_args, "max_sequence_length", sequence_length)
            elif isinstance(model_cfg, dict):
                model_cfg.setdefault("init_args", {})["max_sequence_length"] = sequence_length


def _resolve_vocab_size(datamodule: LightningDataModule) -> int:
    if getattr(datamodule, "vocab_size", None) is None:
        datamodule.prepare_data()
        datamodule.setup("fit")
    vocab_size = getattr(datamodule, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("Failed to infer vocab_size from datamodule; please set it explicitly.")
    return vocab_size


def _infer_datamodule_metadata(config: Namespace) -> Optional[dict[str, Any]]:
    data_cfg = getattr(config, "data", None)
    if data_cfg is None:
        return None
    data_dict = _namespace_to_dict(data_cfg)
    class_path = data_dict.get("class_path")
    if not class_path:
        return None
    init_args = _namespace_to_dict(data_dict.get("init_args"))
    datamodule_cls = _import_class(class_path)
    datamodule = datamodule_cls(**init_args)
    try:
        metadata: dict[str, Any] = {}
        if hasattr(datamodule, "sequence_length"):
            metadata["sequence_length"] = getattr(datamodule, "sequence_length")
        metadata["vocab_size"] = _resolve_vocab_size(datamodule)
        return metadata
    finally:
        teardown = getattr(datamodule, "teardown", None)
        if callable(teardown):
            try:
                teardown(stage="fit")
            except TypeError:
                teardown()


def _namespace_to_dict(value: Optional[Any]) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, Namespace):
        return value.as_dict()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unable to convert value of type {type(value)!r} to dict.")


def _import_class(class_path: str) -> type:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid class path '{class_path}'. Expected 'module.ClassName'.")
    module = import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Class '{class_name}' not found in module '{module_name}'.") from exc


def _get_active_config(config: Namespace) -> Namespace:
    subcommand = getattr(config, "subcommand", None)
    if subcommand and hasattr(config, subcommand):
        return getattr(config, subcommand)
    return config


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["fit"]
    first = argv[0]
    if first in {"fit", "validate", "test", "predict", "tune"}:
        return argv
    if first.startswith("-"):
        return ["fit", *argv]
    return argv


def main() -> None:
    cli_args = _normalize_argv(sys.argv[1:])
    sys.argv = [sys.argv[0], *cli_args]
    TrainCLI()


if __name__ == "__main__":
    main()

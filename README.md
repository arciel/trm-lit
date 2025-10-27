# trm-lit


### wotsit
Unofficial implementation of Tiny Recursive Models (arxiv, official code).

This implementation makes the following changes
* Uses Pytorch-Lightning to drive the train loop and boilerplate
* Adds support for DDP for multi GPU training
* Simplified model implementation while keeping the same structure
* Makes it easy to train on different datasets
* Exposes a notebook for playing with trained models


I started working on this after I got frustrated at the original implementation's code,
which is hard to follow and compare to the paper. It builds on top of the HRM code release and there are many artefacts which hinder understanding, such as the CastedXyz layers. 

Switching to using Lightning as a driver allows for easier device and dtype management and simplifies the expression of the core ideas.

### Usage

This project uses [`uv`](https://docs.astral.sh/uv/) for environment management. After cloning:

```sh
uv sync
```

### Train the default Shakespeare model

```sh
uv run python train.py
```

`train.py` wraps the [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html), so any CLI override that Lightning understands will work. For example:

```sh
uv run python train.py --trainer.fast_dev_run 1
uv run python train.py --config configs/overrides.yaml
uv run python train.py --fit.model.init_args.num_layers 8
```

By default configuration lives in `configs/default.yaml`. The CLI merges files (via `--config`) followed by CLI flags, making it easy to define repeatable experiment presets. Because Lightning's CLI is subcommand-based, configuration for the active command goes under the `fit:` block (e.g. `fit.model`, `fit.data`). The default config wires the `ShakespeareDataModule` and `CharDecoderTransformer` together, enables a W&B logger (offline by default), and adds standard callbacks such as `ModelCheckpoint`, `EarlyStopping`, and `LearningRateMonitor`.

### Working with datasets and models

- Datasets live under `datasets/` as Lightning `DataModule`s. The included `ShakespeareDataModule` automatically downloads the Tiny Shakespeare corpus (with an embedded fallback when offline).
- Models live under `models/` as Lightning `LightningModule`s. The default `CharDecoderTransformer` provides a compact decoder-only Transformer ready for character-level language modelling and exposes a `generate` helper for quick sampling.

To plug in your own dataset or model, add a new module and reference its import path in either the config file or via CLI override (e.g. `--data.class_path yourpackage.YourDataModule`).

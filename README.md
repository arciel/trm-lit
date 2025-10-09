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

This is built with `uv`.

Grab the code
```sh
$ git clone github.com/arciel/trm-lit
$ cd trm-lit
$ uv sync
```

Prepare dataset
```sh
$ uv run data_prepare.py --data=<see-below>
```

Start training
```sh
$ uv run train.py --arch=trm --data=<see-below>
```

Here,
`arch in [trm, transformer]` and `data in [sudoku1k, maze1k, arc_agi_1, arc_agi_2]`


















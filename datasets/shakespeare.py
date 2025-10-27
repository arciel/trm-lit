"""Lightning DataModule for the Tiny Shakespeare character-level dataset."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

_LOG = logging.getLogger(__name__)

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
FALLBACK_SAMPLE = (
    "From fairest creatures we desire increase,\n"
    "That thereby beauty's rose might never die,\n"
    "But as the riper should by time decease,\n"
    "His tender heir might bear his memory.\n"
)


@dataclass
class CharTokenizer:
    """Simple character-level tokenizer."""

    vocab: str

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = "".join(sorted(set(text)))
        return cls(vocab=vocab)

    @property
    def stoi(self) -> dict[str, int]:
        return {ch: idx for idx, ch in enumerate(self.vocab)}

    @property
    def itos(self) -> dict[int, str]:
        return dict(enumerate(self.vocab))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        table = self.stoi
        return [table[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        table = self.itos
        return "".join(table[token] for token in tokens)

    def state_dict(self) -> dict[str, str]:
        return {"vocab": self.vocab}

    def load_state_dict(self, state_dict: dict[str, str]) -> None:
        self.vocab = state_dict["vocab"]


class CharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Slice a long token sequence into contiguous language-model samples."""

    def __init__(self, tokens: torch.Tensor, sequence_length: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("Expected tokens tensor to be 1D")
        if tokens.dtype != torch.long:
            raise TypeError("Expected tokens tensor dtype=torch.long")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if tokens.numel() <= sequence_length:
            raise ValueError("Token sequence too short for requested sequence_length")

        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.tokens.numel() - self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.sequence_length
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


class ShakespeareDataModule(LightningDataModule):
    """LightningDataModule producing character-level sequences from Shakespeare."""

    def __init__(
        self,
        data_dir: str | Path = "data/shakespeare",
        sequence_length: int = 128,
        batch_size: int = 64,
        num_workers: int = 0,
        train_fraction: float = 0.9,
        pin_memory: bool = True,
        source_url: str = DEFAULT_SOURCE_URL,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.pin_memory = pin_memory
        self.source_url = source_url
        self.persistent_workers = persistent_workers and num_workers > 0

        self._tokenizer: Optional[CharTokenizer] = None
        self.train_dataset: Optional[CharDataset] = None
        self.val_dataset: Optional[CharDataset] = None
        self.vocab_size: Optional[int] = None

    @property
    def tokenizer(self) -> CharTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer accessed before setup")
        return self._tokenizer

    def prepare_data(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self._dataset_path
        if dataset_path.exists():
            return
        try:
            _LOG.info("Downloading Tiny Shakespeare dataset from %s", self.source_url)
            urllib.request.urlretrieve(self.source_url, dataset_path)
        except Exception as exc:  # pragma: no cover - best effort download
            _LOG.warning(
                "Falling back to embedded Tiny Shakespeare sample because download failed: %s",
                exc,
            )
            dataset_path.write_text(FALLBACK_SAMPLE, encoding="utf-8")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        text = self._dataset_path.read_text(encoding="utf-8")
        tokenizer = CharTokenizer.from_text(text)
        encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        split_idx = int(encoded.numel() * self.train_fraction)
        split_idx = max(split_idx, self.sequence_length + 1)
        train_tokens = encoded[:split_idx]
        val_tokens = encoded[split_idx:]
        if val_tokens.numel() <= self.sequence_length:
            # ensure validation set has at least one sample
            val_tokens = encoded[-(self.sequence_length + 2) :]

        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.train_dataset = CharDataset(train_tokens, self.sequence_length)
        self.val_dataset = CharDataset(val_tokens, self.sequence_length)

    def state_dict(self) -> dict[str, dict[str, str]]:
        if self._tokenizer is None:
            return {}
        return {"tokenizer": self._tokenizer.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, str]]) -> None:
        token_state = state_dict.get("tokenizer")
        if token_state is None:
            return
        if self._tokenizer is None:
            self._tokenizer = CharTokenizer(vocab="")
        self._tokenizer.load_state_dict(token_state)
        self.vocab_size = self._tokenizer.vocab_size

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must be called before requesting train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self.val_dataset is None:
            raise RuntimeError("setup('validate') must be called before requesting val_dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    @property
    def _dataset_path(self) -> Path:
        return self.data_dir / "tiny_shakespeare.txt"

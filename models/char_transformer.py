"""LightningModule implementing a small autoregressive decoder transformer."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import nn
from torch.optim import AdamW

class SinusoidalPositionalEncoding(nn.Module):
    """Classic transformer sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len]


class CharDecoderTransformer(LightningModule):
    """Tiny causal decoder tailored for character-level language modelling."""

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int = 256,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        optimizer_betas: Tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        warmup_steps: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        if seq_len > self.causal_mask.size(0):
            raise ValueError("Sequence length exceeds model max_sequence_length")
        x = self.embedding(tokens) * math.sqrt(self.hparams.d_model)
        x = x + self.positional_encoding(seq_len).to(x.device)
        x = self.dropout(x)
        causal_mask = self.causal_mask[:seq_len, :seq_len].to(x.device)
        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self._compute_loss(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log_perplexity(loss, stage="train")
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        inputs, targets = batch
        logits = self(inputs)
        loss = self._compute_loss(logits, targets)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log_perplexity(loss, stage="val")

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        loss = self.criterion(logits.view(-1, vocab_size), targets.view(-1))
        return loss

    @rank_zero_only
    def log_perplexity(self, loss: torch.Tensor, stage: str) -> None:
        ppl = torch.exp(torch.clamp(loss.detach(), max=20.0))
        self.log(f"{stage}/perplexity", ppl, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.optimizer_betas,
            eps=self.hparams.optimizer_eps,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = self.hparams.warmup_steps
        if warmup_steps <= 0:
            return optimizer

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max((step + 1) / float(warmup_steps), 1e-3)
            return 1.0

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def generate(self, seed: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        generated = seed.clone()
        max_len = self.causal_mask.size(0)
        for _ in range(max_new_tokens):
            input_seq = generated[:, -max_len:]
            logits = self(input_seq)
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.distributions.Categorical(probs=probs).sample().unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class TinyRecursiveReasoningModelCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class ReasoningBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, expansion: float, norm_eps: float
    ) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states
            + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class TinyRecursiveReasoningModel(nn.Module):
    """Simplified Tiny Recursive Reasoning Model with fixed configuration."""

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        num_puzzle_identifiers: int,
        vocab_size: int,
    ) -> None:
        super().__init__()

        # Fixed configuration taken from config/arch/trm.yaml
        self.hidden_size = 512
        self.expansion = 4.0
        self.num_heads = 8
        self.rms_norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.h_cycles = 3
        self.l_cycles = 6
        self.l_layers = 2
        self.halt_max_steps = 16
        self.halt_exploration_prob = 0.1
        self.puzzle_emb_ndim = self.hidden_size
        self.puzzle_emb_len = 16
        self.forward_dtype = torch.bfloat16

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.total_sequence_length = self.seq_len + self.puzzle_emb_len

        self.embed_scale = math.sqrt(self.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.vocab_size,
            self.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.puzzle_emb = CastedSparseEmbedding(
            self.num_puzzle_identifiers,
            self.puzzle_emb_ndim,
            batch_size=self.batch_size,
            init_std=0,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.hidden_size, self.vocab_size, bias=False)
        self.q_head = CastedLinear(self.hidden_size, 2, bias=True)

        self.rotary_emb = RotaryEmbedding(
            dim=self.hidden_size // self.num_heads,
            max_position_embeddings=self.total_sequence_length,
            base=self.rope_theta,
        )

        self.reasoning_layers = nn.ModuleList(
            ReasoningBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                expansion=self.expansion,
                norm_eps=self.rms_norm_eps,
            )
            for _ in range(self.l_layers)
        )

        self.register_buffer(
            "H_init",
            trunc_normal_init_(
                torch.empty(self.hidden_size, dtype=self.forward_dtype),
                std=1.0,
            ),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(
                torch.empty(self.hidden_size, dtype=self.forward_dtype),
                std=1.0,
            ),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def initial_carry(
        self, batch: Dict[str, torch.Tensor]
    ) -> TinyRecursiveReasoningModelCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        z_H, z_L = self._empty_state(batch_size, device)
        steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        current_data = {key: torch.empty_like(value) for key, value in batch.items()}
        return TinyRecursiveReasoningModelCarry(
            z_H=z_H, z_L=z_L, steps=steps, halted=halted, current_data=current_data
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModelCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModelCarry, Dict[str, torch.Tensor]]:
        z_H, z_L = self._reset_state(carry)
        steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        current_data = self._update_batch_data(carry, batch)

        cos_sin = self.rotary_emb()
        input_embeddings = self._input_embeddings(
            current_data["inputs"], current_data["puzzle_identifiers"]
        )

        z_H, z_L = self._run_reasoning_cycles(z_H, z_L, input_embeddings, cos_sin)

        logits = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        new_carry = TinyRecursiveReasoningModelCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            steps=steps,
            halted=carry.halted.clone(),
            current_data=current_data,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_logits[..., 0],
            "q_continue_logits": q_logits[..., 1],
        }

        with torch.no_grad():
            new_carry.steps = new_carry.steps + torch.ones_like(new_carry.steps)
            is_last_step = new_carry.steps >= self.halt_max_steps

            halted = is_last_step | (q_logits[..., 0] > 0)

            exploration_mask = (
                torch.rand_like(q_logits[..., 0]) < self.halt_exploration_prob
            )
            sampled_steps = torch.randint_like(
                new_carry.steps,
                low=2,
                high=self.halt_max_steps + 1,
            )
            min_halt_steps = exploration_mask * sampled_steps
            halted = halted & (new_carry.steps >= min_halt_steps)

            new_carry.halted = halted

        return new_carry, outputs

    def _empty_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.total_sequence_length, self.hidden_size)
        z_H = torch.empty(shape, dtype=self.forward_dtype, device=device)
        z_L = torch.empty(shape, dtype=self.forward_dtype, device=device)
        return z_H, z_L

    def _reset_state(
        self, carry: TinyRecursiveReasoningModelCarry
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = carry.halted.view(-1, 1, 1)
        z_H = torch.where(mask, self.H_init, carry.z_H)
        z_L = torch.where(mask, self.L_init, carry.z_L)
        return z_H, z_L

    def _update_batch_data(
        self,
        carry: TinyRecursiveReasoningModelCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        updated = {}
        for key, value in batch.items():
            dims = (1,) * (value.ndim - 1)
            mask = carry.halted.view(-1, *dims)
            updated[key] = torch.where(mask, value, carry.current_data[key])
        return updated

    def _input_embeddings(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.embed_tokens(inputs.to(torch.int32))

        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
        pad_count = self.puzzle_emb_len * self.hidden_size - puzzle_embedding.shape[-1]
        if pad_count > 0:
            puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
        puzzle_embedding = puzzle_embedding.view(
            -1, self.puzzle_emb_len, self.hidden_size
        )

        embeddings = torch.cat((puzzle_embedding, token_embeddings), dim=-2)
        return embeddings * self.embed_scale

    def _apply_reasoning_layers(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: CosSin,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.reasoning_layers:
            hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin)
        return hidden_states

    def _run_reasoning_cycles(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        input_embeddings: torch.Tensor,
        cos_sin: CosSin,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            for _ in range(self.h_cycles - 1):
                for _ in range(self.l_cycles):
                    z_L = self._apply_reasoning_layers(
                        z_L, z_H + input_embeddings, cos_sin
                    )
                z_H = self._apply_reasoning_layers(z_H, z_L, cos_sin)

        for _ in range(self.l_cycles):
            z_L = self._apply_reasoning_layers(z_L, z_H + input_embeddings, cos_sin)
        z_H = self._apply_reasoning_layers(z_H, z_L, cos_sin)

        return z_H, z_L

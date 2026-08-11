from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import modulate, rope_apply
from .ovis_u1_imports import ensure_ovis_u1_remote_code_importable
from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# flex_attention path for MageFlow packed mixed attention.
#
# The dense `[1, L, L]` bool mask built by `_build_mot_attention_mask_*` is
# ~96% zeros (only intra-sample text/target/reference/action blocks are True).
# SDPA rejects it for the flash backend (arbitrary bool mask) and falls back to
# the memory-efficient backend, which still computes the *full* O(L^2) attention
# and masks the output — wasting ~27x the work and dominating backward time.
#
# `flex_attention` + `BlockMask` skips the all-zero blocks entirely, computing
# only the ~3.7% nonzero block pairs. Exactness at non-block-aligned token
# boundaries (variable text length, target/ref/action spans) is guaranteed by a
# `score_mod` that re-applies the identical per-position rule used by the dense
# mask builder, so the flex path is numerically equivalent to the SDPA path.
#
# Token block types (per-sample packed order): 0=text, 1=target, 2=reference,
# 3=action. The visibility rule (target read only by itself; reference/action
# never read target; target/action mutually isolated) is encoded directly in
# `_flex_attention_call`'s score_mod via the is_target/is_action flags.
# ---------------------------------------------------------------------------
def _build_mage_flow_token_maps(
    text_cu_lens: torch.Tensor,
    image_cu_lens: torch.Tensor,
    action_cu_lens: torch.Tensor,
    target_len: int,
    text_total: int,
    image_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (sample_id, block_type) for the global [text|image|action] pack.

    Mirrors the geometry of `_build_mot_attention_mask_mage_flow_packed` so the
    flex score_mod reproduces the exact same mask. `target_len` is the per-sample
    target-image token count (reference follows it in the image span).
    """
    text_cu_lens = text_cu_lens.to(device=device, dtype=torch.long)
    image_cu_lens = image_cu_lens.to(device=device, dtype=torch.long)
    action_cu_lens = action_cu_lens.to(device=device, dtype=torch.long)
    batch_size = int(text_cu_lens.numel() - 1)
    action_total = int(action_cu_lens[-1])
    total = text_total + image_total + action_total
    sample_id = torch.empty(total, dtype=torch.long, device=device)
    block_type = torch.empty(total, dtype=torch.long, device=device)
    image_base = text_total
    action_base = text_total + image_total
    for s in range(batch_size):
        t0, t1 = int(text_cu_lens[s]), int(text_cu_lens[s + 1])
        i0 = image_base + int(image_cu_lens[s])
        i1 = image_base + int(image_cu_lens[s + 1])
        a0 = action_base + int(action_cu_lens[s])
        a1 = action_base + int(action_cu_lens[s + 1])
        target_end = i0 + target_len
        if target_end > i1:
            raise ValueError(
                f"target_len={target_len} exceeds image span "
                f"[{i0 - image_base}, {i1 - image_base}) for sample {s}"
            )
        sample_id[t0:t1] = s
        block_type[t0:t1] = 0  # text
        sample_id[i0:target_end] = s
        block_type[i0:target_end] = 1  # target
        sample_id[target_end:i1] = s
        block_type[target_end:i1] = 2  # reference
        sample_id[a0:a1] = s
        block_type[a0:a1] = 3  # action
    return sample_id, block_type


def _flex_attention_call(q, k, v, block_mask, sample_id, is_target, is_action):
    """Compiled body for block-sparse MageFlow mixed attention.

    `score_mod` is defined *inside* the compiled region and closes over the
    tensor ARGS, so torch.compile sees one stable graph; the per-forward tensors
    are dynamic-shape inputs (variable packed L), not captured constants.

    Exactness: `block_mask` (built from `sample_id[q]==sample_id[k]`) isolates
    samples; `score_mod` then removes the intra-sample holes by forbidding any
    query from reading target/action tokens unless it is itself target/action.
    This reproduces every cell of the 4x4 visibility matrix used by the dense
    mask builder, without 2D advanced indexing (which crashes the flex backward
    Triton kernel in torch 2.7.x).
    """
    from torch.nn.attention.flex_attention import flex_attention

    def score_mod(score, b, h, q_idx, kv_idx):
        # cross-sample is already handled by the block_mask; here we only carve
        # out the within-sample holes.
        score = score.masked_fill(is_target[kv_idx] & ~is_target[q_idx], float("-inf"))
        score = score.masked_fill(is_action[kv_idx] & ~is_action[q_idx], float("-inf"))
        return score

    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)


_FLEX_ATTENTION_COMPILED = None


def _get_flex_attention_compiled():
    """Lazily torch.compile the flex body once (dynamic shapes)."""
    global _FLEX_ATTENTION_COMPILED
    if _FLEX_ATTENTION_COMPILED is None:
        _FLEX_ATTENTION_COMPILED = torch.compile(_flex_attention_call, dynamic=True)
    return _FLEX_ATTENTION_COMPILED


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        gqa_implementation: str = "repeat",
        force_flash_attention: bool = False,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.gqa_implementation = str(gqa_implementation).strip().lower()
        if self.gqa_implementation not in {"repeat", "sdpa"}:
            raise ValueError(
                f"`gqa_implementation` must be 'repeat' or 'sdpa', got {gqa_implementation!r}."
            )
        self.force_flash_attention = bool(force_flash_attention)
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention.")

        # MageFlow packed mixed-attention backend. "sdpa" (default) builds the
        # dense `[1, L, L]` bool mask and runs SDPA (flash rejects the mask, so
        # it falls back to the memory-efficient backend at full O(L^2) cost).
        # "flex" uses `flex_attention` + BlockMask to skip the ~96% all-zero
        # block pairs, computing only the nonzero intra-sample blocks. Set via
        # `MAGEFLOW_MIXED_ATTN=flex`. See `_forward_mage_flow`.
        # NOTE: flex REQUIRES torch.compile (eager flex_attention is ~9x slower
        # than dense SDPA). The compiled body (`_flex_attention_call`) keeps
        # score_mod inside the graph and takes the per-forward token maps as
        # dynamic-shape tensor args, so one graph is reused across steps.
        self.mage_flow_attn_backend = os.environ.get("MAGEFLOW_MIXED_ATTN", "sdpa").strip().lower()
        if self.mage_flow_attn_backend not in {"sdpa", "flex"}:
            raise ValueError(
                f"MAGEFLOW_MIXED_ATTN must be 'sdpa' or 'flex', got {self.mage_flow_attn_backend!r}."
            )

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = int(first_expert.num_heads)
        self.num_kv_heads = int(getattr(first_expert, "num_kv_heads", first_expert.num_heads))
        self.attn_head_dim = int(first_expert.attn_head_dim)
        self.block_protocol = str(getattr(first_expert, "block_protocol", "wan22"))

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            protocol = str(getattr(expert, "block_protocol", "wan22"))
            num_kv_heads = int(getattr(expert, "num_kv_heads", expert.num_heads))
            checks = {
                "num_layers": (len(expert.blocks), self.num_layers),
                "num_heads": (int(expert.num_heads), self.num_heads),
                "num_kv_heads": (num_kv_heads, self.num_kv_heads),
                "attn_head_dim": (int(expert.attn_head_dim), self.attn_head_dim),
                "block_protocol": (protocol, self.block_protocol),
            }
            for attr, (got, expected) in checks.items():
                if got != expected:
                    raise ValueError(f"All experts must share {attr}; got {got} vs {expected} for expert {name}.")

        logger.info(
            "Initialized MoT with experts=%s protocol=%s layers=%d heads=%d kv_heads=%d head_dim=%d",
            self.expert_order,
            self.block_protocol,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.attn_head_dim,
        )
        logger.info(
            "MoT attention config: gqa_implementation=%s force_flash_attention=%s",
            self.gqa_implementation,
            self.force_flash_attention,
        )

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _format_attention_mask(
        attention_mask: torch.Tensor,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 2:
            if tuple(mask.shape) != (query_len, key_len):
                raise ValueError(f"2D attention mask must be {(query_len, key_len)}, got {tuple(mask.shape)}")
            return mask.view(1, 1, query_len, key_len)
        if mask.ndim == 3:
            if mask.shape[0] != batch_size or tuple(mask.shape[1:]) != (query_len, key_len):
                raise ValueError(
                    f"3D attention mask must be {(batch_size, query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask.unsqueeze(1)
        if mask.ndim == 4:
            if mask.shape[0] not in (1, batch_size) or tuple(mask.shape[-2:]) != (query_len, key_len):
                raise ValueError(
                    f"4D attention mask must end with {(query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask
        raise ValueError(f"attention_mask must be 2D/3D/4D, got shape {tuple(mask.shape)}")

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attn_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = q_cat.shape
        key_len = k_cat.shape[1]
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.attn_head_dim
        attn_mask = self._format_attention_mask(attention_mask, batch_size, query_len, key_len, q_cat.device)

        if return_attn_probs:
            q = q_cat.view(batch_size, query_len, H, D).transpose(1, 2)
            k = k_cat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            v = v_cat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            if H_kv != H:
                if self.gqa_implementation != "repeat":
                    raise ValueError("Attention capture with GQA currently requires gqa_implementation='repeat'.")
                repeat_factor = H // H_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (D ** -0.5)
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            attn_probs = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
            out = torch.matmul(attn_probs, v)
            out = out.transpose(1, 2).reshape(batch_size, query_len, H * D)
            return out, attn_probs

        def _sdpa_context():
            if not self.force_flash_attention:
                return nullcontext()
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
            except Exception as exc:  # pragma: no cover - depends on torch build
                raise RuntimeError("`force_flash_attention=True` requires torch.nn.attention.sdpa_kernel.") from exc
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
            force_flash_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION])

        def _forward(q_flat: torch.Tensor, k_flat: torch.Tensor, v_flat: torch.Tensor) -> torch.Tensor:
            q = q_flat.view(batch_size, query_len, H, D).transpose(1, 2)
            k = k_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            v = v_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            enable_gqa = False
            if H_kv != H and self.gqa_implementation == "repeat":
                repeat_factor = H // H_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            elif H_kv != H:
                enable_gqa = True
            with _sdpa_context():
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
            return out.transpose(1, 2).reshape(batch_size, query_len, H * D)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(_forward, q_cat, k_cat, v_cat, use_reentrant=False)
        return _forward(q_cat, k_cat, v_cat)

    def _flex_mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        block_mask,
        sample_id: torch.Tensor,
        is_target: torch.Tensor,
        is_action: torch.Tensor,
    ) -> torch.Tensor:
        """Block-sparse mixed attention via compiled `flex_attention`.

        `block_mask` (sample-gated, built once per forward) skips the all-zero
        cross-sample block pairs; the in-graph `score_mod` carves the
        within-sample holes so the result is numerically identical to
        `_mixed_attention(..., full_mask)`. MageFlow has no GQA
        (num_kv_heads == num_heads), so q/k/v share head geometry.
        """
        if self.num_kv_heads != self.num_heads:
            raise ValueError("flex mixed attention currently requires num_kv_heads == num_heads.")
        H, D = self.num_heads, self.attn_head_dim
        query_len = q_cat.shape[1]
        q = q_cat.view(1, query_len, H, D).transpose(1, 2)
        k = k_cat.view(1, k_cat.shape[1], H, D).transpose(1, 2)
        v = v_cat.view(1, v_cat.shape[1], H, D).transpose(1, 2)
        out = _get_flex_attention_compiled()(
            q, k, v, block_mask, sample_id, is_target, is_action
        )
        return out.transpose(1, 2).reshape(1, query_len, H * D)

    def _build_io_wan22(self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor) -> dict:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        return {
            "q": q,
            "k": k,
            "v": v,
            "residual_x": x,
            "post_state": {
                "gate_msa": gate_msa,
                "shift_mlp": shift_mlp,
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp,
            },
            "use_gradient_checkpointing": bool(getattr(expert, "use_gradient_checkpointing", False)),
        }

    def _build_io_omnigen2(self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor) -> dict:
        from omnigen2.models.embeddings import apply_rotary_emb

        norm_x, gate_msa, scale_mlp, gate_mlp = block.norm1(x, t_mod)
        batch_size, seq_len = norm_x.shape[:2]
        H, H_kv, D = int(expert.num_heads), int(expert.num_kv_heads), int(expert.attn_head_dim)
        q = block.attn.to_q(norm_x).view(batch_size, seq_len, H, D)
        k = block.attn.to_k(norm_x).view(batch_size, seq_len, H_kv, D)
        v = block.attn.to_v(norm_x).view(batch_size, seq_len, H_kv, D)
        if block.attn.norm_q is not None:
            q = block.attn.norm_q(q)
        if block.attn.norm_k is not None:
            k = block.attn.norm_k(k)
        q = apply_rotary_emb(q, freqs, use_real=False)
        k = apply_rotary_emb(k, freqs, use_real=False)
        return {
            "q": q.reshape(batch_size, seq_len, H * D),
            "k": k.reshape(batch_size, seq_len, H_kv * D),
            "v": v.reshape(batch_size, seq_len, H_kv * D),
            "residual_x": x,
            "post_state": {
                "gate_msa": gate_msa,
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp,
            },
            "use_gradient_checkpointing": bool(getattr(expert, "use_gradient_checkpointing", False)),
        }

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> dict:
        if self.block_protocol == "omnigen2":
            return self._build_io_omnigen2(expert, block, x, freqs, t_mod)
        if self.block_protocol == "wan22":
            return self._build_io_wan22(expert, block, x, freqs, t_mod)
        raise ValueError(f"Unsupported block_protocol={self.block_protocol!r}")

    @staticmethod
    def _sana_split_modulation(block, t_mod: torch.Tensor):
        return (block.scale_shift_table[None].to(t_mod.dtype) + t_mod.reshape(t_mod.shape[0], 6, -1)).chunk(6, dim=1)

    def _sana_video_io(self, block, x: torch.Tensor, t_mod: torch.Tensor, hw: tuple[int, int]) -> dict:
        from models.diffusion.model.nets.sana_blocks import t2i_modulate

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._sana_split_modulation(block, t_mod)
        attn_in = t2i_modulate(block.norm1(x), shift_msa, scale_msa)
        batch_size, seq_len, hidden = attn_in.shape
        qkv = block.attn.qkv(attn_in).reshape(batch_size, seq_len, 3, hidden)
        q, k, v = qkv.unbind(2)
        q = block.attn.q_norm(q).transpose(-1, -2)
        k = block.attn.k_norm(k).transpose(-1, -2)
        v = v.transpose(-1, -2)
        head_dim = int(block.attn.dim)
        q = q.reshape(batch_size, hidden // head_dim, head_dim, seq_len)
        k = k.reshape(batch_size, hidden // head_dim, head_dim, seq_len)
        v = v.reshape(batch_size, hidden // head_dim, head_dim, seq_len)
        return {
            "q": q,
            "k": k,
            "v": v,
            "residual_x": x,
            "hw": hw,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "use_gradient_checkpointing": False,
        }

    @staticmethod
    def _sana_linear_attention(attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = attn.attn_matmul(q, k.transpose(-1, -2), v).to(dtype=v.dtype)
        return out.view(q.shape[0], -1, q.shape[-1]).permute(0, 2, 1)

    @staticmethod
    def _sana_video_post(block, attn_out: torch.Tensor, state: dict, context_payload: Optional[dict]) -> torch.Tensor:
        from models.diffusion.model.nets.sana_blocks import t2i_modulate

        x = state["residual_x"] + state["gate_msa"] * block.attn.proj(attn_out)
        if context_payload is not None and context_payload.get("context") is not None:
            x = x + block.cross_attn(x, context_payload["context"], context_payload.get("mask"))
        mlp_input = t2i_modulate(block.norm2(x), state["shift_mlp"], state["scale_mlp"])
        return x + state["gate_mlp"] * block.mlp(mlp_input, HW=state["hw"])

    def _forward_sana(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: object,
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        del attention_mask, freqs_all
        if "video" not in embeds_all or "cond" not in embeds_all or "action" not in embeds_all:
            raise ValueError("SANA MoT expects embeds_all with keys: video, cond, action.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        video = embeds_all["video"]
        cond = embeds_all["cond"]
        action = embeds_all["action"]
        video_hw = tuple(context_all["video"]["hw"])
        cond_hw = tuple(context_all["cond"]["hw"])
        video_context = context_all.get("video")
        cond_context = context_all.get("cond")
        action_context = context_all.get("action")

        for layer_idx in range(self.num_layers):
            v_block = video_expert.blocks[layer_idx]
            a_block = action_expert.blocks[layer_idx]
            v_state = self._sana_video_io(v_block, video, t_mod_all["video"], video_hw)
            c_state = self._sana_video_io(v_block, cond, t_mod_all["cond"], cond_hw)
            a_state = a_block.build_self_attention_io(action, t_mod_all["action"])

            v_attn = self._sana_linear_attention(v_block.attn, v_state["q"], v_state["k"], v_state["v"])
            c_attn = self._sana_linear_attention(v_block.attn, c_state["q"], c_state["k"], c_state["v"])
            a_attn = a_block.linear_attention(
                a_state["q"],
                torch.cat([c_state["k"].to(a_state["k"].dtype), a_state["k"]], dim=-1),
                torch.cat([c_state["v"].to(a_state["v"].dtype), a_state["v"]], dim=-1),
            )

            video = self._sana_video_post(v_block, v_attn, v_state, video_context)
            cond = self._sana_video_post(v_block, c_attn, c_state, cond_context)
            action = a_block.apply_post(
                a_attn,
                a_state,
                None if action_context is None else action_context.get("context_tokens"),
                None if action_context is None else action_context.get("context_mask_bool"),
            )
        return {"video": video, "cond": cond, "action": action}

    def prefill_sana_condition_cache(
        self,
        cond_tokens: torch.Tensor,
        cond_t_mod: torch.Tensor,
        cond_context_payload: dict,
    ) -> dict[str, object]:
        if self.block_protocol != "sana":
            raise ValueError("`prefill_sana_condition_cache` requires block_protocol='sana'.")
        video_expert = self.mixtures["video"]
        cond = cond_tokens
        hw = tuple(cond_context_payload["hw"])
        caches = []
        for layer_idx in range(self.num_layers):
            block = video_expert.blocks[layer_idx]
            state = self._sana_video_io(block, cond, cond_t_mod, hw)
            caches.append({"k": state["k"], "v": state["v"]})
            attn = self._sana_linear_attention(block.attn, state["q"], state["k"], state["v"])
            cond = self._sana_video_post(block, attn, state, cond_context_payload)
        return {"layers": caches, "final_cond": cond, "hw": hw}

    def forward_sana_action_with_condition_cache(
        self,
        action_tokens: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        condition_cache: dict[str, object],
    ) -> torch.Tensor:
        if self.block_protocol != "sana":
            raise ValueError("`forward_sana_action_with_condition_cache` requires block_protocol='sana'.")
        action_expert = self.mixtures["action"]
        action = action_tokens
        for layer_idx, cache in enumerate(condition_cache["layers"]):
            block = action_expert.blocks[layer_idx]
            state = block.build_self_attention_io(action, action_t_mod)
            attn = block.linear_attention(
                state["q"],
                torch.cat([cache["k"].to(state["k"].dtype), state["k"]], dim=-1),
                torch.cat([cache["v"].to(state["v"].dtype), state["v"]], dim=-1),
            )
            action = block.apply_post(
                attn,
                state,
                None if action_context_payload is None else action_context_payload.get("context_tokens"),
                None if action_context_payload is None else action_context_payload.get("context_mask_bool"),
            )
        return action

    @staticmethod
    def _apply_post_wan22(block, residual_x: torch.Tensor, mixed_attn_out: torch.Tensor, post_state: dict, context_payload: Optional[dict]) -> torch.Tensor:
        x = block.gate(residual_x, post_state["gate_msa"], block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), post_state["shift_mlp"], post_state["scale_mlp"])
        return block.gate(x, post_state["gate_mlp"], block.ffn(mlp_input))

    @staticmethod
    def _apply_post_omnigen2(block, residual_x: torch.Tensor, mixed_attn_out: torch.Tensor, post_state: dict) -> torch.Tensor:
        attn_proj = block.attn.to_out[0](mixed_attn_out)
        attn_proj = block.attn.to_out[1](attn_proj)
        x = residual_x + post_state["gate_msa"].unsqueeze(1).tanh() * block.norm2(attn_proj)
        mlp_input = block.ffn_norm1(x) * (1 + post_state["scale_mlp"].unsqueeze(1))
        mlp_output = block.feed_forward(mlp_input)
        return x + post_state["gate_mlp"].unsqueeze(1).tanh() * block.ffn_norm2(mlp_output)

    def _apply_expert_post_block(
        self,
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        post_state: dict,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        if self.block_protocol == "omnigen2":
            return self._apply_post_omnigen2(block, residual_x, mixed_attn_out, post_state)
        return self._apply_post_wan22(block, residual_x, mixed_attn_out, post_state, context_payload)

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        post_state: dict,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        if self.block_protocol == "wan22":
            def _post_fn(_mixed, _x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
                return self._apply_expert_post_block(
                    block,
                    _x,
                    _mixed,
                    {
                        "gate_msa": gate_msa,
                        "shift_mlp": shift_mlp,
                        "scale_mlp": scale_mlp,
                        "gate_mlp": gate_mlp,
                    },
                    context_payload,
                )

            args = (
                mixed_slice,
                residual_x,
                post_state["gate_msa"],
                post_state["shift_mlp"],
                post_state["scale_mlp"],
                post_state["gate_mlp"],
            )
        else:
            def _post_fn(_mixed, _x, gate_msa, scale_mlp, gate_mlp):
                return self._apply_expert_post_block(
                    block,
                    _x,
                    _mixed,
                    {"gate_msa": gate_msa, "scale_mlp": scale_mlp, "gate_mlp": gate_mlp},
                    context_payload,
                )

            args = (
                mixed_slice,
                residual_x,
                post_state["gate_msa"],
                post_state["scale_mlp"],
                post_state["gate_mlp"],
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(_post_fn, *args, use_reentrant=False)
        return _post_fn(*args)

    def _yak_flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

    def _yak_qkv(self, attn, x: torch.Tensor, pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ensure_ovis_u1_remote_code_importable()
        from ovis_u1_hf.modeling_yak import apply_rope

        batch_size, seq_len = x.shape[:2]
        qkv = attn.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.attn_head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return self._yak_flatten_heads(q), self._yak_flatten_heads(k), self._yak_flatten_heads(v)

    def _yak_single_io(self, block, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> dict:
        ensure_ovis_u1_remote_code_importable()
        from ovis_u1_hf.modeling_yak import apply_rope

        mod, _ = block.modulation(vec)
        x_mod = (1 + mod.scale) * block.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(block.linear1(x_mod), [3 * block.hidden_size, block.mlp_hidden_dim], dim=-1)
        batch_size, seq_len = x.shape[:2]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.attn_head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = block.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._yak_flatten_heads(q),
            "k": self._yak_flatten_heads(k),
            "v": self._yak_flatten_heads(v),
            "mlp": mlp,
            "gate": mod.gate,
            "residual_x": x,
        }

    @staticmethod
    def _yak_single_post(block, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        output = block.linear2(torch.cat((mixed_attn_out, block.mlp_act(state["mlp"])), dim=2))
        return state["residual_x"] + state["gate"] * output

    def _flux2_flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

    def _flux2_video_single_io(self, block, x: torch.Tensor, pe: torch.Tensor, mod) -> dict:
        from flux2.model import apply_rope

        q, k, v, mlp, gate = block._qkv(x, mod)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._flux2_flatten_heads(q),
            "k": self._flux2_flatten_heads(k),
            "v": self._flux2_flatten_heads(v),
            "mlp": mlp,
            "gate": gate,
            "residual_x": x,
        }

    def _forward_flux2(
        self,
        embeds_all: Dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, object],
    ):
        if "video" not in self.mixtures or "action" not in self.mixtures:
            raise ValueError("FLUX.2 MoT requires `video` and `action` experts.")
        if not isinstance(attention_mask, dict):
            raise ValueError("FLUX.2 MoT expects attention_mask={'double_joint', 'single'}.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        video_state = embeds_all["video"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video embeds must be a dict with `txt` and `img` tensors.")
        txt = video_state["txt"]
        img = video_state["img"]
        action = embeds_all["action"]
        if not isinstance(action, torch.Tensor):
            raise ValueError("FLUX.2 action embeds must be a tensor.")

        video_freqs = freqs_all["video"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        action_ids = context_all["action"]["ids"]
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=img.device, dtype=img.dtype))
        video_t_mod = t_mod_all["video"]
        action_t_mod = t_mod_all["action"]

        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            v_block = video_expert.double_blocks[layer_idx]
            a_block = action_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = v_block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            from flux2.model import apply_rope

            q, k = apply_rope(q, k, pe_full)
            video_q = self._flux2_flatten_heads(q)
            video_k = self._flux2_flatten_heads(k)
            video_v = self._flux2_flatten_heads(v)
            action_state = a_block.prepare_qkv(action, action_pe, action_t_mod["double_img"])
            mixed = self._mixed_attention(
                torch.cat([video_q, action_state["q"]], dim=1),
                torch.cat([video_k, action_state["k"]], dim=1),
                torch.cat([video_v, action_state["v"]], dim=1),
                attention_mask["double_joint"],
            )
            video_attn, action_attn = torch.split(mixed, [txt.shape[1] + img.shape[1], action.shape[1]], dim=1)
            txt_attn, img_attn = torch.split(video_attn, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = v_block._apply_residuals(img, txt, img_attn, txt_attn, mods)
            action = a_block.apply_post(action_attn, action_state)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            v_block = video_expert.single_blocks[layer_idx]
            a_block = action_expert.single_blocks[layer_idx]
            video_state = self._flux2_video_single_io(v_block, video_stream, stream_pe, video_t_mod["single"])
            action_state = a_block.prepare_qkv(action, action_pe, action_t_mod["single"])
            mixed = self._mixed_attention(
                torch.cat([video_state["q"], action_state["q"]], dim=1),
                torch.cat([video_state["k"], action_state["k"]], dim=1),
                torch.cat([video_state["v"], action_state["v"]], dim=1),
                attention_mask["single"],
            )
            video_attn, action_attn = torch.split(mixed, [video_stream.shape[1], action.shape[1]], dim=1)
            video_stream = v_block._out(video_state["residual_x"], video_attn, video_state["mlp"], video_state["gate"])
            action = a_block.apply_post(action_attn, action_state)

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"video": {"txt": txt, "img": img}, "action": action}

    def prefill_flux2_video_cache(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        if self.block_protocol != "flux2":
            raise ValueError("`prefill_flux2_video_cache` requires block_protocol='flux2'.")
        video_expert = self.mixtures["video"]
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        from flux2.model import apply_rope

        double_cache = []
        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            flat_q = self._flux2_flatten_heads(q)
            flat_k = self._flux2_flatten_heads(k)
            flat_v = self._flux2_flatten_heads(v)
            double_cache.append({"k": flat_k, "v": flat_v})
            mixed = self._mixed_attention(flat_q, flat_k, flat_v, attention_mask["double_joint"])
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_cache = []
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            state = self._flux2_video_single_io(block, video_stream, stream_pe, video_t_mod["single"])
            single_cache.append({"k": state["k"], "v": state["v"]})
            mixed = self._mixed_attention(state["q"], state["k"], state["v"], attention_mask["single"])
            video_stream = block._out(state["residual_x"], mixed, state["mlp"], state["gate"])

        return {
            "double": double_cache,
            "single": single_cache,
            "txt_len": int(txt.shape[1]),
            "img_len": int(img.shape[1]),
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
        }

    def forward_flux2_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        action_t_mod: dict[str, object],
        video_kv_cache: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        video_seq_len: int,
    ) -> torch.Tensor:
        if self.block_protocol != "flux2":
            raise ValueError("`forward_flux2_action_with_video_cache` requires block_protocol='flux2'.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        action = action_tokens
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=action.device, dtype=action.dtype))
        action_seq_len = int(action.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len

        def _action_mask(mask: torch.Tensor) -> torch.Tensor:
            if mask.ndim == 2:
                return mask[video_seq_len:total_seq_len, :total_seq_len]
            if mask.ndim == 3:
                return mask[:, video_seq_len:total_seq_len, :total_seq_len]
            return mask[:, :, video_seq_len:total_seq_len, :total_seq_len]

        double_mask = _action_mask(attention_mask["double_joint"])
        capture = getattr(self, "action_attention_capture", None)
        for layer_idx, cache in enumerate(video_kv_cache["double"]):
            block = action_expert.double_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["double_img"])
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            if capture is not None and capture.should_capture(layer_idx):
                mixed, attn_probs = self._mixed_attention(
                    state["q"],
                    k_cat,
                    v_cat,
                    double_mask,
                    return_attn_probs=True,
                )
                capture.update(
                    attn_probs,
                    layer_idx=layer_idx,
                    block_type="flux2_double",
                    prefix_len=video_seq_len,
                    action_len=action_seq_len,
                )
            else:
                mixed = self._mixed_attention(state["q"], k_cat, v_cat, double_mask)
            action = block.apply_post(mixed, state)

        single_mask = _action_mask(attention_mask["single"])
        for layer_idx, cache in enumerate(video_kv_cache["single"]):
            block = action_expert.single_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["single"])
            global_layer_idx = int(len(video_kv_cache["double"])) + int(layer_idx)
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            if capture is not None and capture.should_capture(global_layer_idx):
                mixed, attn_probs = self._mixed_attention(
                    state["q"],
                    k_cat,
                    v_cat,
                    single_mask,
                    return_attn_probs=True,
                )
                capture.update(
                    attn_probs,
                    layer_idx=global_layer_idx,
                    block_type="flux2_single",
                    prefix_len=video_seq_len,
                    action_len=action_seq_len,
                )
            else:
                mixed = self._mixed_attention(state["q"], k_cat, v_cat, single_mask)
            action = block.apply_post(mixed, state)
        return action

    def _forward_yak(
        self,
        embeds_all: Dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        if "video" not in self.mixtures or "action" not in self.mixtures:
            raise ValueError("Yak MoT requires `video` and `action` experts.")
        if not isinstance(attention_mask, dict):
            raise ValueError("Yak MoT expects attention_mask={'joint_double', 'image_double', 'single'}.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        video_state = embeds_all["video"]
        if not isinstance(video_state, dict):
            raise ValueError("Yak video embeds must be a dict with `txt` and `img` tensors.")
        txt = video_state["txt"]
        img = video_state["img"]
        action = embeds_all["action"]
        if not isinstance(action, torch.Tensor):
            raise ValueError("Yak action embeds must be a tensor.")

        video_freqs = freqs_all["video"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        action_ids = context_all["action"]["ids"]
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=img.device, dtype=img.dtype))
        joint_pe = torch.cat([txt_pe, img_pe, action_pe], dim=2)
        image_pe = torch.cat([img_pe, action_pe], dim=2)
        single_pe = joint_pe

        video_vec = t_mod_all["video"]
        action_vec = t_mod_all["action"]
        double_layers = int(getattr(video_expert, "double_layers"))
        single_layers = int(getattr(video_expert, "single_layers"))

        for layer_idx in range(double_layers):
            v_block = video_expert.double_blocks[layer_idx]
            a_block = action_expert.double_blocks[layer_idx]

            img_mod1, img_mod2, img_mod3 = v_block.img_mod(video_vec)
            txt_mod1, txt_mod2 = v_block.txt_mod(video_vec)
            action_mod1, action_mod2, action_mod3 = a_block.img_mod(action_vec)

            img_norm = v_block.img_norm1(img)
            img_joint_in = (1 + img_mod1.scale) * img_norm + img_mod1.shift
            img_self_in = (1 + img_mod3.scale) * img_norm + img_mod3.shift
            txt_joint_in = (1 + txt_mod1.scale) * v_block.txt_norm1(txt) + txt_mod1.shift
            action_norm = a_block.img_norm1(action)
            action_joint_in = (1 + action_mod1.scale) * action_norm + action_mod1.shift
            action_self_in = (1 + action_mod3.scale) * action_norm + action_mod3.shift

            txt_q, txt_k, txt_v = self._yak_qkv(v_block.txt_attn, txt_joint_in, txt_pe)
            img_q, img_k, img_v = self._yak_qkv(v_block.img_attn, img_joint_in, img_pe)
            action_q, action_k, action_v = self._yak_qkv(a_block.img_attn, action_joint_in, action_pe)
            joint = self._mixed_attention(
                torch.cat([txt_q, img_q, action_q], dim=1),
                torch.cat([txt_k, img_k, action_k], dim=1),
                torch.cat([txt_v, img_v, action_v], dim=1),
                attention_mask["joint_double"],
            )
            txt_attn, img_attn, action_joint_attn = torch.split(
                joint,
                [txt.shape[1], img.shape[1], action.shape[1]],
                dim=1,
            )

            img_self_q, img_self_k, img_self_v = self._yak_qkv(v_block.img_self_attn, img_self_in, img_pe)
            action_self_q, action_self_k, action_self_v = self._yak_qkv(
                a_block.img_self_attn,
                action_self_in,
                action_pe,
            )
            image_self = self._mixed_attention(
                torch.cat([img_self_q, action_self_q], dim=1),
                torch.cat([img_self_k, action_self_k], dim=1),
                torch.cat([img_self_v, action_self_v], dim=1),
                attention_mask["image_double"],
            )
            img_self_attn, action_self_attn = torch.split(image_self, [img.shape[1], action.shape[1]], dim=1)

            img = img + img_mod1.gate * v_block.img_attn.proj(img_attn)
            img = img + img_mod3.gate * v_block.img_self_attn.proj(img_self_attn)
            img = img + img_mod2.gate * v_block.img_mlp((1 + img_mod2.scale) * v_block.img_norm2(img) + img_mod2.shift)
            txt = txt + txt_mod1.gate * v_block.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * v_block.txt_mlp((1 + txt_mod2.scale) * v_block.txt_norm2(txt) + txt_mod2.shift)

            action = action + action_mod1.gate * a_block.img_attn.proj(action_joint_attn)
            action = action + action_mod3.gate * a_block.img_self_attn.proj(action_self_attn)
            action = action + action_mod2.gate * a_block.img_mlp(
                (1 + action_mod2.scale) * a_block.img_norm2(action) + action_mod2.shift
            )

        video_stream = torch.cat([txt, img], dim=1)
        for layer_idx in range(single_layers):
            v_block = video_expert.single_blocks[layer_idx]
            a_block = action_expert.single_blocks[layer_idx]
            v_state = self._yak_single_io(v_block, video_stream, video_vec, torch.cat([txt_pe, img_pe], dim=2))
            a_state = self._yak_single_io(a_block, action, action_vec, action_pe)
            mixed = self._mixed_attention(
                torch.cat([v_state["q"], a_state["q"]], dim=1),
                torch.cat([v_state["k"], a_state["k"]], dim=1),
                torch.cat([v_state["v"], a_state["v"]], dim=1),
                attention_mask["single"],
            )
            video_attn, action_attn = torch.split(mixed, [video_stream.shape[1], action.shape[1]], dim=1)
            video_stream = self._yak_single_post(v_block, video_attn, v_state)
            action = self._yak_single_post(a_block, action_attn, a_state)

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"video": {"txt": txt, "img": img}, "action": action}

    def prefill_yak_video_cache(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        if self.block_protocol != "yak":
            raise ValueError("`prefill_yak_video_cache` requires block_protocol='yak'.")
        video_expert = self.mixtures["video"]
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        double_cache = []
        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            img_mod1, img_mod2, img_mod3 = block.img_mod(video_t_mod)
            txt_mod1, txt_mod2 = block.txt_mod(video_t_mod)
            img_norm = block.img_norm1(img)
            img_joint_in = (1 + img_mod1.scale) * img_norm + img_mod1.shift
            img_self_in = (1 + img_mod3.scale) * img_norm + img_mod3.shift
            txt_joint_in = (1 + txt_mod1.scale) * block.txt_norm1(txt) + txt_mod1.shift
            txt_q, txt_k, txt_v = self._yak_qkv(block.txt_attn, txt_joint_in, txt_pe)
            img_q, img_k, img_v = self._yak_qkv(block.img_attn, img_joint_in, img_pe)
            img_self_q, img_self_k, img_self_v = self._yak_qkv(block.img_self_attn, img_self_in, img_pe)
            double_cache.append(
                {
                    "joint_k": torch.cat([txt_k, img_k], dim=1),
                    "joint_v": torch.cat([txt_v, img_v], dim=1),
                    "image_k": img_self_k,
                    "image_v": img_self_v,
                }
            )
            joint = self._mixed_attention(
                torch.cat([txt_q, img_q], dim=1),
                torch.cat([txt_k, img_k], dim=1),
                torch.cat([txt_v, img_v], dim=1),
                attention_mask["joint_double"],
            )
            txt_attn, img_attn = torch.split(joint, [txt.shape[1], img.shape[1]], dim=1)
            img_self_attn = self._mixed_attention(img_self_q, img_self_k, img_self_v, attention_mask["image_double"])
            img = img + img_mod1.gate * block.img_attn.proj(img_attn)
            img = img + img_mod3.gate * block.img_self_attn.proj(img_self_attn)
            img = img + img_mod2.gate * block.img_mlp((1 + img_mod2.scale) * block.img_norm2(img) + img_mod2.shift)
            txt = txt + txt_mod1.gate * block.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * block.txt_mlp((1 + txt_mod2.scale) * block.txt_norm2(txt) + txt_mod2.shift)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_cache = []
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            state = self._yak_single_io(block, video_stream, video_t_mod, stream_pe)
            single_cache.append({"k": state["k"], "v": state["v"]})
            mixed = self._mixed_attention(state["q"], state["k"], state["v"], attention_mask["single"])
            video_stream = self._yak_single_post(block, mixed, state)
        return {
            "double": double_cache,
            "single": single_cache,
            "txt_len": int(txt.shape[1]),
            "img_len": int(img.shape[1]),
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
        }

    def forward_yak_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_kv_cache: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.block_protocol != "yak":
            raise ValueError("`forward_yak_action_with_video_cache` requires block_protocol='yak'.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        action = action_tokens
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=action.device, dtype=action.dtype))
        for layer_idx, cache in enumerate(video_kv_cache["double"]):
            block = action_expert.double_blocks[layer_idx]
            action_mod1, action_mod2, action_mod3 = block.img_mod(action_t_mod)
            action_norm = block.img_norm1(action)
            action_joint_in = (1 + action_mod1.scale) * action_norm + action_mod1.shift
            action_self_in = (1 + action_mod3.scale) * action_norm + action_mod3.shift
            action_q, action_k, action_v = self._yak_qkv(block.img_attn, action_joint_in, action_pe)
            joint = self._mixed_attention(
                action_q,
                torch.cat([cache["joint_k"], action_k], dim=1),
                torch.cat([cache["joint_v"], action_v], dim=1),
                attention_mask["joint_double"],
            )
            action_self_q, action_self_k, action_self_v = self._yak_qkv(block.img_self_attn, action_self_in, action_pe)
            image_self = self._mixed_attention(
                action_self_q,
                torch.cat([cache["image_k"], action_self_k], dim=1),
                torch.cat([cache["image_v"], action_self_v], dim=1),
                attention_mask["image_double"],
            )
            action = action + action_mod1.gate * block.img_attn.proj(joint)
            action = action + action_mod3.gate * block.img_self_attn.proj(image_self)
            action = action + action_mod2.gate * block.img_mlp(
                (1 + action_mod2.scale) * block.img_norm2(action) + action_mod2.shift
            )

        for layer_idx, cache in enumerate(video_kv_cache["single"]):
            block = action_expert.single_blocks[layer_idx]
            state = self._yak_single_io(block, action, action_t_mod, action_pe)
            mixed = self._mixed_attention(
                state["q"],
                torch.cat([cache["k"], state["k"]], dim=1),
                torch.cat([cache["v"], state["v"]], dim=1),
                attention_mask["single"],
            )
            action = self._yak_single_post(block, mixed, state)
        return action

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            built = self._build_expert_attention_io(expert, block, x, video_freqs, video_t_mod)
            mixed = self._mixed_attention(built["q"], built["k"], built["v"], video_attention_mask)
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=built["residual_x"],
                post_state=built["post_state"],
                use_gradient_checkpointing=built["use_gradient_checkpointing"],
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": built["k"], "v": built["v"]})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        if self.block_protocol == "yak":
            return self.forward_yak_action_with_video_cache(
                action_tokens=action_tokens,
                action_ids=action_context_payload["ids"],
                action_t_mod=action_t_mod,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
            )
        if self.block_protocol == "flux2":
            return self.forward_flux2_action_with_video_cache(
                action_tokens=action_tokens,
                action_ids=action_context_payload["ids"],
                action_t_mod=action_t_mod,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.ndim == 2:
            action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        elif attention_mask.ndim == 3:
            action_attention_mask = attention_mask[:, video_seq_len:total_seq_len, :total_seq_len]
        else:
            action_attention_mask = attention_mask[:, :, video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        capture = getattr(self, "action_attention_capture", None)
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            built = self._build_expert_attention_io(expert, block, x, action_freqs, action_t_mod)
            k_cat = torch.cat([video_kv_cache[layer_idx]["k"], built["k"]], dim=1)
            v_cat = torch.cat([video_kv_cache[layer_idx]["v"], built["v"]], dim=1)
            if capture is not None and capture.should_capture(layer_idx):
                mixed, attn_probs = self._mixed_attention(
                    built["q"],
                    k_cat,
                    v_cat,
                    action_attention_mask,
                    return_attn_probs=True,
                )
                capture.update(
                    attn_probs,
                    layer_idx=layer_idx,
                    block_type=self.block_protocol,
                    prefix_len=video_seq_len,
                    action_len=action_seq_len,
                )
            else:
                mixed = self._mixed_attention(built["q"], k_cat, v_cat, action_attention_mask)
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=built["residual_x"],
                post_state=built["post_state"],
                use_gradient_checkpointing=built["use_gradient_checkpointing"],
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        freqs_all: Dict[str, torch.Tensor] | None = None,
        context_all: Dict[str, Optional[dict]] | None = None,
        t_mod_all: Dict[str, torch.Tensor] | None = None,
    ):
        if self.block_protocol == "mage_flow":
            return self._forward_mage_flow(embeds_all, context_all or {})

        if self.block_protocol == "sana":
            return self._forward_sana(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
            )
        if self.block_protocol == "yak":
            return self._forward_yak(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
            )
        if self.block_protocol == "flux2":
            return self._forward_flux2(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
            )
        for key in self.expert_order:
            if key not in embeds_all or key not in freqs_all or key not in t_mod_all:
                raise ValueError(f"Missing MoT inputs for expert {key!r}")

        tokens_all = {k: v for k, v in embeds_all.items()}
        for layer_idx in range(self.num_layers):
            q_chunks, k_chunks, v_chunks, seq_lens = [], [], [], []
            cached = {}
            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                built = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=tokens_all[name],
                    freqs=freqs_all[name],
                    t_mod=t_mod_all[name],
                )
                q_chunks.append(built["q"])
                k_chunks.append(built["k"])
                v_chunks.append(built["v"])
                seq_lens.append(tokens_all[name].shape[1])
                cached[name] = {"block": block, **built}

            mixed = self._mixed_attention(
                q_cat=torch.cat(q_chunks, dim=1),
                k_cat=torch.cat(k_chunks, dim=1),
                v_cat=torch.cat(v_chunks, dim=1),
                attention_mask=attention_mask,
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                item = cached[name]
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=item["block"],
                    residual_x=item["residual_x"],
                    post_state=item["post_state"],
                    use_gradient_checkpointing=item["use_gradient_checkpointing"],
                    mixed_slice=mixed[:, start:end, :],
                    context_payload=context_all.get(name),
                )
                start = end
        return tokens_all

    @torch.no_grad()
    def prefill_mage_flow_video_cache(self, video_tokens, video_state, attention_mask):
        """Run the stable MageFlow text/reference prefix once for action inference."""
        if self.block_protocol != "mage_flow":
            raise ValueError("`prefill_mage_flow_video_cache` requires block_protocol='mage_flow'.")
        video_expert = self.mixtures["video"]
        vtr = video_expert.transformer
        text = video_state["packed_context"]
        vtemb, vrope = video_state["temb"], video_state["rope"]
        vcu, tcu = video_state["img_cu_lens"], video_state["txt_cu_lens"]
        if attention_mask.shape[-2:] != (
            int(text.shape[1] + video_tokens.shape[1]),
            int(text.shape[1] + video_tokens.shape[1]),
        ):
            raise ValueError("MageFlow prefix attention mask does not match video/text prefix.")

        cache = []
        for vblock in vtr.transformer_blocks:
            state = vblock.prepare_qkv(
                video_tokens, text, vtemb, vrope, tcu, vcu
            )
            mixed = self._mixed_attention(
                state["q"], state["k"], state["v"], attention_mask
            )
            text, video_tokens = vblock.apply_attention(mixed, state)
            cache.append({"k": state["k"], "v": state["v"]})
        return {
            "cache": cache,
            "text": text,
            "video": video_tokens,
            "video_state": video_state,
            "video_seq_len": int(text.shape[1] + video_tokens.shape[1]),
        }

    @torch.no_grad()
    def forward_mage_flow_action_with_video_cache(
        self, action_tokens, action_state, video_cache, attention_mask
    ):
        """Run one MageFlow action step against a cached text/reference prefix."""
        if self.block_protocol != "mage_flow":
            raise ValueError("`forward_mage_flow_action_with_video_cache` requires block_protocol='mage_flow'.")
        action_expert = self.mixtures["action"]
        atemb, arope = action_state["temb"], action_state["rope"]
        acu = action_state["cu_lens"]
        prefix_len = int(video_cache["video_seq_len"])
        total_len = prefix_len + int(action_tokens.shape[1])
        if attention_mask.shape[-2:] != (total_len, total_len):
            raise ValueError("MageFlow action attention mask does not match cached prefix.")
        action = action_tokens
        for layer_idx, ablock in enumerate(action_expert.blocks):
            state = ablock.prepare_qkv(action, None, atemb, arope, acu)
            mixed = self._mixed_attention(
                state["q"],
                torch.cat((video_cache["cache"][layer_idx]["k"], state["k"]), dim=1),
                torch.cat((video_cache["cache"][layer_idx]["v"], state["v"]), dim=1),
                attention_mask[:, prefix_len:total_len, :total_len],
            )
            action = ablock.apply_attention(mixed, state)
        return action

    def _forward_mage_flow(self, embeds_all, context_all):
        video_state = context_all.get("video") or {}
        action_state = context_all.get("action") or {}
        video = embeds_all["video"]
        action = embeds_all["action"]
        required = (
            video_state.get("packed_context"), video_state.get("temb"),
            video_state.get("rope"), video_state.get("img_cu_lens"),
            video_state.get("txt_cu_lens"), action_state.get("temb"),
            action_state.get("cu_lens"),
        )
        if any(value is None for value in required):
            raise ValueError("Mage-Flow MoT requires prepared video/action pre_dit states")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        vtr = video_expert.transformer
        text = video_state["packed_context"]
        vtemb, vrope = video_state["temb"], video_state["rope"]
        vcu, tcu = video_state["img_cu_lens"], video_state["txt_cu_lens"]
        atemb, arope = action_state["temb"], action_state["rope"]
        acu = action_state["cu_lens"]
        full_mask = context_all.get("mot_attention_mask")
        # The mixed sequence is packed as [text | video image | action].
        # `video` contains image tokens only; the packed text stream is kept
        # separately in `text` and must be included in the mask geometry.
        total_len = int(text.shape[1] + video.shape[1] + action.shape[1])

        # ---- Backend selection for the packed mixed attention. ----
        # sdpa: materialise the dense `[1, L, L]` bool mask (caller-built) and
        #   run SDPA (falls back to the memory-efficient backend at full O(L^2)).
        # flex: build a BlockMask + exact score_mod from the per-sample cu_lens
        #   and skip the ~96% all-zero block pairs. No dense mask needed.
        flex_block_mask = flex_sample_id = flex_is_target = flex_is_action = None
        if self.mage_flow_attn_backend == "flex":
            sample_id, block_type = _build_mage_flow_token_maps(
                tcu, vcu, acu,
                target_len=int(video_state["target_len"]),
                text_total=int(text.shape[1]),
                image_total=int(video.shape[1]),
                device=video.device,
            )
            flex_sample_id = sample_id
            flex_is_target = block_type == 1
            flex_is_action = block_type == 3
            try:
                from torch.nn.attention.flex_attention import create_block_mask
            except ImportError as exc:  # pragma: no cover - depends on torch build
                raise RuntimeError(
                    "MAGEFLOW_MIXED_ATTN=flex requires torch.nn.attention.flex_attention "
                    "(torch >= 2.5)."
                ) from exc
            # BlockMask is pure cross-sample isolation (sample_id[q]==sample_id[k]);
            # the within-sample 4x4 holes are applied by score_mod inside the
            # compiled flex body. Built once per forward, reused across all layers.
            _sid = sample_id

            def _mask_mod(b, h, q_idx, kv_idx):
                return _sid[q_idx] == _sid[kv_idx]

            flex_block_mask = create_block_mask(
                _mask_mod, B=1, H=1, Q_LEN=total_len, KV_LEN=total_len, device=video.device,
            )
        else:
            if full_mask is None:
                raise ValueError("Mage-Flow MoT requires a full packed mixed attention mask")
            if full_mask.ndim == 3 and full_mask.shape[0] != 1:
                raise ValueError(
                    "Mage-Flow packed mixed attention currently requires a single packed mask; "
                    f"got {tuple(full_mask.shape)}"
                )
            if full_mask.shape[-2:] != (total_len, total_len):
                raise ValueError(
                    "Mage-Flow packed mixed attention mask does not match joint streams: "
                    f"got {tuple(full_mask.shape)}, expected trailing shape {(total_len, total_len)}"
                )

        # The mid head is a training-only auxiliary (ReWorld Stage 1); its output
        # is unused at inference. Gate it on self.training so eval/inference
        # (model.eval(), @torch.no_grad) never builds or runs it — keeps eval
        # bit-identical and avoids a float32-head vs bf16-activation mismatch
        # that only autocast (present in training) would paper over.
        mid_enabled = self.training and video_expert.mid_layer_index is not None
        mid_layer_index = video_expert.mid_layer_index
        mid_hidden = None
        for idx, (vblock, ablock) in enumerate(zip(vtr.transformer_blocks, action_expert.blocks)):
            def layer_fn(video_in, text_in, action_in):
                v_state = vblock.prepare_qkv(
                    video_in, text_in, vtemb, vrope, tcu, vcu)
                a_state = ablock.prepare_qkv(
                    action_in, video_in, atemb, arope, acu
                )
                q = torch.cat((v_state["q"], a_state["q"]), dim=1)
                k = torch.cat((v_state["k"], a_state["k"]), dim=1)
                value = torch.cat((v_state["v"], a_state["v"]), dim=1)
                if flex_block_mask is not None:
                    mixed = self._flex_mixed_attention(
                        q, k, value, flex_block_mask, flex_sample_id, flex_is_target, flex_is_action)
                else:
                    mixed = self._mixed_attention(q, k, value, full_mask)
                v_len = v_state["text_len"] + v_state["image_len"]
                video_mixed, action_mixed = torch.split(
                    mixed, [v_len, a_state["q"].shape[1]], dim=1)
                text_out, video_out = vblock.apply_attention(video_mixed, v_state)
                action_out = ablock.apply_attention(action_mixed, a_state)
                return video_out, text_out, action_out
            if self.mot_checkpoint_mixed_attn and self.training:
                video, text, action = torch.utils.checkpoint.checkpoint(
                    layer_fn, video, text, action, use_reentrant=False)
            else:
                video, text, action = layer_fn(video, text, action)
            if mid_enabled and idx == mid_layer_index:
                mid_hidden = video

        norm_temb = vtemb[0] if isinstance(vtemb, tuple) else vtemb
        video = vtr.norm_out(video, norm_temb, cu_seqlens=vcu)
        video = vtr.proj_out(video).reshape(
            int(video_state["batch_size"]), -1, vtr.out_channels)
        action = action_expert.final_norm(action).reshape(
            int(action_state["batch_size"]), -1, action_expert.hidden_dim)
        out = {"video": video, "action": action}
        if mid_hidden is not None:
            # Auxiliary velocity prediction from the l-th video block (ReWorld
            # Eq.6). Same norm+proj parameterization and reshape as the main head.
            mid = video_expert.mid_norm(mid_hidden, norm_temb, cu_seqlens=vcu)
            mid = video_expert.mid_proj(mid).reshape(
                int(video_state["batch_size"]), -1, vtr.out_channels)
            out["mid_video"] = mid
        return out

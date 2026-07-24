from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class MageFlowActionBlock(nn.Module):
    """Mage image-stream block with a 1024-dim action bottleneck.

    The block keeps Mage's modulation, RMS/LayerNorm, attention and FeedForward
    structure. Its Q/K/V projections retain the video attention width so the
    action and video streams can share one mixed attention call.
    """

    def __init__(self, action_hidden_dim: int, video_hidden_dim: int,
                 num_heads: int, attn_head_dim: int):
        super().__init__()
        from .mage_flow_imports import ensure_mage_flow_importable
        ensure_mage_flow_importable()
        from diffusers.models.attention import FeedForward
        from mage_flow.models.modules.mage_layers import Attention

        self.action_hidden_dim = int(action_hidden_dim)
        self.video_hidden_dim = int(video_hidden_dim)
        self.num_heads = int(num_heads)
        self.attention_head_dim = int(attn_head_dim)
        # This is MageFlowTransformerBlock's image stream, kept at the
        # 1024-d action width except for QKV/output attention width. Text
        # stream modules are intentionally absent.
        self.block = nn.Module()
        self.block.img_mod = nn.Sequential(
            nn.SiLU(), nn.Linear(self.action_hidden_dim, 6 * self.action_hidden_dim, bias=True)
        )
        self.block.img_norm1 = nn.LayerNorm(self.action_hidden_dim, elementwise_affine=False, eps=1e-6)
        self.block.attn = Attention(
            query_dim=self.action_hidden_dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None,
            dim_head=self.attention_head_dim,
            heads=self.num_heads,
            out_dim=self.video_hidden_dim,
            bias=True,
            processor=None,
            eps=1e-6,
        )
        # Attention uses `out_dim` for both inner QKV width and output width;
        # keep QKV at the video width but project the mixed result back to the
        # action bottleneck. The added text-stream output is unused.
        self.block.attn.to_out[0] = nn.Linear(
            self.video_hidden_dim, self.action_hidden_dim, bias=True
        )
        self.block.attn.to_add_out = None
        self.block.img_norm2 = nn.LayerNorm(self.action_hidden_dim, elementwise_affine=False, eps=1e-6)
        self.block.img_mlp = FeedForward(
            dim=self.action_hidden_dim, dim_out=self.action_hidden_dim,
            activation_fn="gelu-approximate",
        )

    @staticmethod
    def _modulate(x, mod_params, cu_lens=None):
        """Apply Mage's shift/scale/gate modulation to packed action tokens."""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if cu_lens is None:
            return x * (1 + scale) + shift, gate
        if x.shape[0] != 1:
            raise ValueError("Packed Mage action tokens must have batch dimension 1")
        lengths = (cu_lens[1:] - cu_lens[:-1]).to(device=x.device, dtype=torch.long)
        shift = shift.repeat_interleave(lengths, dim=0)
        scale = scale.repeat_interleave(lengths, dim=0)
        gate = gate.repeat_interleave(lengths, dim=0)
        x = x.reshape(-1, x.shape[-1])
        x = (x * (1 + scale) + shift).reshape(1, -1, x.shape[-1])
        return x, gate

    def prepare_qkv(self, hidden_states, encoder_hidden_states, temb, action_cu_lens=None):
        del encoder_hidden_states
        if action_cu_lens is None:
            action_cu_lens = torch.tensor(
                [0, hidden_states.shape[1]], dtype=torch.int32, device=hidden_states.device)
        img_mod1, img_mod2 = self.block.img_mod(temb).chunk(2, dim=-1)
        img_modulated, img_gate1 = self._modulate(
            self.block.img_norm1(hidden_states), img_mod1,
            cu_lens=action_cu_lens,
        )
        attn = self.block.attn
        q = attn.to_q(img_modulated).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(img_modulated).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(img_modulated).unflatten(-1, (attn.heads, -1))
        q, k = attn.norm_q(q), attn.norm_k(k)
        return {
            "q": q.flatten(-2), "k": k.flatten(-2), "v": v.flatten(-2),
            "residual": hidden_states,
            "img_mod2": img_mod2,
            "img_gate1": img_gate1,
            "img_cu_lens": action_cu_lens,
        }

    def apply_attention(self, mixed_output, state):
        hidden = state["residual"] + state["img_gate1"] * self.block.attn.to_out[0](mixed_output)
        img_mod2, img_gate2 = self._modulate(
            self.block.img_norm2(hidden), state["img_mod2"], state["img_cu_lens"])
        return hidden + img_gate2 * self.block.img_mlp(img_mod2)


# Backward-compatible test/import name.
MageFlowSlimActionBlock = MageFlowActionBlock


class MageFlowActionDiT(nn.Module):
    """Slim Mage action expert; attention geometry follows the video DiT."""

    block_protocol = "mage_flow"

    def __init__(self, action_dim: int, video_hidden_dim: int, num_heads: int,
                 attn_head_dim: int, depth: int, action_hidden_dim: int = 1024,
                 max_action_horizon: int = 64):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(action_hidden_dim)
        self.video_hidden_dim = int(video_hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = int(attn_head_dim)
        self.max_action_horizon = int(max_action_horizon)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            MageFlowActionBlock(self.hidden_dim, self.video_hidden_dim,
                                    self.num_heads, self.attn_head_dim)
            for _ in range(int(depth))
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.action_decoder = nn.Linear(self.hidden_dim, self.action_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(256, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    @classmethod
    def from_video_transformer(cls, transformer: nn.Module, action_dim: int,
                               action_hidden_dim: int = 1024,
                               max_action_horizon: int = 64,
                               pretrained_path: str | None = None,
                               device: str = "cuda",
                               torch_dtype: torch.dtype = torch.bfloat16):
        model = cls(action_dim, int(transformer.inner_dim),
                    int(transformer.num_attention_heads),
                    int(transformer.attention_head_dim),
                    len(transformer.transformer_blocks), action_hidden_dim,
                    max_action_horizon).to(device=device, dtype=torch_dtype)
        if pretrained_path:
            payload = torch.load(pretrained_path, map_location="cpu")
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            load_result = model.load_state_dict(state, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                print(
                    "[MageFlowActionDiT] checkpoint loaded with key mismatches: "
                    f"missing_keys={load_result.missing_keys}, "
                    f"unexpected_keys={load_result.unexpected_keys}, "
                    f"checkpoint={pretrained_path}",
                    flush=True,
                )
        return model

    @staticmethod
    def _resize(src: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        if tuple(src.shape) == shape:
            return src
        value = src.float()
        while value.ndim < len(shape):
            value = value.unsqueeze(0)
        while value.ndim > len(shape):
            value = value.squeeze(0)
        for dim, size in enumerate(shape):
            if value.shape[dim] == size:
                continue
            order = [i for i in range(value.ndim) if i != dim] + [dim]
            inverse = [0] * value.ndim
            for i, item in enumerate(order):
                inverse[item] = i
            value = F.interpolate(
                value.permute(*order).reshape(-1, 1, value.shape[dim]),
                size=size, mode="linear", align_corners=True,
            ).reshape(*[value.shape[i] for i in order[:-1]], size).permute(*inverse)
        return value.to(dtype=src.dtype)

    @classmethod
    def initialize_from_video(cls, video_state: dict[str, torch.Tensor], action_model: nn.Module):
        """Flux-style initialization from a Mage video state dict."""
        target = action_model.state_dict()
        copied = resized = 0
        out = dict(target)
        for key, dst in target.items():
            if key.startswith(("action_encoder.", "action_decoder.")):
                continue
            candidates = []
            if key.startswith("blocks.") and ".block." in key:
                prefix, suffix = key.split(".block.", 1)
                layer = prefix.split(".")[1]
                # Copy Mage's image stream verbatim; action uses a narrower
                # hidden state, so only shape-changing projections are resized.
                candidates.append(f"transformer_blocks.{layer}.{suffix}")
            elif key.startswith("time_embed."):
                candidates.append(key.replace("time_embed.", "time_text_embed.timestep_embedder.linear_1."))
            for source_key in candidates:
                if source_key in video_state:
                    value = video_state[source_key]
                    if tuple(value.shape) != tuple(dst.shape):
                        value = cls._resize(value, tuple(dst.shape)); resized += 1
                    else:
                        copied += 1
                    out[key] = value.to(dtype=dst.dtype, device=dst.device)
                    break
        action_model.load_state_dict(out, strict=True)
        return {"copied": copied, "resized": resized, "total": len(target)}

    def pre_dit(self, action_tokens: torch.Tensor, timestep: torch.Tensor, **_: Any):
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action tokens [B,T,{self.action_dim}]")
        if action_tokens.shape[1] > self.max_action_horizon:
            raise ValueError(
                f"Action length {action_tokens.shape[1]} exceeds "
                f"max_action_horizon={self.max_action_horizon}"
            )
        from .mage_flow_imports import ensure_mage_flow_importable
        ensure_mage_flow_importable()
        from mage_flow.models.modules.mage_layers import get_timestep_embedding
        batch, length = action_tokens.shape[:2]
        encoded = self.action_encoder(action_tokens)
        temb = get_timestep_embedding(
            timestep.to(dtype=torch.float32), 256,
            flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000,
        ).to(dtype=encoded.dtype)
        temb = self.time_embed(temb)
        cu = torch.arange(0, (batch + 1) * length, length,
                          device=encoded.device, dtype=torch.int32)
        return {
            "tokens": encoded.reshape(1, -1, encoded.shape[-1]),
            "temb": temb,
            "cu_lens": cu,
            "batch_size": int(batch),
            "length": int(length),
            "timestep": timestep,
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]):
        return self.action_decoder(self.final_norm(tokens))

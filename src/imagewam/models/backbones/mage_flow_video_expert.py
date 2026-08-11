from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from PIL import Image

from .mage_flow_imports import ensure_mage_flow_importable


class MageFlowVideoExpert(nn.Module):
    """Native Mage-Flow transformer adapter used by ImageWAM."""

    block_protocol = "mage_flow"

    def __init__(self, model: nn.Module, model_path: str, load_text_encoder: bool = True,
                 per_segment_temb: bool = False, mid_layer_index: int | None = None):
        super().__init__()
        self.model = model
        self.transformer = getattr(model, "transformer", None)
        self.vae = getattr(model, "vae", None)
        self.txt_enc = model.txt_enc if load_text_encoder else None
        if not load_text_encoder:
            model.txt_enc = None
        self.model_path = model_path
        self.hidden_dim = int(self.transformer.inner_dim) if self.transformer is not None else 0
        self.num_heads = int(self.transformer.num_attention_heads) if self.transformer is not None else 0
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = int(self.transformer.attention_head_dim) if self.transformer is not None else 0
        self.blocks = self.transformer.transformer_blocks if self.transformer is not None else nn.ModuleList()
        self.double_layers = len(self.blocks)
        self.single_layers = 0
        self.vae_downsample_rate = 16
        self.caption_dim = int(self.transformer.txt_in.in_features) if self.transformer is not None else 0
        self.load_text_encoder = bool(load_text_encoder)
        # When enabled, reclass the vendored video blocks in place to a thin
        # subclass that modulates the reference/text stream with a constant t=0
        # temb (only the target slice uses the live tv temb). Parameters and
        # state-dict keys are unchanged; see
        # ``mage_flow_video_block_per_segment`` for the rationale.
        self.per_segment_temb = bool(per_segment_temb)
        if self.per_segment_temb and self.transformer is not None:
            from .mage_flow_video_block_per_segment import make_per_segment_video_block_class
            base_cls = type(self.transformer.transformer_blocks[0])
            sub_cls = make_per_segment_video_block_class(base_cls)
            for block in self.transformer.transformer_blocks:
                block.__class__ = sub_cls

        # ReWorld-style future-predictive intermediate supervision (Stage 1):
        # a lightweight auxiliary head on the output of the l-th video block that
        # regresses the same rectified-flow velocity target as the main head.
        # `mid_layer_index=None` disables it (no extra params, no overhead).
        # See [[reworld-mid-loss]] / ReWorld §3.3 Eq.(6)-(8).
        self.mid_layer_index = None
        if mid_layer_index is not None and self.transformer is not None:
            depth = self.double_layers
            idx = int(mid_layer_index)
            if not (0 <= idx < depth):
                raise ValueError(
                    f"mid_layer_index {idx} is out of range [0, {depth}) for this "
                    f"{depth}-layer Video DiT; pick roughly depth // 3.")
            self.mid_layer_index = idx
            from mage_flow.models.modules.mage_layers import AdaLayerNormContinuous
            inner_dim = int(self.transformer.inner_dim)
            patch = int(self.transformer.patch_size)
            out_ch = int(self.transformer.out_channels)
            # Mirror the main output head (norm_out + proj_out) so the auxiliary
            # target lives in the same modulation/parameterization as the final
            # velocity prediction.
            self.mid_norm = AdaLayerNormContinuous(
                inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
            self.mid_proj = nn.Linear(inner_dim, patch * patch * out_ch, bias=True)

    def compute_vae_encodings(self, pixel_values, with_ids: bool = True):
        return self.model.compute_vae_encodings(pixel_values, with_ids=with_ids)

    @torch.no_grad()
    def encode_edit_conditions(self, instructions, references, device=None):
        """Encode Mage edit text plus reference images for uncached training."""
        if self.txt_enc is None:
            raise RuntimeError("Mage text encoder is disabled; provide cached context/context_mask.")
        ensure_mage_flow_importable()
        from mage_flow.pipeline import _edit_prompt_body, _template_info, _lens_to_cu

        device = torch.device(device or self.device)
        info = _template_info("mage-flow-edit")
        template = info.get("template", "{}")
        drop_idx = int(info.get("start_idx", 0))
        processor = self.txt_enc.processor
        ids_list, pixel_values, grids = [], [], []
        for instruction, refs in zip(instructions, references, strict=True):
            if isinstance(refs, Image.Image):
                refs = [refs]
            prompt = template.format(_edit_prompt_body(str(instruction), len(refs)))
            encoded = processor(text=[prompt], images=list(refs), padding=True, return_tensors="pt")
            ids_list.append(encoded["input_ids"].squeeze(0))
            if encoded.get("pixel_values") is not None:
                pixel_values.append(encoded["pixel_values"])
                grids.append(encoded["image_grid_thw"])
        input_ids = torch.cat(ids_list).to(device)
        cu = _lens_to_cu([int(x.numel()) for x in ids_list], device)
        inputs = {"input_ids": input_ids, "cu_seqlens": cu}
        if pixel_values:
            inputs["pixel_values"] = torch.cat(pixel_values).to(device)
            inputs["image_grid_thw"] = torch.cat(grids).to(device)
        result = self.txt_enc(input_ids, cu, inputs=inputs, drop_idx_override=drop_idx)
        lengths = result["txt_seq_lens"].tolist()
        max_len = max(lengths)
        hidden = result["txt"].new_zeros(len(lengths), max_len, result["txt"].shape[-1])
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=hidden.device)
        offset = 0
        for row, length in enumerate(lengths):
            hidden[row, :length] = result["txt"][offset:offset + length]
            mask[row, :length] = True
            offset += length
        return hidden.to(device=self.device, dtype=self.torch_dtype), mask.to(self.device)

    @classmethod
    def from_pretrained(cls, model_path: str, mage_flow_src_path: str | None = None,
                        device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16,
                        load_text_encoder: bool = True, text_encoder_only: bool = False,
                        per_segment_temb: bool = False, mid_layer_index: int | None = None):
        ensure_mage_flow_importable(mage_flow_src_path)
        from mage_flow.pipeline import load_from_repo

        if text_encoder_only:
            model = load_from_repo(model_path, device="cpu", load_text_encoder=True)
            model.transformer = None
            model.vae = None
            model.txt_enc.to(device=device, dtype=torch_dtype).eval()
        elif load_text_encoder:
            model = load_from_repo(model_path, device="cpu", load_text_encoder=True)
            model.to(device=device, dtype=torch_dtype).eval()
        else:
            model = load_from_repo(model_path, device="cpu", load_text_encoder=False)
            model.transformer.to(device=device, dtype=torch_dtype)
            if model.vae is not None:
                model.vae.to(device=device, dtype=torch_dtype)
            model.eval()
        return cls(model, model_path, load_text_encoder=load_text_encoder,
                   per_segment_temb=per_segment_temb, mid_layer_index=mid_layer_index)

    def encode_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(image.to(device=self.device, dtype=self.torch_dtype))

    def decode_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents.to(device=self.device, dtype=self.torch_dtype))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def torch_dtype(self):
        return next(self.parameters()).dtype

    def pre_dit(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor | None = None,
                ref_image_hidden_states: torch.Tensor | None = None,
                img_shapes=None,
                **_: Any) -> dict[str, Any]:
        if x.ndim != 3 or x.shape[-1] != int(self.transformer.in_channels):
            raise ValueError(f"Mage image tokens must be [B,N,{self.transformer.in_channels}]")
        batch_size, target_len = int(x.shape[0]), int(x.shape[1])
        ref_len = 0 if ref_image_hidden_states is None else int(ref_image_hidden_states.shape[1])
        if ref_image_hidden_states is not None:
            # Mage-Flow-Edit packs the denoised target before clean references.
            x = torch.cat([x, ref_image_hidden_states], dim=1)
        if context.ndim != 3:
            raise ValueError("Mage text context must be [B,L,D]")
        if context_mask is None and context.shape[0] > 1:
            raise ValueError(
                "Packed MageFlow text requires context_mask for batch_size > 1."
            )
        transformer = self.transformer
        img = transformer.img_in(x)
        if context_mask is None:
            lengths = [int(context.shape[1])] * int(context.shape[0])
            txt = transformer.txt_in(transformer.txt_norm(context))
        else:
            lengths = context_mask.to(dtype=torch.bool).sum(dim=1).tolist()
            txt_rows = [context[i, :int(length)] for i, length in enumerate(lengths)]
            packed_context = torch.cat(txt_rows, dim=0)
            txt = transformer.txt_in(transformer.txt_norm(packed_context.unsqueeze(0)))
        temb = transformer.time_text_embed(timestep.to(img.dtype), img)
        if self.per_segment_temb:
            temb_zero = transformer.time_text_embed(
                torch.zeros_like(timestep).to(img.dtype), img)
            temb = (temb, temb_zero, int(target_len))
        rope_shapes = img_shapes
        rope_skip = 0
        if img_shapes and img_shapes[0] and int(img_shapes[0][0][0]) == 0:
            # MageFlow's RoPE implementation cannot reshape a zero-frame
            # segment. Generate the omitted target segment as a temporary
            # frame, then retain only the reference frequencies (frame 1).
            zero_frame = img_shapes[0][0]
            rope_shapes = [[(1, int(zero_frame[1]), int(zero_frame[2])), *img_shapes[0][1:]]]
            rope_skip = int(zero_frame[1]) * int(zero_frame[2])
        rope = transformer.pos_embed(rope_shapes, device=img.device)
        if rope_skip:
            rope = rope[rope_skip:]
        img_cu = torch.arange(0, (img.shape[0] + 1) * img.shape[1], img.shape[1],
                              device=img.device, dtype=torch.int32)
        txt_cu = torch.tensor([0] + [sum(lengths[:i + 1]) for i in range(len(lengths))],
                              device=img.device, dtype=torch.int32)
        return {
            "tokens": img.reshape(1, -1, img.shape[-1]),
            "context": context,
            "packed_context": txt,
            "context_mask": context_mask,
            "temb": temb,
            "rope": rope,
            "img_cu_lens": img_cu,
            "txt_cu_lens": txt_cu,
            "batch_size": int(context.shape[0]),
            "target_len": target_len,
            "ref_len": ref_len,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        # The target is the first image segment in Mage-Flow-Edit's layout.
        return tokens[:, :int(pre_state["target_len"])]

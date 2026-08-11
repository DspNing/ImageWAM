"""Per-segment timestep modulation for the MageFlow video block.

Background
----------
The MageFlow video DiT packs the denoised *target* and the clean *reference*
into one image-token stream and modulates the whole stream with a single temb
derived from the shared video timestep ``tv``. At inference, the deployment
fast path freezes this modulation once (single ``tv``) and reuses the cached
reference K/V at every action step. Because the reference representation was
trained to be ``tv``-dependent, any single frozen ``tv`` is off-manifold and
the action flow trajectory diverges (verified: frozen cache ``mean`` diff
``0.175`` / spikes ``2.0`` vs per-step cache ``0.042``).

This module decouples the two streams: the *target* slice keeps the live ``tv``
temb, while the *reference* (and text) slice is modulated by a constant ``t=0``
temb. With the inference prefill also run at ``t=0`` (and ``target_len=0``), the
frozen reference cache becomes train/test-consistent.

Implementation
--------------
We do NOT edit the vendored ``mage_flow`` block. Instead
``make_per_segment_video_block_class`` subclasses the concrete vendored block
class and overrides only ``prepare_qkv`` / ``apply_attention`` (the two methods
the MoT calls directly; the block's own ``forward`` is never used). A
flag-controlled reclass (``block.__class__ = SubClass``) keeps parameters,
submodules and state-dict keys byte-identical to the original, so no checkpoint
migration is needed.

The t=0 signal is delivered without touching ``mot.py``: ``pre_dit`` emits
``temb`` as a tuple ``(temb_target, temb_cond, target_len)`` and the MoT passes
that object through opaquely to ``prepare_qkv``.
"""
from __future__ import annotations

import torch


def make_per_segment_video_block_class(base_block_cls):
    """Build a per-segment-temb subclass of the vendored video block class.

    Only method resolution changes; no new parameters or attributes are added,
    so existing block instances can be reclassed in place.
    """

    class PerSegmentVideoBlock(base_block_cls):  # type: ignore[misc, valid-type]
        # NOTE: ``_modulate`` is inherited from the vendored block and reused
        # verbatim for the (homogeneous) text stream and for the flag-off path.

        def _modulate_segmented(self, x, mod_params_tgt, mod_params_cond, target_len):
            """adaLN with two parameter sets inside each image segment.

            Tokens ``[0:target_len]`` of every segment use the *target* params
            (live ``tv``); the remaining reference tokens use the *cond* params
            (``t=0``). Image segments are uniform length (target+ref is constant
            within a batch).

            Compile-friendly: never calls ``Tensor.item()`` (a dynamo graph-break
            point). The segment length is inferred via ``reshape(batch, -1, dim)``,
            the target slice uses the static python int ``target_len``, and the
            reference slice size is left symbolic for broadcasting/``expand_as``.

            Mirrors the vendored ``_modulate`` return contract: ``(x_out, gate)``
            where ``x_out`` keeps ``x``'s shape and ``gate`` is per-token.
            """
            shift_t, scale_t, gate_t = mod_params_tgt.chunk(3, dim=-1)
            shift_c, scale_c, gate_c = mod_params_cond.chunk(3, dim=-1)
            batch = shift_t.shape[0]
            dim = shift_t.shape[-1]
            tl = int(target_len)  # python int from pre_dit -> static, no graph break
            # x is [1, B*seg, D]; reshape to [B, seg, D] (seg inferred by -1).
            x3 = x.reshape(batch, -1, dim)
            tgt = x3[:, :tl, :]
            ref = x3[:, tl:, :]

            def seg_params(p_tgt, p_cond):
                # [B, D] -> [B, seg, D]: broadcast over each slice then concat.
                return torch.cat(
                    [p_tgt.unsqueeze(1).expand_as(tgt),
                     p_cond.unsqueeze(1).expand_as(ref)],
                    dim=1,
                )

            shift = seg_params(shift_t, shift_c)
            scale = seg_params(scale_t, scale_c)
            gate = seg_params(gate_t, gate_c)
            out3 = x3 * (1.0 + scale) + shift            # [B, seg, D]
            return out3.reshape(x.shape), gate.reshape(-1, dim)

        def prepare_qkv(self, hidden_states, encoder_hidden_states, temb,
                        image_rotary_emb, txt_cu_lens, img_cu_lens):
            from mage_flow.models.modules.mage_layers import apply_rotary_emb_mageflow

            if isinstance(temb, tuple):
                temb_target, temb_cond, target_len = temb
                img_mod1_t, img_mod2_t = self.img_mod(temb_target).chunk(2, dim=-1)
                img_mod1_c, img_mod2_c = self.img_mod(temb_cond).chunk(2, dim=-1)
                img_modulated, img_gate1 = self._modulate_segmented(
                    self.img_norm1(hidden_states), img_mod1_t, img_mod1_c,
                    target_len,
                )
                # Text is a fixed condition: modulate it with the t=0 temb too,
                # since its K/V are frozen alongside the reference at inference.
                txt_mod1, txt_mod2 = self.txt_mod(temb_cond).chunk(2, dim=-1)
                img_mod2_tgt = img_mod2_t
                img_mod2_cond = img_mod2_c
            else:
                img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
                txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)
                img_modulated, img_gate1 = self._modulate(
                    self.img_norm1(hidden_states), img_mod1, cu_lens=img_cu_lens)
                img_mod2_tgt = img_mod2
                img_mod2_cond = None
                target_len = 0

            txt_modulated, txt_gate1 = self._modulate(
                self.txt_norm1(encoder_hidden_states), txt_mod1, cu_lens=txt_cu_lens)

            attn = self.attn
            iq = attn.to_q(img_modulated).unflatten(-1, (attn.heads, -1))
            ik = attn.to_k(img_modulated).unflatten(-1, (attn.heads, -1))
            iv = attn.to_v(img_modulated).unflatten(-1, (attn.heads, -1))
            tq = attn.add_q_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
            tk = attn.add_k_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
            tv = attn.add_v_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
            iq, ik = attn.norm_q(iq), attn.norm_k(ik)
            tq, tk = attn.norm_added_q(tq), attn.norm_added_k(tk)
            iq = apply_rotary_emb_mageflow(iq, image_rotary_emb)
            ik = apply_rotary_emb_mageflow(ik, image_rotary_emb)
            return {
                "q": torch.cat((tq, iq), dim=1).flatten(-2),
                "k": torch.cat((tk, ik), dim=1).flatten(-2),
                "v": torch.cat((tv, iv), dim=1).flatten(-2),
                "text_len": int(encoder_hidden_states.shape[1]),
                "image_len": int(hidden_states.shape[1]),
                "text_residual": encoder_hidden_states,
                "image_residual": hidden_states,
                "text_gate": txt_gate1,
                "image_gate": img_gate1,
                "img_mod2_tgt": img_mod2_tgt,
                "img_mod2_cond": img_mod2_cond,
                "target_len": int(target_len),
                "txt_mod2": txt_mod2,
                "img_cu_lens": img_cu_lens,
                "txt_cu_lens": txt_cu_lens,
            }

        def apply_attention(self, mixed_output, state):
            text_len = state["text_len"]
            txt_out, img_out = mixed_output[:, :text_len], mixed_output[:, text_len:]
            img = state["image_residual"] + state["image_gate"] * self.attn.to_out[0](img_out)
            txt = state["text_residual"] + state["text_gate"] * self.attn.to_add_out(txt_out)
            if state.get("img_mod2_cond") is not None:
                img_mod2, img_gate2 = self._modulate_segmented(
                    self.img_norm2(img), state["img_mod2_tgt"], state["img_mod2_cond"],
                    state["target_len"],
                )
            else:
                img_mod2, img_gate2 = self._modulate(
                    self.img_norm2(img), state["img_mod2_tgt"], state["img_cu_lens"])
            txt_mod2, txt_gate2 = self._modulate(
                self.txt_norm2(txt), state["txt_mod2"], state["txt_cu_lens"])
            img = img + img_gate2 * self.img_mlp(img_mod2)
            txt = txt + txt_gate2 * self.txt_mlp(txt_mod2)
            return txt, img

    PerSegmentVideoBlock.__name__ = "PerSegmentVideoBlock"
    PerSegmentVideoBlock.__qualname__ = "PerSegmentVideoBlock"
    return PerSegmentVideoBlock

#!/usr/bin/env python3
"""Precompute Qwen3 text embeddings for LIBERO evaluation benchmarks.

Unlike ``precompute_flux2_qwen3_embeds.py`` (which scans dataset
``meta/tasks.jsonl`` for training prompts), this script extracts every
unique task description from the LIBERO benchmark suite, encodes them once
with the Qwen3 text encoder, and writes them to a shared cache directory.
The eval scripts can then load these embeddings and pass
``context/context_mask`` to ``model.infer_action()`` instead of
``prompt``, avoiding repeated Qwen3 forward passes and the ~2-4 GB
VRAM cost of loading the text encoder.

Usage
-----
# LIBERO-Plus (standard suites):
python scripts/flux2/precompute_libero_qwen3_embeds.py \\
    task=libero_flux2_klein_4b_base_imagewam \\
    EVALUATION.output_dir=./data/text_embeds_cache/eval_libero_plus

# Overwrite existing cache:
python scripts/flux2/precompute_libero_qwen3_embeds.py \\
    task=libero_flux2_klein_4b_base_imagewam \\
    EVALUATION.output_dir=./data/text_embeds_cache/eval_libero_plus \\
    overwrite=false

Cache file format (matches training cache):
    {sha256(prompt)}.qwen3_flux2_len{context_len}.pt
    -> {"text_hidden_states": [B, L, D], "text_attention_mask": [B, L]}
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
from einops import rearrange
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from imagewam.models.backbones.flux2_imports import ensure_flux2_importable  # noqa: E402
from imagewam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from imagewam.utils.logging_config import get_logger, setup_logging  # noqa: E402
from scripts.flux2.precompute_flux2_qwen3_embeds import _collect_dataset_settings  # noqa: E402

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_CONTEXT_LEN = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_QWEN3_MODEL_SPEC = "Qwen/Qwen3-4B"

# Must match flux2.text_encoder.OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]


def _atomic_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    """Atomically save a torch.Tensor dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _collect_libero_prompts() -> list[str]:
    """Extract unique prompts from all LIBERO benchmark suites."""
    from libero.libero import benchmark

    prompts: list[str] = []
    seen: set[str] = set()
    suite_counts: dict[str, int] = {}

    benchmark_dict = benchmark.get_benchmark_dict()
    known_broken = {"libero_100"}
    for suite_name in sorted(benchmark_dict.keys()):
        if suite_name in known_broken:
            logger.info("Skipping known-broken suite: %s", suite_name)
            continue
        try:
            suite = benchmark_dict[suite_name]()
        except KeyError as exc:
            logger.warning("Skipping suite '%s': %s", suite_name, exc)
            continue
        n_tasks = int(suite.n_tasks)
        suite_counts[suite_name] = n_tasks
        for tid in range(n_tasks):
            task = suite.get_task(tid)
            task_desc = task.language
            prompt = DEFAULT_PROMPT.format(task=task_desc)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)

    total = sum(suite_counts.values())
    logger.info(
        "LIBERO: %d suites, %d total tasks -> %d unique prompts",
        len(suite_counts), total, len(prompts),
    )
    for name, cnt in suite_counts.items():
        logger.info("  %s: %d tasks", name, cnt)

    return prompts


def _encode_prompts(
    prompts: list[str],
    cache_dir: Path,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str,
    torch_dtype: torch.dtype,
    context_len: int,
    batch_size: int,
    overwrite: bool,
) -> dict[str, int]:
    """Encode prompts and write per-prompt cache files."""
    stats = {"new": 0, "overwrite": 0, "skip": 0, "error": 0}
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Pre-check which prompts need encoding
    if not overwrite:
        to_encode = []
        for prompt in prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = cache_dir / f"{hashed}.qwen3_flux2_len{context_len}.pt"
            if cache_path.exists():
                stats["skip"] += 1
            else:
                to_encode.append(prompt)
        prompts = to_encode
        logger.info("overwrite=false: %d prompts already cached, %d to encode", stats["skip"], len(prompts))

    if not prompts:
        logger.info("Nothing to encode.")
        return stats

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc="Encoding", unit="batch"):
            batch_prompts = prompts[start : start + batch_size]
            try:
                messages = [
                    [{"role": "user", "content": p}]
                    for p in batch_prompts
                ]
                rendered = [
                    tokenizer.apply_chat_template(
                        msg,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for msg in messages
                ]
            except TypeError:
                # Older tokenizer versions lack enable_thinking
                rendered = [
                    tokenizer.apply_chat_template(
                        msg,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for msg in messages
                ]

            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=context_len,
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
            hidden = rearrange(hidden, "b c l d -> b l (c d)")

            for i, prompt in enumerate(batch_prompts):
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                cache_path = cache_dir / f"{hashed}.qwen3_flux2_len{context_len}.pt"

                if cache_path.exists() and not overwrite:
                    stats["skip"] += 1
                    continue

                if cache_path.exists():
                    stats["overwrite"] += 1
                else:
                    stats["new"] += 1

                payload = {
                    "text_hidden_states": hidden[i : i + 1].detach().to(device="cpu", dtype=torch.bfloat16),
                    "text_attention_mask": attention_mask[i : i + 1].to(dtype=torch.bool, device="cpu"),
                }
                _atomic_save(payload, cache_path)

    logger.info("Finished encoding: new=%d overwrite=%d skip=%d", stats["new"], stats["overwrite"], stats["skip"])
    return stats


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)

    overwrite = bool(cfg.get("overwrite", True))
    model_cfg = cfg.model
    if model_cfg is None:
        raise ValueError("`cfg.model` is required.")

    # Resolve output cache directory
    output_dir_raw = cfg.get("EVALUATION", {}).get("output_dir", "./data/text_embeds_cache/eval_libero_plus")
    cache_dir = Path(os.path.expanduser(str(output_dir_raw)))

    _, _, context_lens, _ = _collect_dataset_settings(cfg.data)
    if len(context_lens) != 1:
        raise ValueError(f"Expected one qwen_context_len, got {sorted(context_lens)}")
    context_len = next(iter(context_lens))
    logger.info("Inferred context_len=%d from data config", context_len)

    # Resolve Qwen3 model spec
    qwen3_model_spec = str(cfg.get("flux2_qwen3_model_spec") or cfg.model.get("qwen3_model_spec", DEFAULT_QWEN3_MODEL_SPEC))

    # Batch size
    batch_size = int(cfg.get("batch_size", DEFAULT_BATCH_SIZE))

    # Collect prompts from LIBERO benchmark
    prompts = _collect_libero_prompts()
    if not prompts:
        logger.warning("No prompts collected; nothing to do.")
        return

    # Resolve device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info(
        "Precomputing LIBERO Qwen3 text embeddings: cache=%s, model=%s, ctx_len=%d, batch=%d, overwrite=%s",
        cache_dir, qwen3_model_spec, context_len, batch_size, overwrite,
    )

    # Ensure FLUX.2 importable (needed for OUTPUT_LAYERS_QWEN3 resolver)
    flux2_src = cfg.model.get("flux2_src_path") or os.environ.get("FLUX2_SRC")
    if flux2_src:
        ensure_flux2_importable(str(flux2_src))

    # Load Qwen3 model
    tokenizer = AutoTokenizer.from_pretrained(qwen3_model_spec)
    model = AutoModelForCausalLM.from_pretrained(
        qwen3_model_spec,
        torch_dtype=torch_dtype,
    ).to(device).eval()

    # Encode and save
    stats = _encode_prompts(
        prompts=prompts,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        model=model,
        device=device,
        torch_dtype=torch_dtype,
        context_len=context_len,
        batch_size=batch_size,
        overwrite=overwrite,
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    main()

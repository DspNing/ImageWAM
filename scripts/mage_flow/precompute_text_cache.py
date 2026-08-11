#!/usr/bin/env python3
"""Precompute Mage-Flow Qwen3-VL edit contexts for RobotVideoDataset.

Each cache file contains one sample's padded context and mask.  The reference
image is encoded together with the instruction because Mage-Flow uses a
multimodal Qwen3-VL condition, unlike the text-only Flux cache.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Subset

from imagewam.models.backbones.mage_flow_video_expert import MageFlowVideoExpert
from imagewam.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="libero_mage_flow_imagewam")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--video-frame-cache-dir",
        default=None,
        help="data.train.video_frame_cache_dir override; must equal training's.",
    )
    return p.parse_args()


def to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = images.detach().float().cpu()
    if images.min() < 0:
        images = (images + 1.0) * 0.5
    images = images.clamp(0, 1)
    return [Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
            for x in images]


def cache_key(instruction: str, reference: torch.Tensor) -> str:
    reference = reference.detach().float().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(instruction.encode("utf-8"))
    digest.update(reference.numpy().tobytes())
    return digest.hexdigest()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "configs"
    overrides = [
        f"task={args.task}",
        "data.train.require_text_cache=false",
        "data.train.text_embedding_cache_dir=null",
        "data.train.qwen_text_cache_dir=null",
        # This script produces Mage caches; do not make the dataset
        # try to load the same cache before the missing entries exist.
        "data.train.mage_text_cache_dir=null",
        "data.train.video_augmentation=null",
        # Keep the exact training index order used by ImageWAM.
        "data.train.is_training_set=true",
    ]
    if args.video_frame_cache_dir:
        # Pin the exact frame cache training uses so cache keys match.
        overrides.append(
            f"data.train.video_frame_cache_dir={args.video_frame_cache_dir}"
        )
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="train",
            overrides=overrides,
        )
    dataset = instantiate(cfg.data.train)
    # Shard via custom env vars (NOT RANK/WORLD_SIZE) and use NO process group:
    # this script is launched as independent single-GPU processes (see
    # pre_compute_embed.sh). Each process sees one GPU and writes its own shard's
    # files. Avoiding torchrun / init_process_group prevents the external
    # mage_flow encoder from spinning up NCCL and timing out on cold-mmap I/O.
    rank = int(os.environ.get("IMAGEWAM_SHARD_RANK", "0"))
    world_size = int(os.environ.get("IMAGEWAM_SHARD_WORLD", "1"))
    total_samples = len(dataset)
    if world_size > 1:
        dataset = Subset(dataset, range(rank, len(dataset), world_size))
    # args.device stays "cuda" -> cuda:0 (each process has exactly one visible GPU)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    expert = MageFlowVideoExpert.from_pretrained(
        args.model_path, device=args.device, torch_dtype=torch.bfloat16,
        load_text_encoder=True, text_encoder_only=True,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    count = 0
    hidden = None
    with torch.inference_mode():
        for batch in loader:
            videos = batch["video"]
            instructions = list(batch["instruction"])
            keys = [cache_key(instructions[row], videos[row, :, 0])
                    for row in range(len(instructions))]
            missing = [row for row, key in enumerate(keys)
                       if args.overwrite or not (out_dir / f"{key}.pt").exists()]
            if not missing:
                count += len(instructions)
                if count % 128 == 0 or count == total_samples:
                    print(f"[shard{rank}] {count} samples (all cached)", flush=True)
                continue
            missing_videos = videos[missing]
            missing_instructions = [instructions[row] for row in missing]
            references = to_pil(missing_videos[:, :, 0])
            hidden, masks = expert.encode_edit_conditions(
                missing_instructions,
                [[image] for image in references],
                device=args.device,
            )
            for encoded_row, row in enumerate(missing):
                path = out_dir / f"{keys[row]}.pt"
                torch.save({
                    "context": hidden[encoded_row].cpu().to(torch.bfloat16),
                    "context_mask": masks[encoded_row].cpu().bool(),
                    "instruction": instructions[row],
                }, path)
            count += len(instructions)
            print(f"[shard{rank}] {count} samples", flush=True)
            if args.max_samples is not None and count >= args.max_samples:
                count = args.max_samples
                break
    if world_size > 1:
        pass
    if rank != 0:
        return
    if hidden is None:
        existing = next(out_dir.glob("*.pt"), None)
        if existing is None:
            raise RuntimeError(f"No Mage text cache files found in {out_dir}")
        payload = torch.load(existing, map_location="cpu", weights_only=False)
        hidden = payload["context"]
    manifest = {
        "format": "mage_flow_qwen3_vl_edit_v1",
        "num_samples": total_samples,
        "hidden_dim": int(hidden.shape[-1]),
        "max_length": int(hidden.shape[-2]),
        "dtype": "bfloat16",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] wrote {count} caches to {out_dir}")


if __name__ == "__main__":
    main()

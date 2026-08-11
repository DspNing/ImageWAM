#!/usr/bin/env python3
"""Precompute and cache video frames (post-resize, pre-crop) for data augmentation.

This stores the processed video frames as uint8 [C, T, H, W] AFTER:
  mp4 decode → take frames → concat cameras → resize (aspect-ratio preserving)

But BEFORE crop, normalize and augmentation. This allows:
  - Training-time augmentation using existing VideoAugmentation
  - Training-time crop/normalize on augmented frames
  - More effective augmentation due to larger frame size for random crop
  - Skip mp4 decode (the CPU bottleneck)

IMPORTANT: Frames are resized with aspect-ratio preserved, so H,W >= target_size.
Training uses the existing video_augmentation pipeline for consistency.

No GPU needed (no VAE encode). Pure CPU: decode + resize + store.

The script overrides crop_transform and normalize_transform to no-ops,
so only resize is applied during precomputation.

Output: data/video_frames_cache/{task_name}/
  frame_{idx:07d}.pt  — each [C, T, H, W] uint8

Note: Preprocessing only applies resize (aspect-ratio preserving), NOT crop.
Crop happens during training to allow for random crop augmentation.

Usage:
  # single process
  python scripts/precompute_video_frames.py \
    --task libero_flux2_klein_4b_base_imagewam \
    --dataset-dirs /data/WangBizi/dataset/libero/libero_spatial_no_noops_1.0.0_lerobot,/data/WangBizi/dataset/libero/libero_object_no_noops_1.0.0_lerobot \
    --output-dir data/video_frames_cache/libero_flux2

  # 8 processes in parallel
  for s in 0 1 2 3 4 5 6 7; do
    python scripts/precompute_video_frames.py \
      --task libero_flux2_klein_4b_base_imagewam \
      --dataset-dirs /data/WangBizi/dataset/libero/libero_spatial_no_noops_1.0.0_lerobot,/data/WangBizi/dataset/libero/libero_object_no_noops_1.0.0_lerobot \
      --output-dir data/video_frames_cache/libero_flux2 \
      --num-shards 8 --shard $s \
      --pretrained-norm-stats data/dataset_stats.json \
      > runs/logs/vf_shard${s}.log 2>&1 &
  done; wait
"""
import argparse
import sys
import time
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Register custom resolvers (e.g. ${envint:IMAGE_SIZE,224}, ${eval:...}) so the
# data config's image-size interpolations resolve before OmegaConf.resolve().
from imagewam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()


def build_dataset(task: str, pretrained_norm_stats: str = None, dataset_dirs: str = None):
    """Build dataset without caches to get raw video frames."""
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    OmegaConf.resolve(cfg)
    cfg.data.train.is_training_set = True
    OmegaConf.set_struct(cfg.data.train, False)

    if pretrained_norm_stats:
        cfg.data.train.pretrained_norm_stats = pretrained_norm_stats
        print(f"[INFO] Reusing dataset stats from {pretrained_norm_stats}")

    if dataset_dirs:
        cfg.data.train.dataset_dirs = dataset_dirs
        print(f"[INFO] Using dataset_dirs: {dataset_dirs}")

    # Disable caches and augmentation to get only resize-processed frames
    cfg.data.train.action_proprio_cache_path = None
    cfg.data.train.video_frame_cache_dir = None
    cfg.data.train.video_augmentation = None
    cfg.data.train.condition_frame_augmentation = None
    cfg.data.train.qwen_text_cache_dir = None  # Disable qwen cache for precompute
    cfg.data.train.mage_text_cache_dir = None  # Disable mage text cache for precompute

    ds = instantiate(cfg.data.train)

    # Override crop and normalize transforms to no-ops for precompute
    # We only want resize (aspect-ratio preserving), not crop/normalize
    class _NoOp:
        def __call__(self, x):
            return x

    ds.crop_transform = _NoOp()
    ds.normalize_transform = _NoOp()

    return ds


def main():
    ap = argparse.ArgumentParser(description="Precompute video frames for augmentation.")
    ap.add_argument("--task", default="libero_flux2_klein_4b_base_imagewam")
    ap.add_argument("--output-dir", required=True, help="Directory to store frame_*.pt files")
    ap.add_argument("--dataset-dirs", default=None, help="Path to libero datasets (comma-separated)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pretrained-norm-stats", default=None)
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for parallel mp4 decode")
    args = ap.parse_args()

    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError(f"--shard must be in [0, {args.num_shards})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[shard {args.shard}] Building dataset (raw video, no caches)...")
    t0 = time.time()
    ds = build_dataset(args.task, args.pretrained_norm_stats, args.dataset_dirs)
    total = len(ds)
    print(f"[shard {args.shard}] Dataset size: {total}, built in {time.time()-t0:.1f}s")

    end_idx = total if args.end_idx < 0 else args.end_idx
    indices = list(range(args.start_idx, end_idx))
    my_indices = [idx for i, idx in enumerate(indices) if i % args.num_shards == args.shard]

    # Filter already-done
    todo = [idx for idx in my_indices if args.overwrite or not (output_dir / f"frame_{idx:07d}.pt").exists()]
    n_skipped = len(my_indices) - len(todo)
    print(f"[shard {args.shard}] Processing {len(todo)} samples (skipped {n_skipped} existing)...")

    t0 = time.time()
    n_done = 0
    errors = []

    # Use DataLoader with multiple workers to parallelize the CPU-bound mp4 decode
    from torch.utils.data import DataLoader

    class _IdxDataset:
        def __init__(self, base, idx_list):
            self.base = base
            self.idx_list = idx_list
        def __len__(self):
            return len(self.idx_list)
        def __getitem__(self, i):
            idx = self.idx_list[i]
            return idx, self.base[idx]

    loader = DataLoader(
        _IdxDataset(ds, todo),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=lambda batch: batch[0],
    )

    for idx, sample in loader:
        out_path = output_dir / f"frame_{idx:07d}.pt"
        try:
            video = sample["video"]  # [C, T, H, W]

            # # Debug: check first sample
            # if n_done == 0:
            #     print(f"[DEBUG] First sample video:")
            #     print(f"  dtype: {video.dtype}")
            #     print(f"  shape: {video.shape}")
            #     print(f"  min: {video.min()}, max: {video.max()}")
            #     print(f"  is uint8: {video.dtype == torch.uint8}")
            #     print(f"  sample keys: {list(sample.keys())}")

            torch.save(video, out_path)
            n_done += 1
        except Exception as e:
            errors.append((int(idx), str(e)))
            if len(errors) <= 5:
                print(f"[ERROR] idx={int(idx)}: {e}")

        if n_done % 500 == 0 and n_done > 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (len(todo) - n_done) / rate
            print(f"[shard {args.shard}] done={n_done}/{len(todo)} rate={rate:.1f}/s eta={eta/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"[shard {args.shard}] DONE: {n_done} samples, {len(errors)} errors, {elapsed:.0f}s "
          f"({n_done/elapsed:.1f}/s)")
    if errors:
        print(f"[shard {args.shard}] First 5 errors: {errors[:5]}")


if __name__ == "__main__":
    main()

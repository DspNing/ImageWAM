#!/usr/bin/env python3
"""
Convert individual video frame .pt files to a memory-mapped format.

This script reads all frame_*.pt files from a directory and creates:
1. A single concatenated mmap file (video_frames.mmap)
2. An index file (video_frames_index.pt) with offsets and metadata

Benefits:
- Single file instead of 277K individual files
- Memory-mapped access is faster than torch.load()
- OS page cache works more efficiently
- Random access patterns are more predictable

Usage:
    python scripts/flux2/precompute_video_frames_mmap.py \
        --input_dir ./data/video_frames_cache/libero_flux2 \
        --output_dir ./data/video_frames_cache/libero_flux2_mmap
"""

import argparse
import os
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np


def get_frame_files(input_dir: str) -> list[Path]:
    """Get all frame_*.pt files sorted by index."""
    input_path = Path(input_dir)
    frame_files = sorted(input_path.glob("frame_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return frame_files


def validate_files(frame_files: list[Path]) -> dict:
    """Validate first few files to ensure consistent format."""
    if not frame_files:
        raise ValueError(f"No frame_*.pt files found in directory")

    # Check first file for format
    first_tensor = torch.load(frame_files[0], map_location="cpu", weights_only=False)

    info = {
        "shape": tuple(first_tensor.shape),
        "dtype": str(first_tensor.dtype),
        "num_files": len(frame_files),
    }

    print(f"Validation results:")
    print(f"  Shape: {info['shape']}")
    print(f"  Dtype: {info['dtype']}")
    print(f"  Total files: {info['num_files']}")

    return info


def create_mmap_cache(frame_files: list[Path], output_mmap: Path, output_index: Path) -> dict:
    """Create concatenated mmap file and index.

    Args:
        frame_files: List of frame_*.pt files sorted by index
        output_mmap: Path to output mmap file
        output_index: Path to output index file

    Returns:
        Metadata about the created cache
    """
    # Load first file to get shape info
    first_tensor = torch.load(frame_files[0], map_location="cpu", weights_only=False)
    frame_shape = first_tensor.shape  # [C, T, H, W]
    frame_dtype = first_tensor.dtype
    frame_size_bytes = first_tensor.numel() * first_tensor.element_size()

    print(f"\nCreating mmap cache:")
    print(f"  Frame shape: {frame_shape} [C, T, H, W]")
    print(f"  Frame dtype: {frame_dtype}")
    print(f"  Frame size: {frame_size_bytes} bytes")
    print(f"  Total frames: {len(frame_files)}")
    print(f"  Total size: {len(frame_files) * frame_size_bytes / 1024**3:.2f} GB")

    # Calculate total size
    total_size = len(frame_files) * frame_size_bytes

    # Create index metadata
    index = {
        "num_frames": len(frame_files),
        "frame_shape": list(frame_shape),  # [C, T, H, W]
        "frame_dtype": str(frame_dtype),
        "frame_size_bytes": frame_size_bytes,
        "total_size_bytes": total_size,
        "offsets": torch.zeros(len(frame_files) + 1, dtype=torch.int64),  # +1 for end marker
    }

    # Create output directory
    output_mmap.parent.mkdir(parents=True, exist_ok=True)

    # Initialize mmap file with zeros - use flattened shape for 1D array
    # This is easier to work with: we'll flatten each frame to 1D
    total_elements = len(frame_files) * first_tensor.numel()
    print(f"\nWriting to {output_mmap}...")
    print(f"  Total elements: {total_elements:,}")

    mmap_array = np.memmap(
        output_mmap,
        dtype=np.float32 if frame_dtype == torch.float32 else np.uint8,
        mode="w+",
        shape=(len(frame_files), first_tensor.numel()),  # (num_frames, flattened_size)
    )

    # Copy each frame
    t0 = time.time()
    with tqdm(total=len(frame_files), desc="Copying frames", unit="frames") as pbar:
        for i, frame_path in enumerate(frame_files):
            tensor = torch.load(frame_path, map_location="cpu", weights_only=False)
            # Flatten the tensor to 1D and store
            mmap_array[i] = tensor.numpy().reshape(-1)
            index["offsets"][i + 1] = index["offsets"][i] + frame_size_bytes
            pbar.update(1)

    # Flush mmap to disk
    mmap_array.flush()
    elapsed = time.time() - t0

    print(f"Completed in {elapsed:.1f}s ({len(frame_files)/elapsed:.1f} frames/s)")

    # Save index
    torch.save(index, output_index)
    print(f"Index saved to {output_index}")

    return {
        "num_frames": len(frame_files),
        "frame_shape": frame_shape,
        "total_size_gb": total_size / 1024**3,
        "elapsed_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert video frame cache to mmap format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing frame_*.pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for mmap files",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate input files without creating mmap",
    )
    args = parser.parse_args()

    # Get frame files
    print(f"Scanning {args.input_dir}...")
    frame_files = get_frame_files(args.input_dir)

    # Validate
    info = validate_files(frame_files)

    if args.validate_only:
        print("\nValidation complete. Use --validate_only=0 to create mmap cache.")
        return

    # Create mmap cache
    output_mmap = Path(args.output_dir) / "video_frames.mmap"
    output_index = Path(args.output_dir) / "video_frames_index.pt"

    result = create_mmap_cache(frame_files, output_mmap, output_index)

    print(f"\n✅ Mmap cache created successfully:")
    print(f"  Output: {args.output_dir}")
    print(f"  Frames: {result['num_frames']:,}")
    print(f"  Size: {result['total_size_gb']:.2f} GB")
    print(f"  Time: {result['elapsed_time']:.1f}s")
    print(f"\nTo use in training, set:")
    print(f"  video_frame_cache_dir: {args.output_dir}")
    print(f"  video_frame_cache_format: mmap")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Create a shallow FLUX.2 Klein checkpoint from the 4B checkpoint."""

import argparse
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


def interpolation_sources(source_depth: int, target_depth: int):
    result = []
    for target_index in range(target_depth):
        position = (target_index + 0.5) * source_depth / target_depth - 0.5
        position = max(0.0, min(float(source_depth - 1), position))
        left = int(position)
        right = min(left + 1, source_depth - 1)
        result.append((left, right, position - left))
    return tuple(result)


DOUBLE_INDICES = (0, 2, 4)
SINGLE_INDICES = (0, 2, 4, 6, 9, 11, 13, 15, 17, 19)
BLOCK_KEY = re.compile(r"^(double_blocks|single_blocks)\.(\d+)(\..+)$")


def convert(source: Path, destination: Path, strategy: str) -> None:
    state = load_file(str(source), device="cpu")
    converted = {}
    selected = {
        "double_blocks": DOUBLE_INDICES,
        "single_blocks": SINGLE_INDICES,
    }
    remapped = {name: {src: dst for dst, src in enumerate(indices)} for name, indices in selected.items()}
    interpolation = {
        "double_blocks": interpolation_sources(5, len(DOUBLE_INDICES)),
        "single_blocks": interpolation_sources(20, len(SINGLE_INDICES)),
    }

    for key, value in state.items():
        match = BLOCK_KEY.match(key)
        if match is None:
            converted[key] = value
            continue
        block_type, source_index, suffix = match.groups()
        source_index = int(source_index)
        if strategy == "select":
            target_index = remapped[block_type].get(source_index)
            if target_index is not None:
                converted[f"{block_type}.{target_index}{suffix}"] = value
        else:
            for target_index, (left, right, alpha) in enumerate(interpolation[block_type]):
                if source_index == left:
                    key_out = f"{block_type}.{target_index}{suffix}"
                    if left == right:
                        converted[key_out] = value
                    else:
                        right_key = f"{block_type}.{right}{suffix}"
                        if right_key in state:
                            converted[key_out] = value * (1.0 - alpha) + state[right_key] * alpha
                    break

    if not converted:
        raise RuntimeError("Input checkpoint produced no tensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(destination), metadata={"source": str(source), "architecture": "flux2-klein-2b-3d-10s"})
    print(f"saved {len(converted)} tensors to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=["select", "interpolate"], default="select")
    args = parser.parse_args()
    convert(args.source, args.output, args.strategy)


if __name__ == "__main__":
    main()

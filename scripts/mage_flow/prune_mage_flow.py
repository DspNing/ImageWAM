#!/usr/bin/env python3
"""Create a smaller Mage-Flow checkpoint by retaining the first N DiT blocks.

The source checkpoint is never modified. VAE, text encoder, scheduler, and all
non-block transformer weights are copied unchanged; only transformer block
weights and ``transformer/config.json`` are pruned.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)(?:\.|$)")


def select_layer_indices(original_depth: int, keep_layers: int) -> list[int]:
    if keep_layers <= 0 or keep_layers > original_depth:
        raise ValueError(
            f"keep_layers must be in [1, {original_depth}], got {keep_layers}"
        )
    # Cover the complete network depth instead of keeping only its shallow half.
    return sorted({round(i * (original_depth - 1) / (keep_layers - 1)) for i in range(keep_layers)})


def build_pruned_state_dict(
    state_dict: dict[str, Any], keep_layers: int, layer_indices: list[int] | None = None
) -> dict[str, Any]:
    if keep_layers <= 0:
        raise ValueError("keep_layers must be positive")
    source_indices = layer_indices if layer_indices is not None else range(keep_layers)
    layer_map = {source: target for target, source in enumerate(source_indices)}
    result = {}
    for key, value in state_dict.items():
        match = _BLOCK_RE.match(key)
        if match is None:
            result[key] = value
        elif int(match.group(1)) in layer_map:
            source_layer = int(match.group(1))
            result[key.replace(
                f"transformer_blocks.{source_layer}",
                f"transformer_blocks.{layer_map[source_layer]}",
                1,
            )] = value
    return result


def update_transformer_config(config_path: str | Path, keep_layers: int) -> None:
    path = Path(config_path)
    config = json.loads(path.read_text())
    original_depth = config.get("depth")
    if not isinstance(original_depth, int):
        raise ValueError(f"{path} does not contain an integer depth")
    if keep_layers <= 0 or keep_layers > original_depth:
        raise ValueError(
            f"keep_layers must be in [1, {original_depth}], got {keep_layers}"
        )
    config["depth"] = keep_layers
    path.write_text(json.dumps(config, indent=2) + "\n")


def _write_safetensors(path: Path, state_dict: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    save_file(state_dict, str(path), metadata={"format": "pt"})


def _prune_single_file(path: Path, keep_layers: int, layer_indices: list[int]) -> None:
    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    _write_safetensors(path, build_pruned_state_dict(state_dict, keep_layers, layer_indices))


def _prune_sharded_transformer(
    transformer_dir: Path, keep_layers: int, layer_indices: list[int]
) -> None:
    from safetensors.torch import load_file

    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    grouped: dict[str, dict[str, Any]] = {}
    for key, shard_name in index["weight_map"].items():
        grouped.setdefault(shard_name, {})[key] = None
    new_map = {}
    for shard_name, keys in grouped.items():
        shard_path = transformer_dir / shard_name
        source = load_file(str(shard_path), device="cpu")
        pruned = build_pruned_state_dict(source, keep_layers, layer_indices)
        _write_safetensors(shard_path, pruned)
        new_map.update({key: shard_name for key in pruned})
    index["weight_map"] = new_map
    index.pop("metadata", None)
    index_path.write_text(json.dumps(index, indent=2) + "\n")


def prune_checkpoint(source: str | Path, destination: str | Path, keep_layers: int = 6) -> None:
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not (source / "transformer" / "config.json").is_file():
        raise FileNotFoundError(f"Missing Mage-Flow transformer config: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {destination}")
    shutil.copytree(source, destination)
    transformer_dir = destination / "transformer"
    source_config = json.loads((source / "transformer" / "config.json").read_text())
    original_depth = source_config["depth"]
    layer_indices = select_layer_indices(original_depth, keep_layers)
    update_transformer_config(transformer_dir / "config.json", keep_layers)
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        _prune_sharded_transformer(transformer_dir, keep_layers, layer_indices)
    else:
        weight_path = transformer_dir / "diffusion_pytorch_model.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing transformer weights under {source}")
        _prune_single_file(weight_path, keep_layers, layer_indices)
    metadata = {
        "source_checkpoint": str(source),
        "original_depth": original_depth,
        "kept_depth": keep_layers,
        "kept_layer_indices": layer_indices,
        "pruning": "uniform_layer_selection",
    }
    (destination / "pruning_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original Mage-Flow model directory")
    parser.add_argument("--destination", required=True, help="New model directory")
    parser.add_argument("--keep-layers", type=int, default=6)
    args = parser.parse_args()
    prune_checkpoint(args.source, args.destination, args.keep_layers)
    print(f"Wrote {args.keep_layers}-layer Mage-Flow checkpoint to {args.destination}")


if __name__ == "__main__":
    main()

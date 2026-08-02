#!/usr/bin/env python3
"""Initialize MageFlowActionDiT from a Mage-Flow video transformer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imagewam.models.backbones.mage_flow_imports import ensure_mage_flow_importable
from imagewam.models.backbones.mage_flow_action import MageFlowActionDiT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mage-flow-src-path", default=str(REPO_ROOT / "third_party" / "Mage"))
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--max-action-horizon", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ensure_mage_flow_importable(args.mage_flow_src_path)
    from mage_flow.pipeline import load_from_repo

    mage = load_from_repo(args.model_path, device=args.device, load_text_encoder=False)
    transformer = mage.transformer
    action = MageFlowActionDiT.from_video_transformer(
        transformer,
        action_dim=args.action_dim,
        max_action_horizon=args.max_action_horizon,
        device=args.device,
        torch_dtype=torch.bfloat16,
    )
    meta = MageFlowActionDiT.initialize_from_video(transformer.state_dict(), action)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().float().cpu().contiguous()
        for key, value in action.state_dict().items()
    }
    torch.save({"state_dict": state_dict, "meta": {
        "source": args.model_path,
        "action_dim": args.action_dim,
        "copied": meta["copied"],
        "resized": meta["resized"],
        "total": meta["total"],
    }}, output)
    print(f"[ok] saved {output} copied={meta['copied']}/{meta['total']}")


if __name__ == "__main__":
    main()

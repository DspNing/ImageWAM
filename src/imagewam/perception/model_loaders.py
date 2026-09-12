"""Perception model loaders for the object-addr cache pipeline.

API 基于 2026-07-25 部署验证:
- DINOv3: torch.hub.load(source='local') + 手动 load 权重;hubconf 的 import 链
  (经 segmentors)需要 torchmetrics(已装)。vits16 的 embed_dim=384。
- SAM3 加载见 sam3_grounding.load_sam3_grounding(自包含,含 transform + postprocessor)。
"""
import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# The dinov3 hubconf import chain pulls torchmetrics (via eval/segmentation)
# which the mageflow env does not ship. The backbone path only needs the
# `Metric` symbol to import cleanly — stub it if missing (same pattern as the
# probe scripts; install the real package to remove this).
if "torchmetrics" not in sys.modules:
    try:
        import torchmetrics  # noqa: F401
    except ModuleNotFoundError:
        import types as _types
        _tm = _types.ModuleType("torchmetrics")
        _tm.Metric = type("Metric", (), {})
        sys.modules["torchmetrics"] = _tm


def load_dinov3(device="cuda", dtype=torch.float32, repo_dir=None, ckpt=None):
    """加载 DINOv3 vits16(embed_dim=384)。返回 backbone(用于 mask 区域 masked-avg-pool)。"""
    repo_dir = repo_dir or os.path.join(REPO_ROOT, "third_party", "dinov3")
    ckpt = ckpt or os.path.join(REPO_ROOT, "checkpoints", "Dino", "dinov3_vits16_pretrain.pth")
    model = torch.hub.load(repo_dir, "dinov3_vits16", source="local", pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    if isinstance(state, dict) and any(k.startswith("teacher") for k in state):
        state = {k.replace("teacher.", "", 1): v for k, v in state.items() if k.startswith("teacher")}
    elif isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[dinov3] state_dict load: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device, dtype).eval()
    return model

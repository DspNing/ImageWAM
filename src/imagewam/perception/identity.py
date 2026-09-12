"""DINOv3 mask 区域 masked-average-pool 物体身份特征(vits16 embed_dim=384)。

身份跨帧稳定(语义级),用作 e_obj 的视觉分量(spec §4)。只对 mask 区域 pooling,
不混入背景(DINOv3 在干净物体区域取特征)。
"""
import torch
import torch.nn.functional as F

_DINOV3_INPUT_SIZE = 518  # vits16 patch_size=14 → 518/14=37 tokens/边


def _preprocess(image_pil, device, dtype):
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize((_DINOV3_INPUT_SIZE, _DINOV3_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tfm(image_pil).unsqueeze(0).to(device, dtype)


@torch.no_grad()
def compute_identity(image_pil, masks: list[torch.Tensor], dinov3_model) -> torch.Tensor:
    """每个 mask 区域内 DINOv3 dense feature 的 masked-average-pool → [n_obj, d_dino]。

    masks: list of [H,W] bool(原图分辨率);返回每个物体的身份向量。
    """
    params = next(dinov3_model.parameters())
    dtype = params.dtype
    x = _preprocess(image_pil, params.device, dtype)
    feats = dinov3_model.get_intermediate_layers(x, n=1, reshape=True)[0]  # [B, C, h, w]
    feats = feats[0]  # [C, h, w]
    _, h, w = feats.shape
    out = []
    for m in masks:
        m_dev = m.to(feats.device).float()[None, None]
        m_resized = F.interpolate(m_dev, size=(h, w), mode="area")[0, 0]  # [h, w]
        weighted = (feats * m_resized).sum(dim=(1, 2))  # [C]
        denom = m_resized.sum().clamp_min(1e-6)
        out.append(weighted / denom)
    return torch.stack(out)  # [n_obj, C]


@torch.no_grad()
def compute_identity_gpu(cam01: torch.Tensor, masks: list[torch.Tensor], dinov3_model) -> torch.Tensor:
    """GPU 原生版:cam01 = [C,H,W] GPU tensor [0,1](不经过 PIL);masks = list of [H,W](任意设备)。

    GPU 上 resize→518 + ImageNet normalize(替代 _preprocess 的 PIL Resize),再 masked-avg-pool。
    返回 [n_obj, d_dino](CPU)。
    """
    params = next(dinov3_model.parameters())
    dev, dtype = params.device, params.dtype
    x = F.interpolate(cam01.unsqueeze(0).float(), size=(_DINOV3_INPUT_SIZE, _DINOV3_INPUT_SIZE),
                      mode="bicubic", align_corners=False, antialias=True).to(dev, dtype)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev, dtype=dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    feats = dinov3_model.get_intermediate_layers(x, n=1, reshape=True)[0][0]  # [C, h, w]
    _, h, w = feats.shape
    out = []
    for m in masks:
        m_dev = m.to(dev).float()[None, None]
        m_resized = F.interpolate(m_dev, size=(h, w), mode="area")[0, 0]
        weighted = (feats * m_resized).sum(dim=(1, 2))
        denom = m_resized.sum().clamp_min(1e-6)
        out.append(weighted / denom)
    return torch.stack(out).cpu()  # [n_obj, C]

"""Object addr cache 数据结构 + save/load。

分 cam 感知(配合 VideoAugmentation 每 cam 独立几何增强 + 横拼):
per-episode(固定):phrases, lang_emb[n_phrases, d_qwen], identity[N, d_dino],
                   object_phrase_index[N](物体 → 短语), object_cam[N](物体 → cam)
per-frame(每帧 SAM mask):frame_masks[T, N, H, W] bool(每物体在它所属 cam 的 mask)

cache 只存感知产物(固定的);e_obj / K_addr 依赖可训参数,在线算(Plan 2),不 cache。
"""
from dataclasses import dataclass

import torch


@dataclass
class ObjectAddrCache:
    episode_id: str
    phrases: list[str]                 # 指令里的名词短语(语言指代)
    lang_emb: torch.Tensor             # [n_phrases, d_qwen]
    identity: torch.Tensor             # [N, d_dino] 每物体 DINOv3 身份(所属 cam 首帧)
    object_phrase_index: list[int]     # 物体 i 对应 phrases 哪个短语
    object_cam: list[int]              # 物体 i 属哪个 cam(0=image, 1=wrist)—— 横拼定位用
    frame_masks: torch.Tensor          # [T, N, H, W] bool,每物体在它所属 cam 的 mask


def save_cache(cache: ObjectAddrCache, path: str) -> None:
    torch.save(
        {
            "episode_id": cache.episode_id,
            "phrases": cache.phrases,
            "lang_emb": cache.lang_emb.cpu(),
            "identity": cache.identity.cpu(),
            "object_phrase_index": cache.object_phrase_index,
            "object_cam": cache.object_cam,
            "frame_masks": cache.frame_masks.cpu(),
        },
        path,
    )


def load_cache(path: str) -> ObjectAddrCache:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return ObjectAddrCache(**d)


def collate_object_addr(object_addrs: list[dict | None]) -> dict | None:
    """Collate 一 batch 的 per-sample object_addr(可含 None)→ batched dict(pad N_max + P_max)。

    输入 list[B] of dict{lang_emb[P,d_qwen], identity[N,d_dino], object_phrase_index[N],
    object_cam[N], frame_masks[N,H,W]} 或 None(该 sample 无 addr)。
    返回 batched dict 或 None(全 None)。padding: identity/lang_emb→0, phrase_index→-1, cam→0, mask→False。
    """
    valid_idx = [b for b, oa in enumerate(object_addrs) if oa is not None]
    if not valid_idx:
        return None
    B = len(object_addrs)
    N_max = max(object_addrs[b]["identity"].shape[0] for b in valid_idx)
    P_max = max(object_addrs[b]["lang_emb"].shape[0] for b in valid_idx)
    d_qwen = object_addrs[valid_idx[0]]["lang_emb"].shape[1]
    d_dino = object_addrs[valid_idx[0]]["identity"].shape[1]
    H, W = object_addrs[valid_idx[0]]["frame_masks"].shape[-2:]
    lang_emb = torch.zeros(B, P_max, d_qwen, dtype=object_addrs[valid_idx[0]]["lang_emb"].dtype)
    identity = torch.zeros(B, N_max, d_dino, dtype=object_addrs[valid_idx[0]]["identity"].dtype)
    object_phrase_index = torch.full((B, N_max), -1, dtype=torch.long)
    object_cam = torch.zeros(B, N_max, dtype=torch.long)
    frame_masks = torch.zeros(B, N_max, H, W, dtype=torch.bool)
    for b in valid_idx:
        oa = object_addrs[b]
        n = oa["identity"].shape[0]
        p = oa["lang_emb"].shape[0]
        lang_emb[b, :p] = oa["lang_emb"]
        identity[b, :n] = oa["identity"]
        object_phrase_index[b, :n] = torch.as_tensor(oa["object_phrase_index"], dtype=torch.long)
        object_cam[b, :n] = torch.as_tensor(oa["object_cam"], dtype=torch.long)
        frame_masks[b, :n] = oa["frame_masks"]
    return {
        "lang_emb": lang_emb,
        "identity": identity,
        "object_phrase_index": object_phrase_index,
        "object_cam": object_cam,
        "frame_masks": frame_masks,
    }

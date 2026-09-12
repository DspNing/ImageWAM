"""eval 在线 object_addr 感知 —— 与离线 cache 构建(build_object_addr_cache)**同一套管线**。

train 读 cache(离线存盘),eval 用本模块在线算;两者感知函数完全一致:
  SAM3 segment_batch_gpu(det=0.15,GPU bicubic)+ DINOv3 compute_identity_gpu + Qwen 短语。
保证 train/eval 的 object_addr 同分布。

make_object_addr(cams_01, instruction, perc, phrase_cache):
  cams_01 = list of [3,112,112] GPU tensor [0,1](每 cam 一张,和模型吃的那帧同一张);
  返回 cache 同构 dict{lang_emb, identity, object_phrase_index, object_cam, frame_masks}。
"""
import os
import torch
import torch.nn.functional as F

from .noun_phrase import extract_noun_phrases, embed_noun_phrases
from .sam3_grounding import load_sam3_grounding
from .identity import compute_identity_gpu
from .model_loaders import load_dinov3


def _resize_mask(mask, size):
    m = mask.to(torch.float)[None, None]
    m = F.interpolate(m, size=(size, size), mode="nearest")[0, 0]
    return m > 0.5


def load_online_perception(qwen_spec="checkpoints/Qwen3-4B", device="cuda", det_threshold=0.15,
                           load_qwen=True):
    """加载 eval 在线感知模型。返回 (qwen, tok, sam3g, dino)。

    load_qwen=False 时跳过 Qwen3(qwen=tok=None)——配合预算好的 addr 短语缓存
    (load_addr_phrase_cache 预填 phrase_cache),eval 不需要在线 Qwen3,省 ~8GB/worker。
    SAM3 + DINOv3 始终加载(obs 相关,必须在线)。
    """
    qwen, tok = None, None
    if load_qwen:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(qwen_spec)
        qwen = AutoModelForCausalLM.from_pretrained(qwen_spec, torch_dtype=torch.bfloat16).to(device).eval()
    sam3g = load_sam3_grounding(device=device, detection_threshold=det_threshold)
    dino = load_dinov3(device=device)
    return qwen, tok, sam3g, dino


def load_addr_phrase_cache(cache_dir):
    """读预算好的 addr 短语缓存 → {instruction: (phrases, lang_emb)} dict(预填 phrase_cache)。

    eval 把这个 dict 作为 phrase_cache 传给 make_object_addr:命中即不碰 Qwen3。
    每个文件 {task_language, phrases, lang_emb};按 task_language 建 key(eval 传原始 task.language)。
    """
    import glob
    cache = {}
    for f in glob.glob(os.path.join(cache_dir, "*.pt")):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            cache[d["task_language"]] = (d["phrases"], d["lang_emb"])
        except Exception:
            continue
    return cache


def make_object_addr(cams_01, instruction, perc, phrase_cache, mask_size=112):
    """在线产 object_addr(与 build_object_addr_cache.build_cache_dict + segment_batch_gpu 同构)。

    cams_01: list of [3,112,112] GPU tensor [0,1](每 cam 一张,= 模型吃的当前帧)。
    perc: load_online_perception() 的返回 (qwen, tok, sam3g, dino)。
    phrase_cache: dict,instruction → (phrases, lang_emb)(Qwen 按 task 缓存,跨 replan 复用)。
    返回 dict{lang_emb[P,2560], identity[N,384], object_phrase_index[N], object_cam[N], frame_masks[N,112,112]}。
    """
    qwen, tok, sam3g, dino = perc
    # Qwen 短语:phrase_cache 命中(预算缓存或 episode 内复用)则直接用;未命中且有 Qwen 才在线算;
    # 未命中且无 Qwen(load_qwen=False + 缓存没覆盖该 instruction)→ 空 phrases,addr 该步跳过(不崩)。
    if instruction not in phrase_cache:
        if qwen is not None:
            phrases = extract_noun_phrases(instruction, qwen, tok)
            lang_emb = embed_noun_phrases(phrases, qwen, tok) if phrases else torch.zeros(0, 2560)
            phrase_cache[instruction] = (phrases, lang_emb.cpu())
        else:
            phrase_cache[instruction] = ([], torch.zeros(0, 2560))
    phrases, lang_emb = phrase_cache[instruction]

    empty = {"lang_emb": lang_emb, "identity": torch.zeros(0, 384),
             "object_phrase_index": torch.zeros(0, dtype=torch.long),
             "object_cam": torch.zeros(0, dtype=torch.long),
             "frame_masks": torch.zeros(0, mask_size, mask_size, dtype=torch.bool)}
    if not phrases:
        return empty

    # SAM3 分割(同 cache 构建:segment_batch_gpu,det=0.15)
    objs_per_cam = sam3g.segment_batch_gpu([(cams_01, phrases)])[0]  # list[list[dict]],每 cam 一组

    # DINOv3 身份 + 拼装(同 build_cache_dict)
    all_identity, all_masks, all_phrase_idx, all_cam = [], [], [], []
    for cam_id, (cam01, objs) in enumerate(zip(cams_01, objs_per_cam)):
        if not objs:
            continue
        masks0 = [o["mask"] for o in objs]
        identity = compute_identity_gpu(cam01, masks0, dino)  # [n, 384] CPU
        all_identity.append(identity)
        all_masks.extend([_resize_mask(m, mask_size) for m in masks0])
        all_phrase_idx.extend([phrases.index(o["phrase"]) if o["phrase"] in phrases else -1 for o in objs])
        all_cam.extend([cam_id] * len(objs))
    if all_identity:
        return {"lang_emb": lang_emb,
                "identity": torch.cat(all_identity, dim=0).cpu(),
                "object_phrase_index": torch.as_tensor(all_phrase_idx, dtype=torch.long),
                "object_cam": torch.as_tensor(all_cam, dtype=torch.long),
                "frame_masks": torch.stack(all_masks, dim=0).cpu()}
    return empty

"""SAM3 文本提示分割(名词短语 → 精确 mask,一步到位,替代 GD+SAM2)。

SAM3 原生支持文本 prompt grounding:把每个短语作为 find_query 直接喂模型,
一次 forward 出 mask,无需 Grounding DINO 先出 bbox 再喂 SAM。且支持 batch
(多图 / 多 query 一次 forward,每图可带不同短语),SAM2 的 set_image 串行瓶颈没了。

输出:list[list[dict]],每图一组,dict = {"phrase", "mask" Tensor[H,W] bool, "bbox" Tensor[4], "score"}。
"""
import os
import torch

SAM3_SQUARE_INPUT = 1008  # SAM3 image 模型固定 resize(square)


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.bool(); b = b.bool()
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union > 0 else 0.0


def _nms(objs: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """mask-IoU 去重:同物体被多短语命中时(高重叠)保留第一个。"""
    keep = []
    for o in objs:
        if all(_mask_iou(o["mask"], k["mask"]) < iou_threshold for k in keep):
            keep.append(o)
    return keep


class SAM3Grounding:
    """封装 SAM3 image 模型 + transform + postprocessor。

    - segment(images, phrases):单样本(各 cam 共享一组 phrases)。
    - segment_batch(samples):多样本合并 forward(每样本各自的 phrases),build cache 用。
    - forward_raw / postprocess_at:forward 与 postprocess 分离,sweep 阈值复用同一 output。
    """

    def __init__(self, model, transform, postprocessor, device):
        self.model = model
        self.transform = transform
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        self._counter = 1  # 每个 query 唯一 coco_image_id(用作 postprocessor 的 result key)

    def _make_datapoint(self, pil, phrases):
        from sam3.train.data.sam3_image_dataset import (
            InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint,
        )
        w, h = pil.size
        dp = Datapoint(find_queries=[], images=[SAMImage(data=pil, objects=[], size=[h, w])])
        ids = []
        for t in phrases:
            dp.find_queries.append(FindQueryLoaded(
                query_text=t, image_id=0, object_ids_output=[], is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=self._counter, original_image_id=self._counter,
                    original_category_id=1, original_size=[w, h], object_id=0, frame_index=0,
                ),
            ))
            ids.append(self._counter)
            self._counter += 1
        return dp, ids

    @torch.inference_mode()
    def segment(self, images, phrases):
        """单样本:images=list[PIL](各 cam 共享一组 phrases)。返回 list[list[dict]](每图一组)。"""
        if not phrases:
            return [[] for _ in images]
        per_image = [(im, phrases) for im in images]
        output, batch, qid_map = self.forward_raw(per_image)
        return self.postprocess_at(output, batch, per_image, qid_map, threshold=None)

    @torch.inference_mode()
    def segment_batch(self, samples):
        """多样本合并 forward。samples = list[(cams, phrases)],cams=list[PIL]。

        把所有样本的 cam 展平成 per_image 一次 forward(每图带各自 phrases),再按样本归组。
        返回 list[list[list[dict]]]:samples[i] → 每图一组物体(与 cams 顺序对齐)。
        """
        per_image, owner = [], []  # owner[k] = (sample_idx, cam_idx)
        for si, (cams, phrases) in enumerate(samples):
            if not phrases:
                continue
            for ci, pil in enumerate(cams):
                owner.append((si, ci))
                per_image.append((pil, phrases))
        if not per_image:
            return [[[] for _ in cams] for cams, _ in samples]
        output, batch, qid_map = self.forward_raw(per_image)
        objs_per_image = self.postprocess_at(output, batch, per_image, qid_map, threshold=None)
        out = [[None] * len(cams) for cams, _ in samples]  # 归组回每样本
        for k, (si, ci) in enumerate(owner):
            out[si][ci] = objs_per_image[k]
        for si, (cams, _) in enumerate(samples):
            for ci in range(len(cams)):
                if out[si][ci] is None:
                    out[si][ci] = []
        return out

    @torch.inference_mode()
    def forward_raw_gpu(self, per_image):
        """GPU 原生 forward:跳过 PIL transform,resize+normalize 在 GPU 上做(省 CPU)。

        per_image = list[(cam01, phrases)]:cam01 = [C,H,W] GPU tensor 值域 [0,1](每个 cam 一项)。
        返回 (output, batch, qid_map),qid_map = {qid: (img_idx, phrase_idx)}。
        等价于 PIL transform(RandomResize 1008 square + ToTensor + Normalize 0.5/0.5),但在 GPU 上。
        """
        import torch.nn.functional as F
        from sam3.train.data.collator import collate_fn_api as collate
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.data.sam3_image_dataset import (
            InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint,
        )

        dps, qid_map = [], {}
        for img_idx, (cam01, phrases) in enumerate(per_image):
            h, w = cam01.shape[-2], cam01.shape[-1]  # 原始尺寸(112)
            # GPU resize 到 1008(square, bicubic+antialias 近似 PIL bicubic)+ normalize [0,1]→[-1,1]
            t = F.interpolate(cam01.unsqueeze(0).float(), size=SAM3_SQUARE_INPUT,
                              mode="bicubic", align_corners=False, antialias=True)[0]
            t = (t * 2.0 - 1.0).to(torch.bfloat16)
            dp = Datapoint(find_queries=[], images=[SAMImage(data=t, objects=[], size=[h, w])])
            ids = []
            for phrase in phrases:
                dp.find_queries.append(FindQueryLoaded(
                    query_text=phrase, image_id=0, object_ids_output=[], is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=self._counter, original_image_id=self._counter,
                        original_category_id=1, original_size=[w, h], object_id=0, frame_index=0,
                    ),
                ))
                ids.append(self._counter)
                self._counter += 1
            for qid, pidx in zip(ids, range(len(phrases))):
                qid_map[qid] = (img_idx, pidx)
            dps.append(dp)
        batch = collate(dps, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(batch)
        return output, batch, qid_map

    @torch.inference_mode()
    def segment_batch_gpu(self, samples):
        """GPU 原生 segment_batch。samples = list[(cams_01, phrases)],
        cams_01 = list of [C,H,W] GPU tensor [0,1]。返回 list[list[list[dict]]](每样本每 cam 一组)。"""
        per_image, owner = [], []
        for si, (cams_01, phrases) in enumerate(samples):
            if not phrases:
                continue
            for ci, cam01 in enumerate(cams_01):
                owner.append((si, ci))
                per_image.append((cam01, phrases))
        if not per_image:
            return [[[] for _ in cams] for cams, _ in samples]
        output, batch, qid_map = self.forward_raw_gpu(per_image)
        objs_per_image = self.postprocess_at(output, batch, per_image, qid_map, threshold=None, to_cpu=False)
        out = [[None] * len(cams) for cams, _ in samples]
        for k, (si, ci) in enumerate(owner):
            out[si][ci] = objs_per_image[k]
        for si, (cams, _) in enumerate(samples):
            for ci in range(len(cams)):
                if out[si][ci] is None:
                    out[si][ci] = []
        return out

    @torch.inference_mode()
    def forward_raw(self, per_image):
        """跑一次 SAM3 forward(不含 postprocess)。

        per_image = list[(pil, phrases)]:每图带各自 phrases(batch 里不同样本可不同短语)。
        返回 (output, batch, qid_map),qid_map = {qid: (img_idx, phrase_idx)}。
        昂贵的是 forward;阈值只在 postprocess 阶段应用,sweep 阈值复用同一 output。
        """
        from sam3.train.data.collator import collate_fn_api as collate
        from sam3.model.utils.misc import copy_data_to_device

        dps, qid_map = [], {}
        for img_idx, (pil, phrases) in enumerate(per_image):
            dp, ids = self._make_datapoint(pil, phrases)
            for qid, pidx in zip(ids, range(len(phrases))):
                qid_map[qid] = (img_idx, pidx)
            dps.append(self.transform(dp))
        batch = collate(dps, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(batch)
        return output, batch, qid_map

    def postprocess_at(self, output, batch, per_image, qid_map, threshold=None, to_cpu=True):
        """对已 forward 的 output 做 postprocess。threshold=None 用 postprocessor 当前阈值。

        每图每短语取最高分检出(阈值由 postprocessor.detection_threshold 控制),mask-IoU NMS。
        返回 list[list[dict]](每图一组 {phrase,mask,bbox,score}),顺序与 per_image 对齐。
        """
        if threshold is not None:
            self.postprocessor.detection_threshold = threshold
        results = self.postprocessor.process_results(output, batch.find_metadatas)

        buckets = [[] for _ in range(len(per_image))]
        for qid, (img_idx, pidx) in qid_map.items():
            r = results.get(qid)
            if r is None or r["scores"].numel() == 0:
                continue
            top = int(r["scores"].float().argmax())  # 同一短语可能多个检出,取最高分
            mask = r["masks"][top, 0]  # [H, W](已 resize 回原图尺寸)
            mask = mask if mask.dtype == torch.bool else mask > 0.5
            if to_cpu:
                mask = mask.cpu()
            phrases = per_image[img_idx][1]
            buckets[img_idx].append({
                "phrase": phrases[pidx],
                "mask": mask,
                "bbox": r["boxes"][top].detach().cpu(),
                "score": float(r["scores"][top].float()),
            })
        return [_nms(b) for b in buckets]


def load_sam3_grounding(
    device="cuda",
    ckpt=None,
    bpe_path=None,
    load_from_hf=False,
    detection_threshold=0.15,
):
    """构建 SAM3 image 模型 + transform + postprocessor,返回 SAM3Grounding。

    detection_threshold:文本提示检出的置信度门槛。阈值 sweep(多场景可视化)定到 0.15
    ——召回接近上限,mask 仍紧贴物体。
    """
    import sys
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sam3_repo = os.path.join(repo, "third_party", "sam3")
    if sam3_repo not in sys.path:
        sys.path.insert(0, sam3_repo)
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage

    sam3_pkg = os.path.dirname(sam3.__file__)
    ckpt = ckpt or os.path.join(repo, "checkpoints", "sam3.1", "sam3.1_multiplex.pt")
    bpe_path = bpe_path or os.path.join(sam3_pkg, "assets", "bpe_simple_vocab_16e6.txt.gz")

    model = build_sam3_image_model(
        checkpoint_path=ckpt, bpe_path=bpe_path, device=device, load_from_HF=load_from_hf,
    )
    # 注意:不能整体 .to(bf16)。SAM3 decoder 用 activation_ckpt_wrapper(gradient checkpointing),
    # checkpoint 包裹的层不在 autocast 作用域内、其输入是 fp32,权重必须是 fp32 才能对上 dtype。
    # 计算靠 forward 内的 autocast(bf16)处理,存储维持 fp32(3.3GB)。
    model.eval()

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=SAM3_SQUARE_INPUT, max_size=SAM3_SQUARE_INPUT, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm",
        use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=detection_threshold, to_cpu=False,
    )
    return SAM3Grounding(model, transform, postprocessor, device)

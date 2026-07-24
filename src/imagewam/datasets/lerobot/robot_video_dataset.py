import hashlib
import os
import time
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from imagewam.utils.logging_config import get_logger
from imagewam.utils import misc, pytorch_utils
from imagewam.utils.mem_tools import PeriodicTrim
from accelerate import PartialState
logger = get_logger(__name__)

# export IMAGEWAM_MEM_TRIM_EVERY=50          
# export IMAGEWAM_M_TRIM_THRESHOLD=65536    
# export IMAGEWAM_M_MMAP_THRESHOLD=65536   
# export IMAGEWAM_M_TOP_PAD=0
# # export MALLOC_ARENA_MAX=2


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        val_split_level: str = "episode",
        global_sample_stride=1,
        sample_index_stride: int = 1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        robotwin_camera_layout: str = "compact_288x256",
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        require_text_cache: bool = True,
        qwen_text_cache_dir: Optional[str] = None,
        qwen_context_len: int = 128,
        qwen_text_cache_format: str = "qwen2_5_vl",
        mage_text_cache_dir: Optional[str] = None,
        endpoint_frames_only: bool = False,
        nonidle_filter_path: Optional[str] = None,
        profile_getitem: bool = False,
        condition_frame_augmentation: Optional[dict] = None,
        video_augmentation: Optional[dict] = None,
        hetero_bridge: Optional[dict] = None,
        lerobot_meta_cache: Optional[str] = None,
        arrow_cache_dir: Optional[str] = None,
        lerobot_backend: str = "v2",
        lerobot_v3_init_num_workers: int = 1,
        lerobot_v3_index_cache: Optional[str] = None,
        lerobot_v3_video_backend: Optional[str] = None,
        lerobot_tolerance_s: Optional[float] = None,
        episode_index_filter: Optional[dict] = None,
        slow_getitem_log_sec: float = 0.0,
        action_proprio_cache_path: Optional[str] = None,
        video_frame_cache_dir: Optional[str] = None,
    ):
        image_obs_indices = [0, num_frames - 1] if endpoint_frames_only else None
        self.slow_getitem_log_sec = float(
            os.environ.get("IMAGEWAM_SLOW_GETITEM_LOG_SEC", slow_getitem_log_sec)
        )
        effective_profile_getitem = bool(profile_getitem) or self.slow_getitem_log_sec > 0.0
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            val_split_level=val_split_level,
            global_sample_stride=global_sample_stride,
            sample_index_stride=sample_index_stride,
            image_obs_indices=image_obs_indices,
            nonidle_filter_path=nonidle_filter_path,
            profile_getitem=effective_profile_getitem,
            hetero_bridge=OmegaConf.to_container(hetero_bridge, resolve=True) if isinstance(hetero_bridge, DictConfig) else hetero_bridge,
            lerobot_meta_cache=lerobot_meta_cache,
            arrow_cache_dir=arrow_cache_dir,
            lerobot_backend=lerobot_backend,
            lerobot_v3_init_num_workers=lerobot_v3_init_num_workers,
            lerobot_v3_index_cache=lerobot_v3_index_cache,
            lerobot_v3_video_backend=lerobot_v3_video_backend,
            lerobot_tolerance_s=lerobot_tolerance_s,
            episode_index_filter=OmegaConf.to_container(episode_index_filter, resolve=True) if isinstance(episode_index_filter, DictConfig) else episode_index_filter,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.robotwin_camera_layout = str(robotwin_camera_layout)
        self.override_instruction = override_instruction
        self.is_training_set = is_training_set
        self.require_text_cache = bool(require_text_cache)
        self.qwen_text_cache_dir = qwen_text_cache_dir
        self.mage_text_cache_dir = mage_text_cache_dir
        self.qwen_context_len = int(qwen_context_len)
        self.qwen_text_cache_format = str(qwen_text_cache_format)
        self.endpoint_frames_only = bool(endpoint_frames_only)
        self.profile_getitem = effective_profile_getitem
        augmentation_cfg = video_augmentation if video_augmentation is not None else condition_frame_augmentation
        if augmentation_cfg is not None and is_training_set:
            # Hydra's instantiate(..., recursive=True) may have already built nested
            # _target_ configs; do not call instantiate() on an nn.Module twice.
            if isinstance(augmentation_cfg, torch.nn.Module):
                self.video_augmentation = augmentation_cfg
            else:
                self.video_augmentation = instantiate(augmentation_cfg)
        else:
            self.video_augmentation = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        # Per-worker periodic memory reclaim. Each __getitem__ leaves behind
        # small Python allocations (HF row dicts, list comprehensions,
        # pyav-decoded buffers, ...) that pile up as free chunks in the
        # worker's glibc arena. glibc only auto-trims when the free
        # top-of-heap exceeds M_TRIM_THRESHOLD (128 KB by default), so RSS
        # tends to ratchet up between auto-trims. Setting
        # IMAGEWAM_MEM_TRIM_EVERY=N forces gc.collect() + malloc_trim(0) every
        # N samples per worker (50-200 is a reasonable starting point).
        self._mem_trim = PeriodicTrim(
            every=int(os.environ.get("IMAGEWAM_MEM_TRIM_EVERY", "0")),
            do_gc=os.environ.get("IMAGEWAM_MEM_TRIM_GC", "1") != "0",
        )

        # Action/proprio cache support
        self.action_proprio_cache_path = action_proprio_cache_path
        self._ap_cache = None
        if self.action_proprio_cache_path is not None and os.path.exists(self.action_proprio_cache_path):
            _t0 = time.time()
            logger.info(f"Pre-loading action/proprio cache from {self.action_proprio_cache_path}...")
            self._ap_cache = torch.load(self.action_proprio_cache_path, map_location="cpu", weights_only=False)
            logger.info(f"Pre-loaded {len(self._ap_cache)} action/proprio entries in {time.time()-_t0:.1f}s")

        # Video frame cache support (on-demand loading, OS page cache shared across processes)
        self.video_frame_cache_dir = video_frame_cache_dir
        self._video_frame_ondemand = False
        self._video_frame_mmap = None
        self._video_frame_mmap_index = None

        if self.video_frame_cache_dir is not None and os.path.isdir(self.video_frame_cache_dir):
            import glob as _glob

            # Check for mmap format first (preferred)
            mmap_index_path = os.path.join(self.video_frame_cache_dir, "video_frames_index.pt")
            mmap_data_path = os.path.join(self.video_frame_cache_dir, "video_frames.mmap")

            if os.path.exists(mmap_index_path) and os.path.exists(mmap_data_path):
                # Load mmap cache
                _t0 = time.time()
                import numpy as np

                self._video_frame_mmap_index = torch.load(mmap_index_path, map_location="cpu", weights_only=False)
                # Open mmap in read-only mode for memory efficiency
                # Shape is (num_frames, flattened_size) for easier access
                self._video_frame_mmap = np.memmap(
                    mmap_data_path,
                    dtype=np.float32 if "float32" in self._video_frame_mmap_index["frame_dtype"] else np.uint8,
                    mode="r",
                    shape=(
                        self._video_frame_mmap_index["num_frames"],
                        int(np.prod(self._video_frame_mmap_index["frame_shape"])),  # flattened size
                    ),
                )
                logger.info(f"Video frame cache: mmap mode ({self._video_frame_mmap_index['num_frames']} frames, "
                           f"{self._video_frame_mmap_index['total_size_bytes']/1024**3:.2f}GB, loaded in {time.time()-_t0:.2f}s)")
            else:
                # Fall back to individual .pt files
                _test = _glob.glob(os.path.join(self.video_frame_cache_dir, "frame_*.pt"))
                if _test:
                    self._video_frame_ondemand = True
                    logger.info(f"Video frame cache: on-demand mode ({len(_test)} files in {self.video_frame_cache_dir}, "
                                f"no pre-load, OS page cache shared across processes)")

        # Qwen text cache preloading (for faster training)
        self._qwen_text_cache = None
        if self.qwen_text_cache_dir is not None:
            preload_qwen_cache = os.environ.get("IMAGEWAM_PRELOAD_QWEN_CACHE", "1") != "0"
            if preload_qwen_cache:
                self._preload_qwen_text_cache()

    def __len__(self):
        return len(self.lerobot_dataset)

    def _mage_cache_key(self, instruction: str, reference: torch.Tensor) -> str:
        # Match precompute_text_cache.py.  The precompute dataset disables
        # augmentation, so hash the cropped/normalized pre-augmentation frame.
        reference = self.crop_transform(reference.unsqueeze(0))[0]
        reference = (reference * 2.0 - 1.0).float().contiguous().cpu()
        digest = hashlib.sha256(instruction.encode("utf-8"))
        digest.update(reference.numpy().tobytes())
        return digest.hexdigest()

    def _get_from_video_frame_cache(self, idx):
        """Build sample from video frame cache (uint8) + action/proprio cache.

        Video frames are loaded from cache as uint8 [C, T, H, W] (resize-only),
        then processed with augmentation → crop → normalize during training.

        Cached frames: resize (aspect-ratio preserving) → stored as uint8 [0, 255]
        Training adds: augmentation (if enabled) → crop → normalize

        This allows effective data augmentation on larger frames.
        Action/proprio come from _ap_cache.
        """
        profile = {} if self.profile_getitem else None
        t0 = time.perf_counter()

        def _mark(name: str):
            nonlocal t0
            if profile is None:
                return
            now = time.perf_counter()
            profile[f"robot.{name}"] = now - t0
            t0 = now

        # Load action/proprio from cache
        ap = self._ap_cache[idx]
        action = ap["action"]
        proprio = ap["proprio"]
        action_is_pad = ap["action_is_pad"]
        task = ap["task"]
        _mark("ap_cache_load")

        # Load video frame from cache [C, T, H, W] float32 [0, 1]
        # Cache contains already-processed data: processor + resize + concat + permute
        frame_path = os.path.join(self.video_frame_cache_dir, f"frame_{idx:07d}.pt")
        video = torch.load(frame_path, map_location="cpu", weights_only=False)
        _mark("frame_cache_load")

        # Reconstruct multi-camera format for augmentation
        # Cache is [C, T, H, W] where W = num_cameras * original_W (for horizontal concat)
        num_cameras = len(self.shape_meta["images"])
        C, T, H, W_total = video.shape

        # Convert [C, T, H, W] to [T, C, H, W]
        video = video.permute(1, 0, 2, 3)  # [T, C, H, W_total]

        # Split concatenated cameras
        if self.concat_multi_camera == "horizontal":
            # Horizontal concat: [T, C, H, W_total] where W_total = num_cameras * W
            W_per_cam = W_total // num_cameras
            videos_per_cam = torch.chunk(video, num_cameras, dim=-1)  # list of [T, C, H, W_per_cam]
            video = torch.stack(videos_per_cam, dim=0)  # [num_cameras, T, C, H, W_per_cam]
        elif self.concat_multi_camera == "vertical":
            # Vertical concat: [T, C, H_total, W] where H_total = num_cameras * H
            H_per_cam = H // num_cameras
            videos_per_cam = torch.chunk(video, num_cameras, dim=-2)  # list of [T, C, H_per_cam, W]
            video = torch.stack(videos_per_cam, dim=0)  # [num_cameras, T, C, H_per_cam, W]
        else:
            # For robotwin or other complex layouts, skip augmentation in cache mode
            video = None
        # logger.info(f"[DEBUG] Loaded video from cache: {frame_path}, shape={video.shape if video is not None else 'N/A'}")
        mage_reference = torch.cat([video[i] for i in range(num_cameras)], dim=-1) if video is not None else None
        # Apply video augmentation if enabled and we reconstructed multi-camera format
        if video is not None and self.video_augmentation is not None:
            video = self.video_augmentation(video)  # [num_cameras, T, C, H, W] → [num_cameras, T, C, H, W]
        _mark("augmentation")

        # Re-concatenate cameras
        if video is not None:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T, C, num_cameras*H, W]
            else:
                # Fallback: just use original cached video
                video = torch.load(frame_path, map_location="cpu", weights_only=False).permute(1, 0, 2, 3)
        _mark("reconcat")

        # Apply crop (no resize - already done in precompute)
        video = self.crop_transform(video)  # [T, C, H, W] float [0, 1]

        # Normalize: [0, 1] → [-1, 1]
        video = video * 2.0 - 1.0  # [T, C, H, W] float [-1, 1]
        video = video.permute(1, 0, 2, 3)  # [T, C, H, W] → [C, T, H, W] float [-1, 1]
        _mark("crop_normalize")

        # Get image_is_pad from cache (or zeros)
        image_is_pad = ap.get("image_is_pad", None)
        if image_is_pad is None:  # Estimate from video shape
            image_is_pad = torch.zeros(video.shape[1], dtype=torch.bool)

        # Get instruction
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        _mark("instruction")

        # Get text context
        data = {
            "video": video,  # [C, T, H, W] float [-1,1]
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "instruction": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": torch.zeros_like(action_is_pad),
        }
        data["_mage_cache_key"] = self._mage_cache_key(instruction, mage_reference[0])

        if self.require_text_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
            _mark("text_cache")

        if self.qwen_text_cache_dir is not None:
            text_hidden_states, text_attention_mask = self._get_cached_qwen_context(instruction)
            data["text_hidden_states"] = text_hidden_states
            data["text_attention_mask"] = text_attention_mask
            _mark("qwen_cache")

        if profile is not None:
            data["_profile"] = profile

        return data

    def _get_from_mmap_cache(self, idx):
        """Build sample from memory-mapped video frame cache + action/proprio cache.

        Uses np.memmap for fast random access without torch.load() overhead.
        OS automatically manages page cache for efficient access patterns.
        """
        profile = {} if self.profile_getitem else None
        t0 = time.perf_counter()

        def _mark(name: str):
            nonlocal t0
            if profile is None:
                return
            now = time.perf_counter()
            profile[f"robot.{name}"] = now - t0
            t0 = now

        # Load action/proprio from cache
        ap = self._ap_cache[idx]
        action = ap["action"]
        proprio = ap["proprio"]
        action_is_pad = ap["action_is_pad"]
        task = ap["task"]
        _mark("ap_cache_load")

        # Load video frame from mmap (fast: direct array access)
        # mmap_array[idx] returns a flattened 1D array
        video_numpy_flat = self._video_frame_mmap[idx]

        # Reshape back to original shape [C, T, H, W]
        original_shape = tuple(self._video_frame_mmap_index["frame_shape"])
        video_numpy = video_numpy_flat.reshape(original_shape)

        # Convert to torch tensor (this creates a copy, but it's necessary for later operations)
        video = torch.from_numpy(video_numpy.copy()).float() / 255.0 if video_numpy.dtype == np.uint8 else torch.from_numpy(video_numpy.copy())
        _mark("frame_mmap_load")

        # Rest is the same as _get_from_video_frame_cache
        num_cameras = len(self.shape_meta["images"])

        # Convert [C, T, H, W] to [T, C, H, W]
        if video.ndim == 4:  # [C, T, H, W]
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
        elif video.ndim == 3:  # [C, H, W] - single frame
            video = video.unsqueeze(1)  # [C, 1, H, W]
            video = video.permute(1, 0, 2, 3)  # [1, C, H, W]

        # Split concatenated cameras for augmentation
        if self.concat_multi_camera == "horizontal":
            C, T, H, W_total = video.shape
            W_per_cam = W_total // num_cameras
            videos_per_cam = torch.chunk(video, num_cameras, dim=-1)
            video = torch.stack(videos_per_cam, dim=0)  # [num_cameras, T, C, H, W]
        elif self.concat_multi_camera == "vertical":
            C, T, H_total, W = video.shape
            H_per_cam = H_total // num_cameras
            videos_per_cam = torch.chunk(video, num_cameras, dim=-2)
            video = torch.stack(videos_per_cam, dim=0)  # [num_cameras, T, C, H_per_cam, W]
        else:
            video = None

        mage_reference = torch.cat([video[i] for i in range(num_cameras)], dim=-1) if video is not None else None
        # Apply augmentation
        if video is not None and self.video_augmentation is not None:
            video = self.video_augmentation(video)
        _mark("augmentation")

        # Re-concatenate cameras
        if video is not None:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)

        # Apply crop and normalize
        if video is not None:
            video = self.crop_transform(video)
            video = video * 2.0 - 1.0
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W] → [C, T, H, W]
        _mark("crop_normalize")

        # Get image_is_pad from cache (or zeros)
        image_is_pad = ap.get("image_is_pad", None)
        if image_is_pad is None:
            image_is_pad = torch.zeros(video.shape[1], dtype=torch.bool)

        # Get instruction
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        _mark("instruction")

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "instruction": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": torch.zeros_like(action_is_pad),
        }
        data["_mage_cache_key"] = self._mage_cache_key(instruction, mage_reference[0])

        if self.require_text_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
            _mark("text_cache")

        if self.qwen_text_cache_dir is not None:
            text_hidden_states, text_attention_mask = self._get_cached_qwen_context(instruction)
            data["text_hidden_states"] = text_hidden_states
            data["text_attention_mask"] = text_attention_mask
            _mark("qwen_cache")

        if profile is not None:
            data["_profile"] = profile

        return data

    @staticmethod
    def _robotwin_camera_sizes(layout: str) -> tuple[list[int], list[int], list[int]]:
        layout = str(layout).strip().lower()
        if layout in {"compact", "compact_288x256", "288x256"}:
            return [192, 256], [96, 128], [96, 128]
        if layout in {"legacy", "legacy_384x320", "384x320"}:
            return [256, 320], [128, 160], [128, 160]
        raise ValueError(
            f"Unsupported robotwin_camera_layout={layout!r}. "
            "Expected one of: compact_288x256, legacy_384x320."
        )

    def _get(self, idx):
        # Fast path: mmap video cache (preferred) or individual .pt files + action/proprio cache
        if self._video_frame_mmap is not None and self._ap_cache is not None and idx in self._ap_cache:
            return self._get_from_mmap_cache(idx)
        if self._video_frame_ondemand and self._ap_cache is not None and idx in self._ap_cache:
            return self._get_from_video_frame_cache(idx)

        sample_idx = idx
        sample = None
        profile = {} if self.profile_getitem else None
        t0 = time.perf_counter()

        def _mark(name: str):
            nonlocal t0
            if profile is None:
                return
            now = time.perf_counter()
            profile[f"robot.{name}"] = now - t0
            t0 = now

        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if profile is not None and "_profile" in sample:
                profile.update(sample["_profile"])
            _mark("lerobot_dataset_get")

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        _mark("padding_retry")

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if not self.endpoint_frames_only:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if not self.endpoint_frames_only:
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        if not self.endpoint_frames_only:
            image_is_pad = image_is_pad[self.video_sample_indices]
        _mark("video_select")

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.video_augmentation is not None:
            # # Debug: check video type before augmentation (only first time)
            # if not hasattr(self, '_augmentation_debug_printed'):
            #     print(f"[DEBUG] Before video_augmentation:")
            #     print(f"  video dtype: {video.dtype}")
            #     print(f"  video shape: {video.shape}")
            #     print(f"  video min: {video.min()}, max: {video.max()}")
            #     print(f"  is uint8: {video.dtype == torch.uint8}")
            #     print(f"  is_floating_point: {video.is_floating_point()}")
            #     self._augmentation_debug_printed = True
            video = self.video_augmentation(video)

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            top_size, left_size, right_size = self._robotwin_camera_sizes(self.robotwin_camera_layout)
            cam_top = transforms_F.resize(
                video[0],
                size=top_size,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=left_size,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=right_size,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]
        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        _mark("image_postprocess")

        # Check if we can use cached action/proprio
        if self._ap_cache is not None and idx in self._ap_cache:
            # Use cached data, skip processor action/proprio processing
            ap = self._ap_cache[idx]
            action = ap["action"]            # [T-1, action_dim] normalized
            proprio = ap["proprio"]          # [T-1, proprio_dim] normalized (already sliced in cache)
            action_is_pad = ap["action_is_pad"]  # [T-1] bool
            task = ap["task"]                # instruction string
        else:
            # Original path: get from sample
            # Proxy (from lerobot):
            #   action: [num_frames-1, action_dim] # start from t0, except the last frame
            #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
            action = sample["action"] # [T-1, action_dim]
            proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
            action_is_pad = sample["action_is_pad"]

        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        expected_video_transitions = (self.num_frames - 1) // self.action_video_freq_ratio
        if self.endpoint_frames_only:
            expected_video_transitions = max(expected_video_transitions, 1)
        else:
            expected_video_transitions = video.shape[1] - 1
        if action.shape[0] % expected_video_transitions != 0:
            raise ValueError(
                f"`action` horizon must be divisible by video transitions, got {action.shape[0]} and {expected_video_transitions}"
            )

        # Get task/instruction
        if self._ap_cache is None or idx not in self._ap_cache:
            task = sample["instruction"]

        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "instruction": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if "action_dim_is_pad" in sample:
            data["action_dim_is_pad"] = sample["action_dim_is_pad"]
        if "proprio_dim_is_pad" in sample:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if "embodiment" in sample:
            data["embodiment"] = sample["embodiment"]
        if self.require_text_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            # NOTE: to keep consistent with wan2.2's behavior
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
        if self.qwen_text_cache_dir is not None:
            text_hidden_states, text_attention_mask = self._get_cached_qwen_context(instruction)
            data["text_hidden_states"] = text_hidden_states
            data["text_attention_mask"] = text_attention_mask
        _mark("qwen_cache")
        if profile is not None:
            data["_profile"] = profile
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/omnigen2/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def _preload_qwen_text_cache(self):
        """Preload all Qwen text embedding cache files into memory."""
        if self.qwen_text_cache_dir is None or not os.path.isdir(self.qwen_text_cache_dir):
            logger.warning(f"Qwen text cache dir not found: {self.qwen_text_cache_dir}")
            return

        import glob as _glob

        suffix_by_format = {
            "qwen2_5_vl": "qwen2_5_vl",
            "qwen3_flux2": "qwen3_flux2",
        }
        if self.qwen_text_cache_format not in suffix_by_format:
            logger.warning(f"Unsupported qwen_text_cache_format={self.qwen_text_cache_format!r}")
            return

        suffix = suffix_by_format[self.qwen_text_cache_format]
        pattern = f"*.{suffix}_len{self.qwen_context_len}.pt"
        cache_files = _glob.glob(os.path.join(self.qwen_text_cache_dir, pattern))

        if not cache_files:
            logger.warning(f"No Qwen text cache files found matching {pattern} in {self.qwen_text_cache_dir}")
            return

        _t0 = time.time()
        self._qwen_text_cache = {}
        loaded_count = 0
        failed_count = 0

        for cache_path in cache_files:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                # Extract hash from filename (e.g., "abc123.qwen3_flux2_len128.pt" -> "abc123")
                filename = os.path.basename(cache_path)
                hash_key = filename.split(".")[0]
                self._qwen_text_cache[hash_key] = {
                    "text_hidden_states": payload["text_hidden_states"],
                    "text_attention_mask": payload["text_attention_mask"].bool(),
                }
                loaded_count += 1
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to load Qwen cache file {cache_path}: {e}")

        logger.info(f"Preloaded {loaded_count} Qwen text cache entries in {time.time()-_t0:.1f}s "
                    f"(failed: {failed_count}, dir: {self.qwen_text_cache_dir})")

    def _get_cached_qwen_context(self, prompt: str):
        if self.qwen_text_cache_dir is None:
            raise ValueError("qwen_text_cache_dir is not set.")

        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        # Fast path: use in-memory cache if preloaded
        if self._qwen_text_cache is not None:
            if hashed in self._qwen_text_cache:
                cached = self._qwen_text_cache[hashed]
                context = cached["text_hidden_states"]
                context_mask = cached["text_attention_mask"]
                return context, context_mask
            else:
                # Fallback to disk load if hash not in memory cache (shouldn't happen if all preloaded)
                logger.warning(f"Qwen cache hash {hashed} not found in preloaded cache, falling back to disk load")

        # Slow path: load from disk (original code)
        cache_dir = self.qwen_text_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        suffix_by_format = {
            "qwen2_5_vl": "qwen2_5_vl",
            "qwen3_flux2": "qwen3_flux2",
        }
        if self.qwen_text_cache_format not in suffix_by_format:
            raise ValueError(
                f"Unsupported qwen_text_cache_format={self.qwen_text_cache_format!r}; "
                "expected 'qwen2_5_vl' or 'qwen3_flux2'."
            )
        suffix = suffix_by_format[self.qwen_text_cache_format]
        cache_path = os.path.join(cache_dir, f"{hashed}.{suffix}_len{self.qwen_context_len}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing Qwen text embedding cache: {cache_path}. "
                "Use qwen_text_cache_format='qwen2_5_vl' with scripts/omnigen2/precompute_qwen_embeds.py, "
                "or qwen_text_cache_format='qwen3_flux2' with scripts/flux2/precompute_flux2_qwen3_embeds.py."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["text_hidden_states"]
        context_mask = payload["text_attention_mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `text_hidden_states` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `text_attention_mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.qwen_context_len:
            raise ValueError(
                f"Cached qwen_context_len mismatch: expected {self.qwen_context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.qwen_context_len:
            raise ValueError(
                f"Cached qwen mask length mismatch: expected {self.qwen_context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return context, context_mask

    def __getitem__(self, idx):
        t0 = time.perf_counter()
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
            idx = random_idx
        if self.mage_text_cache_dir is not None:
            cache_key = data.get("_mage_cache_key")
            if cache_key is None:
                raise ValueError("Mage text cache key is unavailable for this dataset path")
            cache_path = os.path.join(self.mage_text_cache_dir, f"{cache_key}.pt")
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Missing Mage text cache: {cache_path}")
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Clone tensors loaded from cache so the default DataLoader
            # collator can allocate writable batch storage.
            hidden_states = payload["context"].clone()
            attention_mask = payload["context_mask"].bool().clone()
            target_len = self.context_len
            if hidden_states.shape[0] > target_len:
                hidden_states = hidden_states[:target_len]
                attention_mask = attention_mask[:target_len]
            elif hidden_states.shape[0] < target_len:
                pad_len = target_len - hidden_states.shape[0]
                hidden_states = torch.cat([
                    hidden_states,
                    hidden_states.new_zeros((pad_len, hidden_states.shape[1])),
                ], dim=0)
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_zeros((pad_len,)),
                ], dim=0)
            data["mage_text_hidden_states"] = hidden_states
            data["mage_text_attention_mask"] = attention_mask
        data.pop("_mage_cache_key", None)
        elapsed = time.perf_counter() - t0
        if self.slow_getitem_log_sec > 0.0 and elapsed >= self.slow_getitem_log_sec:
            profile = data.get("_profile", {})
            slow_parts = []
            if isinstance(profile, dict):
                time_profile = []
                counter_profile = []
                for key, value in profile.items():
                    if not isinstance(value, (int, float)):
                        continue
                    row = (key, float(value))
                    if (
                        key.endswith(".calls")
                        or key.endswith(".requested_frames")
                        or key.endswith(".frame_span")
                        or key.endswith(".max_frame_index")
                        or key.endswith(".frame_span_per_call_max")
                        or key.endswith(".max_frame_index_per_call_max")
                        or key.endswith(".frames_decoded")
                        or key.endswith(".pyav_eof_fallbacks")
                    ):
                        counter_profile.append(row)
                    else:
                        time_profile.append(row)
                for key, value in sorted(time_profile, key=lambda item: item[1], reverse=True)[:12]:
                    slow_parts.append(f"{key}={value:.3f}s")
                remaining = max(0, 12 - len(slow_parts))
                for key, value in sorted(counter_profile, key=lambda item: item[1], reverse=True)[:remaining]:
                    slow_parts.append(f"{key}={value:.1f}")
            logger.warning(
                "[slow-getitem] idx=%s elapsed=%.3fs %s",
                idx,
                elapsed,
                " | ".join(slow_parts),
            )
        # Force glibc to return free pages periodically (no-op when disabled).
        self._mem_trim.tick()
        return data

"""Batched LIBERO-Plus evaluation worker for ImageWAM.

Unlike ``eval_libero_single.py`` (one process per task_id, model loaded every
time), this worker loads the model ONCE and then rolls out a *chunk* of
task_ids in the same process. This is what makes LIBERO-Plus tractable:
Plus requires ``num_trials=1`` per task_id (each task_id is already an
independent perturbation sample), so with 2402 spatial tasks the
single-task launcher would pay the ~135s model-load tax 2402 times. Here
we pay it once per worker instead.

The original LIBERO evaluation path (``eval_libero_single.py`` +
``run_libero_parallel_test.sh``) is intentionally left untouched. This file
reuses the per-episode rollout core (``run_single_episode``) but wraps it
with its own task loop, resume-skip logic, and two video knobs.

Result files are written in the exact same contract as the single launcher
so that ``summarize_results.py`` and the scheduler's completion detection
work unchanged:

    $OUTPUT_DIR/<suite>/gpu{worker_id}_task{task_id}_results.json

The filename's ``gpu`` slot is occupied by ``worker_id`` (downstream only
parses ``task_id`` from it), while the JSON body's ``gpu_id`` field carries
the real CUDA device for log correlation.
"""

import glob
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

# try:
#     import rootutils
#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ``eval_libero_single`` does a bare ``from action_ensembler import ...`` which
# relies on the script's own directory being on sys.path[0] (true when run as
# ``python experiments/libero/eval_libero_single.py``, which is how the
# scheduler launches it). Add it explicitly so this module is robust to being
# imported or run via ``-m`` / ``-c`` too.
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Reuse the per-episode rollout core and all helpers from the single launcher.
from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    run_single_episode,
)
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    get_libero_env,
    save_prediction_video,
    save_rollout_video,
)
from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor  # noqa: E402
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from imagewam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402

# The custom resolvers (eval/max/split) are already registered by
# ``eval_libero_single`` (imported above), which is always imported before this
# point. Re-registering would raise, so we register defensively only if a
# resolver is somehow missing.
for _name, _fn in (("eval", eval), ("max", lambda x: max(x)), ("split", lambda s, idx: s.split("/")[int(idx)])):
    try:
        OmegaConf.register_new_resolver(_name, _fn)
    except ValueError:
        pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _read_chunk(chunk_file: Path) -> list[tuple[str, int]]:
    """Read a chunk file (one ``suite,task_id`` per line) into a list of tuples."""
    tasks: list[tuple[str, int]] = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                logging.warning("Skipping malformed chunk line: %s", line)
                continue
            tasks.append((parts[0].strip(), int(parts[1].strip())))
    return tasks


def _result_file_exists(output_dir: Path, suite: str, task_id: int) -> bool:
    """A task is considered done if any ``gpu*_task{id}_results.json`` exists.

    Mirrors the scheduler's completion detection in
    ``run_libero_parallel_test.sh`` (``gpu*_task${task_id}_results.json`` glob),
    so resume-skip and the scheduler stay consistent regardless of which
    worker originally wrote the file.
    """
    pattern = str(output_dir / suite / f"gpu*_task{task_id}_results.json")
    return len(glob.glob(pattern)) > 0


def _run_single_task_batched(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: ImageWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    suite: str,
    task_id: int,
    worker_id: int,
    real_gpu_id: int,
    save_video: bool,
    video_budget: "VideoBudget",
    text_cache: "dict[str, tuple[Any, Any]] | None" = None,
) -> dict:
    """Per-task rollout for the batch worker.

    Structurally mirrors ``run_single_task`` from ``eval_libero_single.py`` but:
      * drives ``run_single_episode`` (the untouched rollout core) directly,
      * gates rollout/prediction video saving behind ``save_video`` and a
        per-worker ``video_budget`` (so 2402 Plus tasks do not write 2402 mp4s),
      * writes the result file in the single-launcher contract with
        ``worker_id`` in the filename and ``real_gpu_id`` in the body.
    """
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results: dict[str, Any] = {
        "task_suite": suite,
        "task_id": task_id,
        "task_description": task_description,
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    num_trials = int(cfg.EVALUATION.num_trials)
    allow_save = save_video and video_budget.has_quota()
    for trial_idx in range(num_trials):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            text_cache=text_cache,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        if allow_save:
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
            if visualize_future_video and len(predicted_future_video_clips) > 0:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_psnr))

    results["total_episodes"] = num_trials
    results["gpu_id"] = real_gpu_id
    results["worker_id"] = worker_id
    results["task_suite"] = suite
    results["task_id"] = task_id
    return results


class VideoBudget:
    """Caps how many tasks a worker will save rollout videos for.

    ``SAVE_VIDEO`` is the global on/off; ``MAX_VIDEOS_PER_WORKER`` bounds the
    total count so a 2402-task Plus run does not flood disk. Once the budget
    is spent, ``has_quota()`` returns False and the worker keeps producing
    result JSONs without mp4s.
    """

    def __init__(self, max_videos: int):
        self._max = int(max_videos)
        self._spent = 0

    def has_quota(self) -> bool:
        return self._spent < self._max

    def consume(self, n: int = 1) -> None:
        self._spent += n


def _load_model_once(cfg: DictConfig):
    """Model + processor + dataset stats, loaded exactly once per worker.

    Lifted verbatim from ``eval_single_process`` so behavior matches the
    single launcher byte-for-byte.
    """
    from experiments.libero.eval_libero_single import _resolve_model_cfg

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model_cfg = _resolve_model_cfg(cfg)
    logging.info("Resolved model_cfg keys: %s", list(OmegaConf.to_container(model_cfg).keys()))
    model = instantiate(model_cfg, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: ImageWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    return model, processor, action_horizon, input_w, input_h, model_device


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_batch_process(cfg: DictConfig):
    start_time = time.time()

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Only env_num=1 is supported in eval_libero_batch.py.")

    # Worker identity + its chunk of work.
    chunk_file = Path(cfg.EVALUATION.chunk_file)
    worker_id = int(cfg.EVALUATION.worker_id)
    real_gpu_id = int(cfg.gpu_id)
    if not chunk_file.exists():
        raise FileNotFoundError(f"Worker chunk file not found: {chunk_file}")

    save_video = bool(cfg.EVALUATION.get("save_video", False))
    max_videos = int(cfg.EVALUATION.get("max_videos_per_worker", 50))

    output_dir = Path(cfg.EVALUATION.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve benchmark dict once; suites are looked up per task.
    benchmark_dict = benchmark.get_benchmark_dict()

    # Load model exactly once for the whole chunk.
    logging.info(
        "[worker %d] Loading model once for chunk %s (save_video=%s, max_videos=%d)",
        worker_id,
        chunk_file,
        save_video,
        max_videos,
    )
    model, processor, action_horizon, input_w, input_h, model_device = _load_model_once(cfg)
    logging.info("[worker %d] Model loaded in %.1fs", worker_id, time.time() - start_time)

    # Load text embedding cache if configured (saves VRAM by skipping text encoder)
    text_cache_dir = cfg.EVALUATION.get("text_cache_dir", None)
    text_cache: dict[str, tuple[Any, Any]] | None = None
    if text_cache_dir is not None and str(text_cache_dir).strip():
        from experiments.libero.eval_libero_single import _load_text_cache
        text_cache = _load_text_cache(str(text_cache_dir))

    tasks = _read_chunk(chunk_file)
    total = len(tasks)
    logging.info("[worker %d] Chunk has %d tasks", worker_id, total)

    video_budget = VideoBudget(max_videos)

    # Group-by-suite lazily: benchmark suites are cheap to instantiate but we
    # avoid re-creating the same suite object across tasks of that suite.
    suite_cache: dict[str, Any] = {}
    completed = 0
    failed = 0
    skipped = 0

    for idx, (suite, task_id) in enumerate(tasks):
        tag = f"[worker {worker_id}] [{idx + 1}/{total}] suite={suite} task_id={task_id}"

        # Resume-skip: result already exists (possibly from a previous run).
        if _result_file_exists(output_dir, suite, task_id):
            skipped += 1
            logging.info("%s SKIP (result already exists)", tag)
            continue

        if suite not in suite_cache:
            suite_cache[suite] = benchmark_dict[suite]()
        task_suite = suite_cache[suite]
        try:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
        except Exception as exc:  # benchmark lookup failure
            failed += 1
            logging.error("%s FAILED to fetch task/init_states: %s\n%s", tag, exc, traceback.format_exc())
            continue

        # Plus mandates num_trials=1; pad defensively in case a suite returns
        # fewer init states (matches the single launcher's guard).
        num_trials = int(cfg.EVALUATION.num_trials)
        while len(initial_states) < num_trials:
            initial_states.extend(initial_states[: (num_trials - len(initial_states))])

        suite_out = output_dir / suite
        suite_out.mkdir(parents=True, exist_ok=True)
        video_dir = suite_out / "videos"
        if save_video:
            video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_dir = suite_out / "predicted_videos"
        if save_video and bool(cfg.EVALUATION.get("visualize_future_video", False)):
            predicted_video_dir.mkdir(parents=True, exist_ok=True)

        # Per-task temporary cfg override so reused helpers see this task's id.
        with open_dict(cfg):
            cfg.EVALUATION.task_id = task_id
            cfg.EVALUATION.task_suite_name = suite

        task_start = time.time()
        try:
            results = _run_single_task_batched(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                suite=suite,
                task_id=task_id,
                worker_id=worker_id,
                real_gpu_id=real_gpu_id,
                save_video=save_video,
                video_budget=video_budget,
                text_cache=text_cache,
            )
            results["duration"] = time.time() - task_start
            output_file = suite_out / f"gpu{worker_id}_task{task_id}_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            completed += 1
            if save_video and video_budget.has_quota():
                video_budget.consume()
            logging.info(
                "%s DONE successes=%d/%d duration=%.1fs (completed=%d failed=%d skipped=%d)",
                tag,
                results["successes"],
                num_trials,
                results["duration"],
                completed,
                failed,
                skipped,
            )
        except Exception as exc:
            failed += 1
            # Skip-on-failure: do NOT write a result file, log and move on.
            # On a re-run the task will be retried (no result file to skip).
            logging.error("%s FAILED during rollout: %s\n%s", tag, exc, traceback.format_exc())
            continue

    total_duration = time.time() - start_time
    logging.info(
        "[worker %d] Chunk finished: completed=%d failed=%d skipped=%d total=%d elapsed=%.1fs",
        worker_id,
        completed,
        failed,
        skipped,
        total,
        total_duration,
    )
    print(
        f"[worker {worker_id}] DONE: completed={completed} failed={failed} "
        f"skipped={skipped} total={total} elapsed={total_duration:.1f}s"
    )
    return {
        "worker_id": worker_id,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "elapsed": total_duration,
    }


if __name__ == "__main__":
    eval_batch_process()

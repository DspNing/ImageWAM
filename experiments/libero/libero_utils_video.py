"""Utils for evaluating policies in LIBERO simulation environments with video rollout visualization."""

import math
import time
import pathlib

import imageio
from PIL import Image, ImageDraw
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from imagewam.utils.video_io import save_mp4

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def get_libero_env(task, resolution, seed, env_num=1):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing

    # [yc] wrist image
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing

    return {
        "image": img,
        "wrist_image": wrist_img
    }

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None, fps=24):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video_vertical(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
    target_size=448,
):
    """Saves an MP4 comparison of ground-truth and predicted future frames with VERTICAL concatenation.

    This version concatenates GT and predicted frames VERTICALLY (GT on top, pred on bottom),
    unlike the original horizontal version.

    Args:
        rollout_dir: Directory to save the video
        gt_frames: List of ground truth frames
        pred_frames: List of predicted frames
        idx: Episode index
        replan_idx: Replan index
        success: Whether the episode succeeded
        task_description: Task description string
        log_file: Optional log file to write to
        fps: Frames per second for output video
        target_size: Target size for resizing (width and height)
    """
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched_frames = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        # Process GT frame - use only "image" key if dict
        if isinstance(gt_frame, dict):
            if "image" in gt_frame:
                value = gt_frame["image"]
                gt_image = np.array(value) if isinstance(value, Image.Image) else value
            else:
                # Fallback to first value
                value = list(gt_frame.values())[0]
                gt_image = np.array(value) if isinstance(value, Image.Image) else value
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        # Process prediction frame
        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        # Resize both frames to half target height (224x448 each)
        gt_image = np.array(Image.fromarray(gt_image).resize((target_size, target_size // 2), resample=Image.BILINEAR))
        pred_image = np.array(Image.fromarray(pred_image).resize((target_size, target_size // 2), resample=Image.BILINEAR))

        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "GT", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "Pred", fill=(255, 255, 255))

        # VERTICAL concatenation: GT on top, prediction on bottom
        stitched_frames.append(
            Image.fromarray(np.concatenate([np.array(gt_pil), np.array(pred_pil)], axis=0))
        )

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--replan={replan_tag}--gt-pred-vertical.mp4"
    )
    save_mp4(stitched_frames, mp4_path, fps=fps)
    print(f"Saved vertical concatenated prediction MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved vertical concatenated prediction MP4 at path {mp4_path}\n")
    return mp4_path


def save_rollout_video_vertical(
    rollout_dir,
    gt_rollout_images,
    pred_rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    fps=24,
    target_size=448,
):
    """Saves an MP4 replay with GT and predicted rollouts stacked vertically.

    Args:
        rollout_dir: Directory to save the video
        gt_rollout_images: List of ground truth rollout frames (each frame can be dict or array)
        pred_rollout_images: List of predicted rollout frames (each frame can be dict or array)
        idx: Episode index
        success: Whether the episode succeeded
        task_description: Task description string
        log_file: Optional log file to write to
        fps: Frames per second for output video
        target_size: Target size for resizing (width and height)
    """
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}--rollout-vertical.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)

    num_frames = min(len(gt_rollout_images), len(pred_rollout_images))

    for i in range(num_frames):
        gt_img = gt_rollout_images[i]
        pred_img = pred_rollout_images[i]

        # Process GT frame - use only "image" key if dict
        if isinstance(gt_img, dict):
            if "image" in gt_img:
                value = gt_img["image"]
                value_array = np.array(value) if isinstance(value, Image.Image) else value
            else:
                # Fallback to first value
                value = list(gt_img.values())[0]
                value_array = np.array(value) if isinstance(value, Image.Image) else value
            gt_frame = np.array(value_array)
        elif isinstance(gt_img, Image.Image):
            gt_frame = np.array(gt_img.convert("RGB"))
        else:
            gt_frame = np.array(gt_img)

        # Process prediction frame - use only "image" key if dict
        if isinstance(pred_img, dict):
            if "image" in pred_img:
                value = pred_img["image"]
                value_array = np.array(value) if isinstance(value, Image.Image) else value
            else:
                # Fallback to first value
                value = list(pred_img.values())[0]
                value_array = np.array(value) if isinstance(value, Image.Image) else value
            pred_frame = np.array(value_array)
        elif isinstance(pred_img, Image.Image):
            pred_frame = np.array(pred_img.convert("RGB"))
        else:
            pred_frame = np.array(pred_img)

        # Resize both frames to half target height (224x448 each)
        gt_frame = np.array(Image.fromarray(gt_frame).resize((target_size, target_size // 2), resample=Image.BILINEAR))
        pred_frame = np.array(Image.fromarray(pred_frame).resize((target_size, target_size // 2), resample=Image.BILINEAR))

        # Add labels
        gt_pil = Image.fromarray(gt_frame)
        ImageDraw.Draw(gt_pil).text((10, 10), "GT", fill=(255, 255, 255))
        gt_frame = np.array(gt_pil)

        pred_pil = Image.fromarray(pred_frame)
        ImageDraw.Draw(pred_pil).text((10, 10), "Pred", fill=(255, 255, 255))
        pred_frame = np.array(pred_pil)

        # VERTICAL concatenation: GT on top, prediction on bottom
        stacked_frame = np.concatenate([gt_frame, pred_frame], axis=0)
        video_writer.append_data(stacked_frame)

    video_writer.close()
    print(f"Saved vertical rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved vertical rollout MP4 at path {mp4_path}\n")
    return mp4_path


def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = (v > 0.5)
    return np.asarray(bin_val, dtype=np.float32)


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

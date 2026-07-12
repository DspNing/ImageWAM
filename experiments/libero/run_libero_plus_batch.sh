#!/bin/bash
# LIBERO-Plus batched evaluation scheduler for ImageWAM.
#
# Unlike run_libero_parallel_test.sh (one short-lived process per task_id,
# model reloaded every time), this scheduler launches one LONG-LIVED worker
# per (GPU, slot). Each worker loads the model ONCE and rolls out a static
# chunk of task_ids (eval_libero_batch.py), so the ~135s model-load tax is
# paid once per worker instead of once per task.
#
# This is what makes LIBERO-Plus tractable: 2402 spatial tasks with
# num_trials=1 would otherwise reload the model 2402 times.
#
# The original LIBERO path (run_libero_parallel_test.sh + eval_libero_single.py)
# is untouched. This script is Plus-only and is selected by run_eval.sh
# when MODE=plus.
#
# Resume: workers skip any task whose gpu*_task{task_id}_results.json already
# exists, so re-running the same command after a crash/interrupt picks up where
# it left off. No queue state is needed.
#
# Completion is detected by counting result files (same contract as the
# single-task scheduler), so summarize_results.py works unchanged.

run_libero_plus_batch() {
    local task_list_file=$1
    echo "[plus-batch] task_file: $task_list_file"

    require_non_empty() {
        local var_name="$1"
        local var_val="${!var_name}"
        if [ -z "$var_val" ]; then
            echo "Error: required variable $var_name is not set"
            exit 1
        fi
    }

    # Basic configuration
    ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
    export ROOT_DIR
    RUN_ID=${RUN_ID:-"eval_$(date +%Y%m%d_%H%M%S)"}
    export RUN_ID
    RESULTS_SUBDIR=${RESULTS_SUBDIR:-"libero_plus"}
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/$RESULTS_SUBDIR/$RUN_ID"}
    export OUTPUT_DIR
    EXP_NAME=${EXP_NAME:-""}
    export EXP_NAME
    SESSION_NAME="libero_plus_batch"

    echo "[plus-batch] EXP_NAME: $EXP_NAME"
    mkdir -p "$OUTPUT_DIR"
    echo "[plus-batch] Results will be saved to: $OUTPUT_DIR"

    # Copy task_list_file into OUTPUT_DIR
    cp "$task_list_file" "$OUTPUT_DIR/"
    task_list_file="$OUTPUT_DIR/$(basename "$task_list_file")"
    echo "[plus-batch] Task list file copied to: $task_list_file"

    # GPU configuration (same parsing as run_libero_parallel_test.sh)
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        require_non_empty "NUM_GPUS"
        AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
    else
        AVAILABLE_GPUS=$CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo "$AVAILABLE_GPUS" | tr ',' '\n' | wc -l)
    fi
    export NUM_GPUS
    IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"
    echo "[plus-batch] NUM_GPUS: $NUM_GPUS, AVAILABLE_GPUS: $AVAILABLE_GPUS"

    require_non_empty "MAX_TASKS_PER_GPU"
    require_non_empty "NUM_TRIALS"

    # Workers per GPU (configurable). Total workers = NUM_GPUS * WORKERS_PER_GPU.
    # Default to MAX_TASKS_PER_GPU so a 2-GPU / MAX_TASKS_PER_GPU=2 run spawns
    # 4 long-lived workers and cuts model loads to 4 instead of thousands.
    WORKERS_PER_GPU=${WORKERS_PER_GPU:-$MAX_TASKS_PER_GPU}
    NUM_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
    echo "[plus-batch] WORKERS_PER_GPU: $WORKERS_PER_GPU -> total workers: $NUM_WORKERS"

    # Video knobs (see eval_libero_batch.py). Plus default: no videos.
    SAVE_VIDEO=${SAVE_VIDEO:-false}
    MAX_VIDEOS_PER_WORKER=${MAX_VIDEOS_PER_WORKER:-50}
    export SAVE_VIDEO MAX_VIDEOS_PER_WORKER

    TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
    CHUNK_DIR="$OUTPUT_DIR/chunks"
    mkdir -p "$TASK_LOG_DIR" "$CHUNK_DIR"

    # Checkpoint and config (same normalization as the single-task scheduler)
    CKPT=${CKPT:-""}
    export CKPT
    CONFIG=${CONFIG:-""}
    require_non_empty "CKPT"
    require_non_empty "CONFIG"
    CONFIG="${CONFIG#configs/}"
    CONFIG="${CONFIG#task/}"
    CONFIG="${CONFIG%.yaml}"
    export CONFIG

    echo "[plus-batch] CKPT: $CKPT"
    echo "[plus-batch] CONFIG: $CONFIG"
    echo "[plus-batch] NUM_TRIALS: $NUM_TRIALS"
    echo "[plus-batch] SAVE_VIDEO: $SAVE_VIDEO  MAX_VIDEOS_PER_WORKER: $MAX_VIDEOS_PER_WORKER"

    local total_tasks=$(wc -l < "$task_list_file")
    echo "[plus-batch] Total tasks: $total_tasks"

    # ---- Static chunking: split task_list into NUM_WORKERS chunks ----
    # Round-robin (interleaved) assignment: task at line N goes to worker
    # (N-1) % NUM_WORKERS. This spreads hard/easy tasks across all workers so
    # no worker gets stuck on a contiguous block of hard tasks.
    local chunk_files=()
    local wid=0
    while [ $wid -lt $NUM_WORKERS ]; do
        chunk_files+=("$CHUNK_DIR/chunk_worker${wid}.txt")
        : > "$CHUNK_DIR/chunk_worker${wid}.txt"
        wid=$((wid + 1))
    done
    local linenum=0
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        linenum=$((linenum + 1))
        wid=$(( (linenum - 1) % NUM_WORKERS ))
        echo "$line" >> "${chunk_files[$wid]}"
    done < "$task_list_file"
    for wid in "${!chunk_files[@]}"; do
        echo "[plus-batch] worker $wid -> $(wc -l < "${chunk_files[$wid]}") tasks (${chunk_files[$wid]})"
    done

    # ---- tmux session: one window per worker ----
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
        echo "[plus-batch] Deleted existing session '$SESSION_NAME'"
    fi
    tmux new-session -d -s "$SESSION_NAME" -n "w0"

    # ---- Launch one long-lived worker per window/pane ----
    echo "[plus-batch] Launching $NUM_WORKERS workers..."
    for wid in "${!chunk_files[@]}"; do
        local chunk_file="${chunk_files[$wid]}"
        # Map worker index -> (real GPU id, slot). Round-robin GPUs.
        local gpu_idx=$((wid % NUM_GPUS))
        local real_gpu_id=${GPU_ARRAY[$gpu_idx]}

        local window_id=$wid
        local pane_info="$window_id"
        if [ $window_id -gt 0 ]; then
            tmux new-window -t "$SESSION_NAME" -n "w$window_id" 2>/dev/null
        fi

        local log_file="$TASK_LOG_DIR/worker${wid}_gpu${real_gpu_id}.log"
        echo "[plus-batch] Launching worker $wid on GPU$real_gpu_id (pane $pane_info), chunk=$chunk_file, log=$log_file"

        tmux send-keys -t "$SESSION_NAME:$pane_info" "clear" C-m 2>/dev/null

        CONDA_ENV="${CONDA_ENV:-imagewam}"
        WORKER_THREADS="${WORKER_THREADS:-3}"

        # Build the worker command — activate conda env, then launch eval_libero_batch.py
        local model_paths=""
        if [ -n "${FLUX2_SRC:-}" ]; then
            model_paths+="model.flux2_src_path=$FLUX2_SRC "
            model_paths+="model.flux2_model_path=$FLUX2_MODEL_PATH "
            model_paths+="model.ae_model_path=$FLUX2_AE_MODEL_PATH "
            model_paths+="model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B} "
        fi
        if [ -n "${OMNIGEN2_SRC:-}" ]; then
            model_paths+="model.omnigen2_model_path=$OMNIGEN2_MODEL_PATH "
            model_paths+="model.omnigen2_vae_path=$OMNIGEN2_MODEL_PATH "
            model_paths+="model.qwen_path=$QWEN_MODEL_PATH "
        fi

        local launch_cmd="cd $ROOT_DIR && \
            ${WORKER_ENV_SOURCE:+source $WORKER_ENV_SOURCE && } \
            conda activate $CONDA_ENV && \
            export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
                OMP_NUM_THREADS=$WORKER_THREADS MKL_NUM_THREADS=$WORKER_THREADS \
                PYTHONPATH=${LIBERO_PKG_DIR}:${REPO_ROOT}/src && \
            CUDA_VISIBLE_DEVICES=$real_gpu_id python experiments/libero/eval_libero_batch.py \
                task=$CONFIG ckpt=$CKPT \
                EVALUATION.num_trials=$NUM_TRIALS \
                EVALUATION.output_dir=$OUTPUT_DIR \
                EVALUATION.action_horizon=${ACTION_HORIZON:-16} \
                EVALUATION.replan_steps=${REPLAN_STEPS:-12} \
                EVALUATION.num_inference_steps=10 \
                +EVALUATION.save_video=$SAVE_VIDEO \
                +EVALUATION.chunk_file=$chunk_file \
                +EVALUATION.worker_id=$wid \
                +EVALUATION.max_videos_per_worker=$MAX_VIDEOS_PER_WORKER \
                ${TEXT_CACHE_DIR:+EVALUATION.text_cache_dir=$TEXT_CACHE_DIR} \
                ${TEXT_CACHE_DIR:+model.load_text_encoder=false} \
                $model_paths \
                gpu_id=$real_gpu_id $EXTRA_ARGS > '${log_file}' 2>&1; \
            echo '[worker $wid] exited rc=\$?'"

        tmux send-keys -t "$SESSION_NAME:$pane_info" "$launch_cmd" C-m 2>/dev/null
        sleep 0.5
    done

    # ---- Wait for completion by counting result files ----
    local monitoring_interval=${MONITORING_INTERVAL:-15}
    local status_interval=${STATUS_INTERVAL:-60}
    local last_status_time=0
    local launch_time=$(date +%s)
    # Grace period: workers spend ~135s loading the model before python process
    # appears in the process table. Skip that check for this many seconds.
    local startup_grace=${STARTUP_GRACE:-300}

    echo "[plus-batch] Workers launched. Waiting for completion (total=$total_tasks)..."
    while true; do
        current_time=$(date +%s)
        local total_completed=$(find "$OUTPUT_DIR" -type f -name "gpu*_task*_results.json" 2>/dev/null | wc -l)
        if [ "$total_completed" -ge "$total_tasks" ]; then
            echo "[plus-batch] All $total_tasks tasks complete!"
            break
        fi

        # Detect total worker death while tasks remain
        local alive_workers=$(pgrep -fc "eval_libero_batch.py" 2>/dev/null | head -n1 | tr -d '[:space:]')
        alive_workers=${alive_workers:-0}
        local elapsed_since_launch=$((current_time - launch_time))
        if [ "$alive_workers" -eq 0 ] 2>/dev/null && [ "$total_completed" -lt "$total_tasks" ] && [ "$elapsed_since_launch" -ge "$startup_grace" ]; then
            echo "[plus-batch] WARNING: no live workers but $total_completed/$total_tasks done."
            echo "[plus-batch] Some workers crashed. Re-run to resume unfinished tasks (done ones are skipped)."
            echo "[plus-batch] Incomplete? Check $TASK_LOG_DIR and re-run."
            break
        fi

        if [ $((current_time - last_status_time)) -ge $status_interval ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Plus-batch Status ==="
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $total_completed / $total_tasks"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Live workers: $alive_workers / $NUM_WORKERS"
            for wid in "${!chunk_files[@]}"; do
                local cdone=$(find "$OUTPUT_DIR" -type f -name "gpu${wid}_task*_results.json" 2>/dev/null | wc -l)
                local csize=$(wc -l < "${chunk_files[$wid]}")
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   worker $wid: $cdone / $csize done"
            done
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================="
            last_status_time=$current_time
        fi

        sleep $monitoring_interval
    done

    # ---- Summarize ----
    echo "[plus-batch] Generating evaluation report..."
    python experiments/libero/summarize_results.py --output_dir="$OUTPUT_DIR"
    echo "[plus-batch] Done. Results: $OUTPUT_DIR"
}


# Entrypoint
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -lt 1 ]; then
        echo "Error: task file path is required"
        echo "Usage: $0 <task_file>"
        exit 1
    fi
    test_file="$1"
    run_libero_plus_batch "$test_file"
    exit $?
fi

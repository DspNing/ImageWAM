#!/bin/bash
# Precompute video frames cache for ImageWAM training
# This script parallelizes the frame precomputation across multiple shards

TASK=${1:-libero_flux2_klein_4b_base_imagewam}
OUTPUT_DIR=${2:-./data/video_frames_cache/libero_flux20}
NUM_SHARDS=${3:-8}
PRETRAINED_STATS=${4:-./data/dataset_stats.json}

echo "=========================================="
echo "Precomputing video frames for: $TASK"
echo "Output directory: $OUTPUT_DIR"
echo "Number of shards: $NUM_SHARDS"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"
mkdir -p runs/logs

# Kill any existing background jobs
trap "kill $(jobs -p) 2>/dev/null; exit" INT TERM

# Launch shards in background
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Starting shard $s..."
    python scripts/precompute_video_frames.py \
        --task "$TASK" \
        --output-dir "$OUTPUT_DIR" \
        --num-shards $NUM_SHARDS \
        --shard $s \
        --pretrained-norm-stats "$PRETRAINED_STATS" \
        --num-workers 4 \
        > runs/logs/vf_shard${s}.log 2>&1 &

    # Small delay to avoid overwhelming the system
    sleep 0.5
done

echo "All $NUM_SHARDS shards launched. Waiting for completion..."
echo "Log files: runs/logs/vf_shard*.log"

# Wait for all background jobs
wait

echo "=========================================="
echo "Video frame precomputation complete!"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To use this cache in training, update your config:"
echo "  data:"
echo "    train:"
echo "      video_frame_cache_dir: $OUTPUT_DIR"
echo "=========================================="

# Print summary
echo ""
echo "Summary:"
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    if [ -f "runs/logs/vf_shard${s}.log" ]; then
        DONE=$(grep -c "DONE:" runs/logs/vf_shard${s}.log || echo "0")
        ERRORS=$(grep "ERROR" runs/logs/vf_shard${s}.log | wc -l)
        echo "  Shard $s: DONE lines=$DONE, ERRORs=$ERRORS"
    fi
done

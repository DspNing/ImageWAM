CUDA_VISIBLE_DEVICES=4,5,6,7 \

torchrun --nproc_per_node=4 \
  --master_port=29530 \
  scripts/mage_flow/precompute_text_cache.py \
  --task libero_mage_flow_imagewam \
  --model-path ./checkpoints/mage_flow/Mage-Flow-Edit-Turbo \
  --output ./data/mage_text_cache/libero \
  --batch-size 64 \
  --num-workers 4

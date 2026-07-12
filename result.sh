MODE=plus   # master | plus | pro

OUT=./evaluate_results/libero_plus/eval_20260712_204812
# OUT=./evaluate_results/libero_plus/eval_20260628_232624-wan21
# OUT=./evaluate_results/libero_plus/eval_20260621_184344-wan22-release
find $OUT -name "gpu*_task*_results.json" | wc -l   # 完成数 / 总任务数
python experiments/libero/summarize_results.py --output_dir=$OUT   # 汇总(脚本跑完会自动调一次)
# 产出:summary.csv / summary.json / task_success_rates.csv


# libero plus
if [ "$MODE" = "plus" ]; then
    python experiments/libero/summarize_by_category.py --output_dir=$OUT
    python experiments/libero/summarize_plus_matrix.py --output_dir=$OUT
fi

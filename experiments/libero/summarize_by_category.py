"""Summarize LIBERO-Plus results grouped by perturbation category.

LIBERO-Plus tasks each carry a `category` (Camera Viewpoints / Robot Initial
States / Language Instructions / Light Conditions / Background Textures /
Sensor Noise / Objects Layout) in
``LIBERO-plus/libero/libero/benchmark/task_classification.json``. The stock
``summarize_results.py`` only breaks results down by suite, which hides how
the model does on each *type* of perturbation. This script joins each result
file (by ``task_suite`` + ``task_id``) to that classification and prints a
per-category table in the compact form:

    Camera  Robot  Language  Light  Background  Noise  Layout  Avg

It reads the same result-file contract as ``summarize_results.py``
(``gpu*_task{task_id}_results.json``), so partial runs are fine -- it just
reports the share of completed tasks per category alongside the accuracy.

Usage:
    python experiments/libero/summarize_by_category.py \
        --output_dir evaluate_results/libero_plus/eval_<RUN_ID> \
        [--classification LIBERO-plus/libero/libero/benchmark/task_classification.json]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

# category -> short column header (matches the requested output format)
CATEGORY_SHORT = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
# Fixed column order (Camera Robot Language Light Background Noise Layout Avg)
CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]


def load_classification(path: str) -> dict[tuple[str, int], str]:
    """Map (suite, task_id) -> category from task_classification.json."""
    data = json.load(open(path, encoding="utf-8"))
    mapping: dict[tuple[str, int], str] = {}
    for suite, tasks in data.items():
        for t in tasks:
            mapping[(suite, int(t["id"]))] = t["category"]
    return mapping


def collect_results(output_dir: str) -> list[dict]:
    """Read all result jsons. Each entry: suite, task_id, success (0/1)."""
    results = []
    for f in glob.glob(os.path.join(output_dir, "**", "gpu*_task*_results.json"), recursive=True):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        suite = r.get("task_suite")
        task_id = r.get("task_id")
        if suite is None or task_id is None:
            continue
        total = int(r.get("total_episodes", 1)) or 1
        succ = int(r.get("successes", 0))
        # num_trials=1 for plus, so success_rate = succ/total (0 or 100)
        results.append({
            "suite": suite,
            "task_id": int(task_id) + 1,  # result files are 0-indexed; classification is 1-indexed
            "success_rate": 100.0 * succ / total,
            "duration": r.get("duration"),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="eval output dir (contains gpu*_task*_results.json)")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_cls = os.path.join(repo_root, "third_party", "LIBERO-plus", "libero", "libero", "benchmark", "task_classification.json")
    ap.add_argument("--classification", default=default_cls, help="task_classification.json path")
    args = ap.parse_args()

    if not os.path.isfile(args.classification):
        raise FileNotFoundError(f"task_classification.json not found: {args.classification}")

    cls = load_classification(args.classification)
    results = collect_results(args.output_dir)

    # Per-category accumulators
    cat_succ = defaultdict(float)   # sum of success_rate
    cat_n = defaultdict(int)        # completed tasks
    cat_total = defaultdict(int)    # total tasks in this category (from classification)
    for cat in CATEGORY_ORDER:
        cat_total[cat] = 0
    for (suite, tid), cat in cls.items():
        cat_total[cat] += 1

    unmatched = 0
    for r in results:
        key = (r["suite"], r["task_id"])
        cat = cls.get(key)
        if cat is None:
            unmatched += 1
            continue
        cat_succ[cat] += r["success_rate"]
        cat_n[cat] += 1

    # Build the compact table
    headers = [CATEGORY_SHORT[c] for c in CATEGORY_ORDER] + ["Avg"]
    acc_row = []      # accuracy (%)
    coverage_row = [] # completed / total
    succ_count = 0
    done_count = 0
    for cat in CATEGORY_ORDER:
        n = cat_n[cat]
        acc = cat_succ[cat] / n if n > 0 else float("nan")
        acc_row.append(acc)
        coverage_row.append(f"{n}/{cat_total[cat]}")
        if n > 0:
            succ_count += cat_succ[cat] / 100.0
            done_count += n
    overall_acc = 100.0 * succ_count / done_count if done_count > 0 else float("nan")
    acc_row.append(overall_acc)
    coverage_row.append(f"{done_count}/{sum(cat_total.values())}")

    def fmt_acc(x):
        return f"{x:6.2f}" if x == x else "  N/A "  # NaN check

    # --- Print compact format ---
    print()
    print("=" * 72)
    print(f"  Results: {os.path.basename(args.output_dir.rstrip('/'))}")
    print(f"  Completed: {done_count} / {sum(cat_total.values())} tasks"
          f"  ({100.0 * done_count / sum(cat_total.values()):.1f}%)")
    if unmatched:
        print(f"  (warning: {unmatched} results could not be matched to a category)")
    print("=" * 72)
    print()
    # Header
    print("  " + "  ".join(f"{h:>9}" for h in headers))
    print("  " + "-" * (11 * len(headers) - 2))
    # Accuracy row
    print("  " + "  ".join(fmt_acc(x) for x in acc_row) + "   <- Accuracy (%)")
    # Coverage row
    print("  " + "  ".join(f"{c:>9}" for c in coverage_row) + "   <- Done/Total")
    print()

    # --- Also a verbose table (per category: acc, done, total, avg time) ---
    print("Per-category detail:")
    print(f"  {'Category':<22} {'Acc(%)':>8} {'Done':>10} {'AvgTime(s)':>10}")
    print("  " + "-" * 54)
    for cat in CATEGORY_ORDER:
        n = cat_n[cat]
        acc = cat_succ[cat] / n if n > 0 else float("nan")
        # avg time over completed tasks in this category
        durs = [r["duration"] for r in results
                if cls.get((r["suite"], r["task_id"])) == cat and r["duration"] is not None]
        avg_t = sum(durs) / len(durs) if durs else float("nan")
        acc_s = f"{acc:.2f}" if n > 0 else "N/A"
        t_s = f"{avg_t:.1f}" if durs else "N/A"
        print(f"  {CATEGORY_SHORT[cat]:<22} {acc_s:>8} {f'{n}/{cat_total[cat]}':>10} {t_s:>10}")
    print(f"  {'Avg':<22} {overall_acc:>8.2f} {f'{done_count}/{sum(cat_total.values())}':>10}")
    print()

    # --- Save CSV (对齐美观,逗号后补空格使各列对齐) ---
    csv_path = os.path.join(args.output_dir, "summary_by_category.csv")
    # 列宽:Category 12, Accuracy(%) 14, Done 8, Total 8, AvgTime(s) 10
    def _fmt_row(cat, acc_s, done_s, total_s, t_s):
        return (f"{cat:<11}, {acc_s:>11}, {done_s:>6}, {total_s:>6}, {t_s:>9}")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(_fmt_row("Category", "Accuracy(%)", "Done", "Total", "AvgTime(s)") + "\n")
        for cat in CATEGORY_ORDER:
            n = cat_n[cat]
            acc = cat_succ[cat] / n if n > 0 else None
            durs = [r["duration"] for r in results
                    if cls.get((r["suite"], r["task_id"])) == cat and r["duration"] is not None]
            avg_t = sum(durs) / len(durs) if durs else None
            acc_s = f"{acc:.4f}" if acc is not None else "-"
            t_s = f"{avg_t:.2f}" if avg_t is not None else "-"
            f.write(_fmt_row(CATEGORY_SHORT[cat], acc_s, str(n), str(cat_total[cat]), t_s) + "\n")
        f.write(_fmt_row("Avg", f"{overall_acc:.4f}", str(done_count),
                         str(sum(cat_total.values())), "-") + "\n")
    print(f"Saved: {csv_path}")

    # --- Save JSON ---
    json_path = os.path.join(args.output_dir, "summary_by_category.json")
    payload = {
        "run_id": os.path.basename(args.output_dir.rstrip("/")),
        "completed": done_count,
        "total": sum(cat_total.values()),
        "overall_accuracy": overall_acc,
        "categories": {},
    }
    for cat in CATEGORY_ORDER:
        n = cat_n[cat]
        durs = [r["duration"] for r in results
                if cls.get((r["suite"], r["task_id"])) == cat and r["duration"] is not None]
        payload["categories"][CATEGORY_SHORT[cat]] = {
            "accuracy": (cat_succ[cat] / n) if n > 0 else None,
            "completed": n,
            "total": cat_total[cat],
            "avg_time": (sum(durs) / len(durs)) if durs else None,
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

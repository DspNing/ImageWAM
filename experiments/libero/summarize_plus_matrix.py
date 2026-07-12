"""Summarize LIBERO-Plus results: 7 perturbation categories x 4 task suites.

LIBERO-Plus classifies each task into one of 7 perturbation categories
(Camera Viewpoints / Robot Initial States / Language Instructions / Light
Conditions / Background Textures / Sensor Noise / Objects Layout).

This script reads the classification from
``LIBERO-plus/libero/libero/benchmark/task_classification.json``, joins every
result file (``gpu*_task{task_id}_results.json``), and prints a compact matrix
with per-cell accuracy, per-category average, and per-suite average.

    Camera  Robot  Language  Light  Background  Noise  Layout  Avg(Suite)
  Goal
  Spatial
  Object
  10
    Avg(Cat)

Usage:
    python experiments/libero/summarize_plus_matrix.py \\
        --output_dir evaluate_results/libero_plus/eval_<RUN_ID>
"""

import argparse
import glob
import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORY_SHORT = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}

CATEGORY_KEYS = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]

# 4 base suites
BASE_SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]


def load_classification(path: str) -> dict[tuple[str, int], str]:
    """Map (suite, task_id) -> category from task_classification.json."""
    data = json.load(open(path, encoding="utf-8"))
    mapping: dict[tuple[str, int], str] = {}
    for suite, tasks in data.items():
        for t in tasks:
            mapping[(suite, int(t["id"]))] = t["category"]
    return mapping


def collect_results(output_dir: str) -> list[dict]:
    """Read all gpu*_task*_results.json."""
    results: list[dict] = []
    for f in glob.glob(
        os.path.join(output_dir, "**", "gpu*_task*_results.json"), recursive=True
    ):
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
        results.append(
            {
                "suite": suite,
                "task_id": int(task_id),
                "success_rate": 100.0 * succ / total,
                "duration": r.get("duration"),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="LIBERO-Plus: 7-category x 4-suite accuracy matrix."
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="eval output dir (contains gpu*_task*_results.json)",
    )
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    default_cls = os.path.join(
        repo_root,
        "third_party",
        "LIBERO-plus",
        "libero",
        "libero",
        "benchmark",
        "task_classification.json",
    )
    ap.add_argument(
        "--classification",
        default=default_cls,
        help="task_classification.json path",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.classification):
        raise FileNotFoundError(f"Not found: {args.classification}")

    cls = load_classification(args.classification)
    results = collect_results(args.output_dir)

    # ---- Accumulate per (suite, category) success rates ----
    # cell[(suite, cat)] = list of success_rate values
    cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        cat = cls.get((r["suite"], r["task_id"]))
        if cat is None:
            continue
        cell[(r["suite"], cat)].append(r["success_rate"])

    # ---- Compute accuracies ----
    # acc[suite][cat] = mean success_rate for that cell
    acc: dict[str, dict[str, float]] = {}
    for suite in BASE_SUITES:
        acc[suite] = {}
        for cat in CATEGORY_KEYS:
            rates = cell.get((suite, cat), [])
            acc[suite][cat] = sum(rates) / len(rates) if rates else float("nan")

    # Per-category average: weighted by number of completed tasks
    cat_avg: dict[str, float] = {}
    cat_succ_weighted: dict[str, float] = {}  # sum of successes (as fraction)
    cat_done_weighted: dict[str, int] = {}
    for cat in CATEGORY_KEYS:
        total_succ = 0.0  # sum of success_rate / 100 (= number of successes)
        total_done = 0
        for s in BASE_SUITES:
            rates = cell.get((s, cat), [])
            if rates:
                total_succ += sum(rates) / 100.0
                total_done += len(rates)
        cat_succ_weighted[cat] = total_succ
        cat_done_weighted[cat] = total_done
        cat_avg[cat] = 100.0 * total_succ / total_done if total_done > 0 else float("nan")

    # Overall average: weighted by total completed tasks across all suites × categories
    grand_succ = sum(cat_succ_weighted.values())
    grand_done = sum(cat_done_weighted.values())
    overall = 100.0 * grand_succ / grand_done if grand_done > 0 else float("nan")

    # Per-suite average: weighted by number of completed tasks within that suite
    suite_avg: dict[str, float] = {}
    for s in BASE_SUITES:
        s_succ = 0.0
        s_done = 0
        for cat in CATEGORY_KEYS:
            rates = cell.get((s, cat), [])
            if rates:
                s_succ += sum(rates) / 100.0
                s_done += len(rates)
        suite_avg[s] = 100.0 * s_succ / s_done if s_done > 0 else float("nan")

    # ---- Count completed tasks ----
    done: dict[str, dict[str, int]] = {}
    total: dict[str, dict[str, int]] = {}
    for suite in BASE_SUITES:
        done[suite] = {}
        total[suite] = {}
        for cat in CATEGORY_KEYS:
            total[(suite, cat)] = cls.get(
                key for key, c in cls.items() if key[0] == suite and c == cat
            )
    # Recompute properly
    total: dict[tuple[str, str], int] = {}
    done: dict[tuple[str, str], int] = {}
    for (suite, tid), cat in cls.items():
        if suite not in BASE_SUITES:
            continue
        key = (suite, cat)
        total[key] = total.get(key, 0) + 1
    for r in results:
        cat = cls.get((r["suite"], r["task_id"]))
        if cat is None:
            continue
        key = (r["suite"], cat)
        done[key] = done.get(key, 0) + 1

    # ---- Print matrix ----
    short_headers = [CATEGORY_SHORT[c] for c in CATEGORY_KEYS]

    print()
    print("=" * 72)
    print(f"  Results: {os.path.basename(args.output_dir.rstrip('/'))}")
    total_done = sum(done.values())
    total_all = sum(total.values())
    print(f"  Completed: {total_done} / {total_all} tasks  ({100.0 * total_done / total_all if total_all else 0:.1f}%)")
    print("=" * 72)
    print()

    # Header row
    print(
        f"  {'':<10}"
        + "".join(f"{h:>9}" for h in short_headers)
        + f"  {'Avg':>9}"
    )
    print("  " + "-" * (12 + 11 * len(short_headers) + 10))

    # Suite rows
    for suite in BASE_SUITES:
        row = f"  {suite.replace('libero_', ''):<10}"
        row += "".join(
            f"{acc[suite][cat]:>9.2f}" if acc[suite][cat] == acc[suite][cat] else f"{'N/A':>9}"
            for cat in CATEGORY_KEYS
        )
        sa = suite_avg[suite]
        row += f"  {sa:>9.2f}" if sa == sa else f"  {'N/A':>9}"
        print(row)

    # Separator
    print("  " + "-" * (12 + 11 * len(short_headers) + 10))

    # Category avg row
    row = f"  {'Avg':<10}"
    row += "".join(
        f"{cat_avg[cat]:>9.2f}" if cat_avg[cat] == cat_avg[cat] else f"{'N/A':>9}"
        for cat in CATEGORY_KEYS
    )
    row += f"  {overall:>9.2f}" if overall == overall else f"  {'N/A':>9}"
    print(row)
    print()

    # ---- Per-category detail ----
    print("Per-category detail (accuracy, done/total, avg time):")
    print(
        f"  {'Category':<22} {'Acc(%)':>8} {'Done':>10} {'AvgTime(s)':>10}"
    )
    print("  " + "-" * 54)
    for cat in CATEGORY_KEYS:
        n = sum(done.get((s, cat), 0) for s in BASE_SUITES)
        t = sum(total.get((s, cat), 0) for s in BASE_SUITES)
        ca = cat_avg[cat]
        durs = [
            r["duration"]
            for r in results
            if cls.get((r["suite"], r["task_id"])) == cat
            and r["duration"] is not None
        ]
        avg_t = sum(durs) / len(durs) if durs else float("nan")
        acc_s = f"{ca:.2f}" if ca == ca else "N/A"
        t_s = f"{avg_t:.1f}" if durs else "N/A"
        print(f"  {CATEGORY_SHORT[cat]:<22} {acc_s:>8} {f'{n}/{t}':>10} {t_s:>10}")
    print()

    # ---- Save CSV ----
    csv_path = os.path.join(args.output_dir, "summary_plus_matrix.csv")
    w = 11
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(f"{'Suite':<{w}}," + ",".join(f"{CATEGORY_SHORT[c]:>{w}}" for c in CATEGORY_KEYS) + f",{'Avg':>{w}}\n")
        for suite in BASE_SUITES:
            parts = [suite.replace("libero_", "")]
            for cat in CATEGORY_KEYS:
                v = acc[suite][cat]
                parts.append(f"{v:.4f}" if v == v else "-")
            sa = suite_avg[suite]
            parts.append(f"{sa:.4f}" if sa == sa else "-")
            f.write(",".join(parts) + "\n")
        parts = ["Avg"]
        for cat in CATEGORY_KEYS:
            v = cat_avg[cat]
            parts.append(f"{v:.4f}" if v == v else "-")
        parts.append(f"{overall:.4f}" if overall == overall else "-")
        f.write(",".join(parts) + "\n")
    print(f"Saved: {csv_path}")

    # ---- Save JSON ----
    json_path = os.path.join(args.output_dir, "summary_plus_matrix.json")
    payload = {
        "run_id": os.path.basename(args.output_dir.rstrip("/")),
        "completed": total_done,
        "total": total_all,
        "overall_accuracy": overall,
        "suite_avg": {s: suite_avg[s] for s in BASE_SUITES},
        "category_avg": {CATEGORY_SHORT[c]: cat_avg[c] for c in CATEGORY_KEYS},
        "matrix": {},
    }
    for suite in BASE_SUITES:
        payload["matrix"][suite] = {}
        for cat in CATEGORY_KEYS:
            rates = cell.get((suite, cat), [])
            n = done.get((suite, cat), 0)
            t = total.get((suite, cat), 0)
            payload["matrix"][suite][CATEGORY_SHORT[cat]] = {
                "accuracy": (sum(rates) / len(rates)) if rates else None,
                "completed": n,
                "total": t,
            }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda o: None if o != o else None)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

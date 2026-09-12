#!/usr/bin/env bash
# watch_progress.sh — 实时监控 LIBERO / LIBERO-Plus / LIBERO-PRO 推理进度
#
# 用法:
#   bash watch_progress.sh <RUN_ID>              # 默认 libero_plus
#   bash watch_progress.sh <RUN_ID> libero       # master 推理(原版 LIBERO)
#   bash watch_progress.sh <RUN_ID> libero_pro   # LIBERO-PRO OOD 推理
#   bash watch_progress.sh /abs/path/to/result_dir [libero|libero_plus|libero_pro]
#                                                # 绝对路径:直接监控该目录,模式仍由第二参指定
#   REFRESH=5 NUM_TRIALS=30 bash watch_progress.sh <RUN_ID> libero_pro   # 自定义 trials 数
#
# 可调变量:
#   NUM_TRIALS   — 每个任务的 trial 次数(默认: libero=50, libero_plus=1, libero_pro=50)
#   REFRESH      — 刷新间隔(秒, 默认 5)
#
# Ctrl+C 退出。

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: bash watch_progress.sh <RUN_ID 或 绝对路径> [libero|libero_plus|libero_pro]"
  exit 1
fi

RUN_ID="$1"
SUBDIR="${2:-libero_plus}"
if [[ "$RUN_ID" = /* ]]; then
  # 绝对路径:直接作为结果目录
  OUT="$RUN_ID"
else
  OUT="evaluate_results/${SUBDIR}/${RUN_ID}"
fi
PYBIN="${PYBIN:-/home/NingZijian/miniconda3/envs/fastwam/bin/python}"

[[ -d "$OUT" ]] || { echo "Error: 目录不存在: $OUT" >&2; exit 1; }

# 各模式的 suite 数量(硬编码，可按需修改)
if [[ "$SUBDIR" == "libero_plus" ]]; then NUM_SUITES=10030;
elif [[ "$SUBDIR" == "libero_pro" ]]; then NUM_SUITES=200;
else NUM_SUITES=40; fi

# 每个任务的 trial 次数(可手动修改, 也可通过环境变量覆盖)
#   libero_plus: 1,  libero/libero_pro: 50
if [[ "$SUBDIR" == "libero_plus" ]]; then NUM_TRIALS=1;
elif [[ "$SUBDIR" == "libero_pro" ]]; then NUM_TRIALS=50;
else NUM_TRIALS=50; fi

TOTAL=$((NUM_SUITES * NUM_TRIALS))
REFRESH="${REFRESH:-5}"

echo "监控: $OUT  (总数 $TOTAL, 每 ${REFRESH}s 刷新, Ctrl+C 退出)"
echo
export NUM_TRIALS
$PYBIN - "$OUT" "$TOTAL" "$REFRESH" "$SUBDIR" <<'PYEOF'
import sys, os, glob, json, re, time

OUT, TOTAL, REFRESH, SUBDIR = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
width = 30
t0 = time.time()
start_n = None

# ========== LIBERO-Plus: 加载分类(category) ==========
clsd = {}
clf = "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
if os.path.exists(clf):
    for s, tasks in json.load(open(clf)).items():
        for t in tasks:
            clsd[(s, int(t["id"]))] = t

def snapshot_plus():
    """LIBERO-Plus 模式:10030 tasks, 每个 1 trial。
    完成数/成功数从结果文件统计(和调度脚本一致,ground truth),
    不从 worker 日志数(日志会被重启覆盖,导致漏数)。"""
    done = 0
    succ = 0
    log_dir = os.path.join(OUT, "task_logs")
    workers = []
    if not os.path.isdir(log_dir):
        return 0, 0, []

    # 数结果文件(和调度脚本 run_libero_plus_batch.sh 的 total_completed 一致)
    result_files = glob.glob(os.path.join(OUT, "**", "gpu*_task*_results.json"), recursive=True)
    done = len(result_files)
    for rf in result_files:
        try:
            with open(rf, 'r') as f:
                r = json.load(f)
            if int(r.get("successes", 0)) > 0:
                succ += 1
        except:
            continue

    # 读所有 worker 日志(文件名: worker0_gpu5.log) —— 仅用于显示当前在跑的任务
    all_logs = glob.glob(os.path.join(log_dir, "worker*.log"))

    # 活跃 worker: 过滤掉已完成(worker_done == 日志总行数)的 worker
    all_logs = glob.glob(os.path.join(log_dir, "worker*.log"))
    active_workers = []
    # 收集所有 DONE 的实际时间戳 (epoch seconds)，用于算最近 5 分钟速度
    done_timestamps = []
    for log_path in all_logs:
        try:
            mtime = os.path.getmtime(log_path)
            if time.time() - mtime > 300:
                continue
            with open(log_path, 'r') as f:
                content = f.read()
        except:
            continue

        fname = os.path.basename(log_path)
        m = re.match(r'worker(\d+)_gpu(\d+)\.log', fname)
        wid = m.group(1) if m else "?"
        gpu = m.group(2) if m else "?"

        # 当前任务: [worker 0] [3/836] suite=libero_spatial task_id=25
        cur = re.findall(r'\[worker \d+\] \[(\d+)/(\d+)\] suite=(\w+) task_id=(\d+)', content)
        # episode 进度
        episodes = re.findall(r'Episode\s+(\d+):\s+(\d+)%.*?(\d+)/(\d+)', content)

        # 该 worker 已完成数 & 日志总行数(总任务数)
        worker_done = len(re.findall(r'\[worker \d+\] \[\d+/\d+\].*DONE', content))
        worker_succ = len(re.findall(r'successes=1/', content))

        # 收集 DONE 的实际时间戳: "[2026-07-09 16:17:01,861] ... DONE"
        for dl in re.findall(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}\].*DONE', content):
            try:
                done_timestamps.append(time.mktime(time.strptime(dl[:19], "%Y-%m-%d %H:%M:%S")))
            except:
                pass

        if cur:
            prog, tot_tasks, suite, tid = cur[-1]
            suite_short = suite.replace("libero_", "")
            t = clsd.get((suite, int(tid)), {})
            cat = t.get("category", "?")
            short = {"Camera Viewpoints":"Camera","Robot Initial States":"Robot",
                     "Language Instructions":"Language","Light Conditions":"Light",
                     "Background Textures":"Background","Sensor Noise":"Noise",
                     "Objects Layout":"Layout"}.get(cat, cat)
            if episodes:
                ep, pct, step_cur, step_tot = episodes[-1]
                step_info = f"step {step_cur}/{step_tot} ({pct}%)"
            else:
                step_info = "?"
            trial_info = f"[{prog}/{tot_tasks}] {suite_short:<8} {short:<12} task={tid} {step_info} | done={worker_done}(succ={worker_succ})"
        elif "Loading" in content[-500:] or "loading" in content[-500:]:
            trial_info = "加载模型中"
        elif "Error" in content[-500:] or "Traceback" in content[-500:]:
            trial_info = "⚠ 报错"
        else:
            trial_info = "启动中"

        active_workers.append({"wid": wid, "gpu": gpu, "trial": trial_info})

    return done, succ, active_workers, done_timestamps


def snapshot_master():
    """LIBERO master/pro 模式: NUM_SUITES tasks × NUM_TRIALS trials。"""
    NT = int(os.environ.get("NUM_TRIALS", "50"))
    done = 0
    succ = 0
    log_dir = os.path.join(OUT, "task_logs")
    if not os.path.isdir(log_dir):
        return 0, 0, []

    # 完成的 trial 和成功数：从结果文件统计(ground truth)
    result_files = glob.glob(os.path.join(OUT, "**", "gpu*_task*_results.json"), recursive=True)
    done = 0
    succ = 0
    for rf in result_files:
        try:
            with open(rf, 'r') as f:
                r = json.load(f)
            done += int(r.get("total_episodes", 0))
            succ += int(r.get("successes", 0))
        except:
            continue

    # 正在跑的任务：从 task_gpu_map.txt 读取真实运行中的任务
    workers = []
    task_map = os.path.join(OUT, "task_gpu_map.txt")
    if os.path.isfile(task_map):
        for line in open(task_map):
            line = line.strip()
            if not line:
                continue
            # 格式: suite,task_id:gpu_id
            parts = line.rsplit(":", 1)
            if len(parts) != 2:
                continue
            task_key, gpu = parts
            suite, task_id = task_key.split(",", 1)
            # 检查是否已经有结果（排除已完成的）
            result_pattern = os.path.join(OUT, suite, f"gpu{gpu}_task{task_id}_results.json")
            if os.path.isfile(result_pattern):
                continue
            # 检查日志是否活跃
            log_file = os.path.join(log_dir, f"{suite}_task{task_id}_gpu{gpu}.log")
            trial_info = "未知"
            if os.path.isfile(log_file):
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                    episodes = re.findall(r'Episode\s+(\d+):\s+(\d+)%.*?(\d+)/(\d+)', content)
                    if episodes:
                        ep, pct, cur, tot = episodes[-1]
                        trial_info = f"trial {ep}/{NT} | {cur}/{tot} steps ({pct}%)"
                    elif "Loading" in content[-500:] or "loading" in content[-500:]:
                        trial_info = "加载模型中"
                    elif "Error" in content[-500:] or "Traceback" in content[-500:]:
                        trial_info = "⚠ 报错"
                    else:
                        trial_info = "启动中"

                    s = len(re.findall(r'--success=True', content))
                    d = len(re.findall(r'--success=(?:True|False)', content))
                    done += d
                    succ += s
                    if d > 0:
                        trial_info += f" | 已完成{d}/{NT} (成功{s})"
                except:
                    pass
            workers.append({"gpu": gpu, "suite": suite, "task": task_id, "trial": trial_info})

    return done, succ, workers

while True:
    if SUBDIR == "libero_plus":
        n, succ, workers, done_ts = snapshot_plus()
        total_trials = TOTAL
        label = "tasks"
    else:
        n, succ, workers = snapshot_master()
        done_ts = []
        total_trials = TOTAL
        label = "trials"

    now = time.time()
    if start_n is None: start_n = n
    dt = now - t0

    # 进度条
    filled = min(int(n * width / total_trials), width)
    bar = "#" * filled + (">" + "-" * (width - filled - 1) if filled < width else "")
    pct = n * 100 / total_trials

    # 速度计算: 最近 5 分钟内完成的任务数
    WINDOW = 300  # 5 分钟
    recent_done = sum(1 for ts in done_ts if now - ts <= WINDOW)
    if recent_done > 0:
        rate = recent_done * 3600.0 / WINDOW  # tasks/hour
        eta_h = (total_trials - n) / rate if rate > 0 else 0
        eta_str = f"{eta_h:.1f}h" if eta_h >= 1 else f"{eta_h*60:.0f}m"
        rate_s = f"{rate:.0f}/h"
    elif n > 0 and dt > 0:
        # 没有最近 5 分钟的 DONE，退化为全局平均
        rate = n * 3600 / dt
        eta_h = (total_trials - n) / rate if rate > 0 else 0
        eta_str = f"{eta_h:.1f}h" if eta_h >= 1 else f"{eta_h*60:.0f}m"
        rate_s = f"{rate:.0f}/h"
    else:
        rate_s = "—"; eta_str = "计算中"

    os.system("clear")
    succ_rate = f"{succ}/{n} ({succ*100/n:.1f}%)" if n > 0 else "—"
    print(f"[{bar}] {pct:5.1f}% | {n}/{total_trials} {label} | {rate_s} | 剩余 {eta_str}")
    print(f"成功: {succ_rate}")
    print("=" * 72)
    if workers:
        print(f"正在跑的任务 ({len(workers)}):")
        print("-" * 72)
        if SUBDIR == "libero_plus":
            for w in workers:
                print(f"  worker{w['wid']:<2} GPU{w['gpu']:<2} {w['trial']}")
        else:
            for w in workers:
                print(f"  GPU{w['gpu']:<2} {w['suite']:<14} task={w['task']:<3} {w['trial']}")
    else:
        print("暂无正在跑的任务(可能都在加载模型或已完成)")
    print("-" * 72)
    print(f"(每 {REFRESH}s 刷新, Ctrl+C 退出)")
    sys.stdout.flush()
    time.sleep(REFRESH)
PYEOF

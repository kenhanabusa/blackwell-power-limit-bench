#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class StepSummary:
    step_idx: int
    limit_w: int
    avg_gpu_util_percent: float
    avg_gpu_temp_c: float
    avg_gpu_power_w: float
    avg_server_power_w: float
    step_start_time: float
    step_end_time: float
    avg_tokens_per_s: Optional[float] = None


def parse_args():
    p = argparse.ArgumentParser(description="Sweep GPU power limit while logging GPU / fan / server metrics.")
    p.add_argument(
        "--train-cmd",
        type=str,
        required=True,
        help='Training command to launch (e.g. "torchrun ... train_ddp_transformer.py ...").',
    )
    p.add_argument("--step-seconds", type=int, default=600, help="Duration of each power level in seconds.")
    p.add_argument("--step-watts", type=int, default=100, help="Step size (W) when decreasing power limit.")
    p.add_argument("--log-dir", type=str, default="power_sweep_logs", help="Prefix for run directory name.")
    p.add_argument("--warmup-seconds", type=int, default=60, help="Warmup time before starting sweep.")
    p.add_argument("--sample-interval", type=float, default=1.0, help="Sampling interval (s) for metrics.")
    p.add_argument("--use-sudo", action="store_true", help="Use sudo for nvidia-smi / ipmitool calls.")
    return p.parse_args()


def run_cmd(cmd: List[str], use_sudo: bool = False) -> str:
    if use_sudo and cmd[0] != "sudo":
        cmd = ["sudo"] + cmd
    out = subprocess.check_output(cmd, encoding="utf-8", errors="ignore")
    return out


def get_power_limits(use_sudo: bool = False) -> Tuple[int, int]:
    out = run_cmd(
        ["nvidia-smi", "--query-gpu=power.min_limit,power.max_limit", "--format=csv,noheader,nounits"],
        use_sudo=use_sudo,
    )
    mins: List[int] = []
    maxs: List[int] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            mn = int(float(parts[0]))
            mx = int(float(parts[1]))
        except ValueError:
            continue
        mins.append(mn)
        maxs.append(mx)
    if not mins or not maxs:
        raise RuntimeError("Failed to parse power limits from nvidia-smi output:\n" + out)
    min_lim = max(mins)
    max_lim = min(maxs)
    return min_lim, max_lim


def set_power_limit(limit_w: int, use_sudo: bool = False) -> None:
    cmd = ["nvidia-smi", "-pl", str(limit_w)]
    if use_sudo:
        cmd.insert(0, "sudo")
    subprocess.run(cmd, check=True)


def sample_gpu_metrics(use_sudo: bool = False) -> List[Dict[str, float]]:
    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        use_sudo=use_sudo,
    )
    res: List[Dict[str, float]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            idx = int(parts[0])
            util = float(parts[1])
            temp = float(parts[2])
            mem_used = float(parts[3])
            mem_total = float(parts[4])
            pwr = float(parts[5])
        except ValueError:
            continue
        res.append(
            {
                "index": idx,
                "util": util,
                "temp": temp,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "power": pwr,
            }
        )
    return res


def sample_fan_metrics(use_sudo: bool = False) -> List[Dict[str, float]]:
    out = run_cmd(["ipmitool", "sdr", "type", "fan"], use_sudo=use_sudo)
    res: List[Dict[str, float]] = []
    for line in out.strip().splitlines():
        # e.g. 'Front Fan RPM    | 41h | ok  | 29.1 | 8400 RPM'
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        name = parts[0]
        reading = parts[4]
        if not reading.lower().endswith("rpm"):
            continue
        try:
            rpm_val = float(reading.split()[0])
        except ValueError:
            continue
        res.append({"name": name, "rpm": rpm_val})
    return res


def sample_server_power(use_sudo: bool = False) -> Optional[Dict[str, float]]:
    try:
        out = run_cmd(["ipmitool", "dcmi", "power", "reading"], use_sudo=use_sudo)
    except subprocess.CalledProcessError:
        return None
    inst = avg = None
    for line in out.strip().splitlines():
        line = line.strip()
        if line.startswith("Instantaneous power reading"):
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    inst = float(parts[1].split()[0])
                except ValueError:
                    pass
        elif line.startswith("Average power reading over sample period"):
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    avg = float(parts[1].split()[0])
                except ValueError:
                    pass
    if inst is None and avg is None:
        return None
    return {"instant_w": inst, "avg_w": avg}


def parse_training_tokens_per_iter(training_log_path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    world_size: Optional[int] = None
    gbs: Optional[int] = None
    seq_len: Optional[int] = None
    try:
        with open(training_log_path, "r") as f:
            for line in f:
                if "world_size" in line and ":" in line:
                    try:
                        world_size = int(line.split(":")[1])
                    except ValueError:
                        pass
                elif "global_batch_size" in line and ":" in line:
                    try:
                        gbs = int(line.split(":")[1])
                    except ValueError:
                        pass
                elif line.strip().startswith("seq_len") and ":" in line:
                    try:
                        seq_len = int(line.split(":")[1])
                    except ValueError:
                        pass
                if world_size is not None and gbs is not None and seq_len is not None:
                    break
    except FileNotFoundError:
        return None, None, None
    return world_size, gbs, seq_len


def compute_tokens_per_step(
    blocks_csv: str, tokens_per_iter: float, step_times: List[Tuple[float, float]]
) -> Dict[int, float]:
    # blocks_csv columns: timestamp, iter
    times: List[float] = []
    iters: List[int] = []
    try:
        with open(blocks_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float(row["timestamp"])
                    it = int(row["iter"])
                except (KeyError, ValueError):
                    continue
                times.append(t)
                iters.append(it)
    except FileNotFoundError:
        return {}
    if len(times) < 2:
        return {}
    step_tot_tokens = [0.0 for _ in step_times]
    step_tot_time = [0.0 for _ in step_times]
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        it0, it1 = iters[i - 1], iters[i]
        dt = t1 - t0
        if dt <= 0:
            continue
        dit = it1 - it0
        if dit <= 0:
            continue
        tokens = dit * tokens_per_iter
        center = 0.5 * (t0 + t1)
        # find step whose [start,end) contains center
        for step_idx, (s, e) in enumerate(step_times):
            if s <= center < e:
                step_tot_tokens[step_idx] += tokens
                step_tot_time[step_idx] += dt
                break
    result: Dict[int, float] = {}
    for idx, (tok, tt) in enumerate(zip(step_tot_tokens, step_tot_time)):
        if tt > 0:
            result[idx] = tok / tt
    return result


def launch_training(train_cmd: str, run_dir: str) -> Tuple[subprocess.Popen, threading.Thread]:
    cmd_list = shlex.split(train_cmd)
    log_path = os.path.join(run_dir, "training.log")
    blocks_csv = os.path.join(run_dir, "training_blocks.csv")
    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def reader():
        os.makedirs(run_dir, exist_ok=True)
        with open(log_path, "w") as log_f, open(blocks_csv, "w", newline="") as blk_f:
            blk_writer = csv.writer(blk_f)
            blk_writer.writerow(["timestamp", "iter"])
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                # Parse measure iter lines
                if "[Measure]" in line and "iter" in line:
                    m = re.search(r"iter\s+(\d+)/", line)
                    if m:
                        try:
                            it = int(m.group(1))
                        except ValueError:
                            continue
                        ts = time.time()
                        blk_writer.writerow([f"{ts:.6f}", it])
                        blk_f.flush()
        # when process ends, reader returns

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return proc, t


def plot_vs_limit(run_dir: str, summaries: List[StepSummary]) -> None:
    limits = [s.limit_w for s in summaries]
    util = [s.avg_gpu_util_percent for s in summaries]
    temp = [s.avg_gpu_temp_c for s in summaries]
    gpup = [s.avg_gpu_power_w for s in summaries]
    srvp = [s.avg_server_power_w for s in summaries]
    toks = [s.avg_tokens_per_s for s in summaries]

    def plot_simple(y, ylabel: str, fname: str):
        plt.figure()
        plt.plot(limits, y, marker="o")
        plt.xlabel("Power limit (W)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()  # show higher power on left if limits are [600,500,...]
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname))
        plt.close()

    plot_simple(util, "Average GPU utilization (%)", "summary_avg_util_vs_limit.png")
    plot_simple(temp, "Average GPU temperature (°C)", "summary_avg_temp_vs_limit.png")
    plot_simple(gpup, "Average GPU power draw (W)", "summary_avg_power_vs_limit.png")
    plot_simple(srvp, "Average server power draw (W)", "summary_avg_server_power_vs_limit.png")

    if any(t is not None and not math.isnan(t) for t in toks):
        y = [float("nan") if t is None or math.isnan(t) else t / 1e6 for t in toks]
        plot_simple(y, "Throughput (M tokens/s)", "summary_tokens_vs_limit.png")


def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.log_dir}_{timestamp}"
    os.makedirs(run_dir, exist_ok=False)
    print(f"[INFO] Run directory: {run_dir}")
    sys.stdout.flush()

    # Query power limits and construct sweep levels
    min_lim, max_lim = get_power_limits(use_sudo=args.use_sudo)
    print(f"[INFO] Reported power limits: min={min_lim} W, max={max_lim} W")
    levels: List[int] = list(range(max_lim, min_lim - 1, -args.step_watts))
    if levels[-1] != min_lim:
        levels.append(min_lim)
    print(f"[INFO] Power sweep levels: {levels}")
    sys.stdout.flush()

    # Ensure starting from max power
    print(f"[INFO] Setting initial power limit to max {max_lim} W")
    set_power_limit(max_lim, use_sudo=args.use_sudo)

    # Launch training
    print("[INFO] Launching training command:")
    print(f"       {args.train_cmd}")
    sys.stdout.flush()
    proc, reader_thread = launch_training(args.train_cmd, run_dir)

    # Warmup
    print(f"[INFO] Waiting warmup {args.warmup_seconds} s ...")
    sys.stdout.flush()
    warmup_deadline = time.time() + args.warmup_seconds
    while time.time() < warmup_deadline:
        if proc.poll() is not None:
            print("[ERROR] Training process exited during warmup.", file=sys.stderr)
            reader_thread.join(timeout=5)
            return
        time.sleep(1.0)

    # Per-step CSV writers
    step_summaries: List[StepSummary] = []
    step_times: List[Tuple[float, float]] = []

    for step_idx, limit in enumerate(levels):
        if proc.poll() is not None:
            print("[WARN] Training process already exited before step", step_idx)
            break
        print(f"[INFO] === Step {step_idx}: set power limit {limit} W ===")
        sys.stdout.flush()
        set_power_limit(limit, use_sudo=args.use_sudo)
        step_ts = time.strftime("%Y%m%d_%H%M%S")
        gpu_csv_path = os.path.join(run_dir, f"step{step_idx:02d}_{limit}W_{step_ts}_gpu.csv")
        fan_csv_path = os.path.join(run_dir, f"step{step_idx:02d}_{limit}W_{step_ts}_fan.csv")
        server_csv_path = os.path.join(run_dir, f"step{step_idx:02d}_{limit}W_{step_ts}_server.csv")

        step_start = time.time()
        step_end = step_start + args.step_seconds
        step_times.append((step_start, step_end))

        gpu_rows = 0
        util_sum = 0.0
        temp_sum = 0.0
        power_sum = 0.0

        server_rows = 0
        server_power_sum = 0.0

        last_sample_time = step_start

        with open(gpu_csv_path, "w", newline="") as gpu_f, \
             open(fan_csv_path, "w", newline="") as fan_f, \
             open(server_csv_path, "w", newline="") as server_f:
            gpu_writer = csv.writer(gpu_f)
            fan_writer = csv.writer(fan_f)
            server_writer = csv.writer(server_f)
            gpu_writer.writerow(
                ["timestamp", "gpu_index", "util_percent", "temp_c", "mem_used_mb", "mem_total_mb", "power_w"]
            )
            fan_writer.writerow(["timestamp", "name", "rpm"])
            server_writer.writerow(["timestamp", "instant_w", "avg_w"])

            while time.time() < step_end:
                now = time.time()
                last_sample_time = now
                if proc.poll() is not None:
                    print("[WARN] Training process exited during step", step_idx)
                    break
                # GPU metrics
                g_list = sample_gpu_metrics(use_sudo=args.use_sudo)
                for g in g_list:
                    gpu_writer.writerow(
                        [
                            f"{now:.3f}",
                            g["index"],
                            f"{g['util']:.1f}",
                            f"{g['temp']:.1f}",
                            f"{g['mem_used']:.1f}",
                            f"{g['mem_total']:.1f}",
                            f"{g['power']:.1f}",
                        ]
                    )
                gpu_f.flush()
                if g_list:
                    for g in g_list:
                        util_sum += g["util"]
                        temp_sum += g["temp"]
                        power_sum += g["power"]
                        gpu_rows += 1

                # Fan metrics
                fans = sample_fan_metrics(use_sudo=args.use_sudo)
                for fan in fans:
                    fan_writer.writerow([f"{now:.3f}", fan["name"], f"{fan['rpm']:.1f}"])
                fan_f.flush()

                # Server power
                srv = sample_server_power(use_sudo=args.use_sudo)
                if srv is not None:
                    server_writer.writerow(
                        [
                            f"{now:.3f}",
                            "" if srv["instant_w"] is None else f"{srv['instant_w']:.1f}",
                            "" if srv["avg_w"] is None else f"{srv['avg_w']:.1f}",
                        ]
                    )
                    server_f.flush()
                    if srv["avg_w"] is not None:
                        server_power_sum += srv["avg_w"]
                        server_rows += 1
                    elif srv["instant_w"] is not None:
                        server_power_sum += srv["instant_w"]
                        server_rows += 1

                time.sleep(args.sample_interval)

        step_real_end = last_sample_time
        if gpu_rows > 0:
            avg_util = util_sum / gpu_rows
            avg_temp = temp_sum / gpu_rows
            avg_power = power_sum / gpu_rows
        else:
            avg_util = float("nan")
            avg_temp = float("nan")
            avg_power = float("nan")
        if server_rows > 0:
            avg_server_power = server_power_sum / server_rows
        else:
            avg_server_power = float("nan")

        summary = StepSummary(
            step_idx=step_idx,
            limit_w=limit,
            avg_gpu_util_percent=avg_util,
            avg_gpu_temp_c=avg_temp,
            avg_gpu_power_w=avg_power,
            avg_server_power_w=avg_server_power,
            step_start_time=step_start,
            step_end_time=step_real_end,
        )
        step_summaries.append(summary)
        print(
            f"[INFO] Step {step_idx} finished: "
            f"avg GPU util={avg_util:.1f}%, temp={avg_temp:.1f}C, power={avg_power:.1f}W, "
            f"server power={avg_server_power:.1f}W"
        )
        sys.stdout.flush()
        if proc.poll() is not None:
            break

    # Stop training process if still running
    if proc.poll() is None:
        print("[INFO] Stopping training process ...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[WARN] Training process did not terminate, killing ...")
            proc.kill()
    reader_thread.join(timeout=10)

    # Compute tokens/s per step
    training_log = os.path.join(run_dir, "training.log")
    blocks_csv = os.path.join(run_dir, "training_blocks.csv")
    world_size, gbs, seq_len = parse_training_tokens_per_iter(training_log)
    if gbs is not None and seq_len is not None:
        tokens_per_iter = gbs * seq_len
        print(
            f"[INFO] Detected world_size={world_size}, global_batch_size={gbs}, "
            f"seq_len={seq_len} -> tokens_per_iter={tokens_per_iter}"
        )
        tokens_per_s_per_step = compute_tokens_per_step(blocks_csv, float(tokens_per_iter), step_times)
        for s in step_summaries:
            if s.step_idx in tokens_per_s_per_step:
                s.avg_tokens_per_s = tokens_per_s_per_step[s.step_idx]
    else:
        print(
            "[WARN] Could not parse global_batch_size and seq_len from training log. "
            "tokens/s per step will not be available."
        )

    # Write summary.csv
    summary_csv = os.path.join(run_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step_idx",
                "limit_w",
                "avg_gpu_util_percent",
                "avg_gpu_temp_c",
                "avg_gpu_power_w",
                "avg_server_power_w",
                "avg_tokens_per_s",
                "step_start_time",
                "step_end_time",
            ]
        )
        for s in step_summaries:
            writer.writerow(
                [
                    s.step_idx,
                    s.limit_w,
                    f"{s.avg_gpu_util_percent:.3f}" if not math.isnan(s.avg_gpu_util_percent) else "",
                    f"{s.avg_gpu_temp_c:.3f}" if not math.isnan(s.avg_gpu_temp_c) else "",
                    f"{s.avg_gpu_power_w:.3f}" if not math.isnan(s.avg_gpu_power_w) else "",
                    f"{s.avg_server_power_w:.3f}" if not math.isnan(s.avg_server_power_w) else "",
                    "" if s.avg_tokens_per_s is None else f"{s.avg_tokens_per_s:.3f}",
                    f"{s.step_start_time:.3f}",
                    f"{s.step_end_time:.3f}",
                ]
            )
    print(f"[INFO] Wrote summary csv: {summary_csv}")
    sys.stdout.flush()

    # Plots
    if step_summaries:
        plot_vs_limit(run_dir, step_summaries)
        print("[INFO] Wrote summary plots:")
        print("       - summary_avg_util_vs_limit.png")
        print("       - summary_avg_temp_vs_limit.png")
        print("       - summary_avg_power_vs_limit.png")
        print("       - summary_avg_server_power_vs_limit.png")
        print("       - summary_tokens_vs_limit.png")
    else:
        print("[WARN] No step summaries collected; skipping plots.")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

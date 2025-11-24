#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import signal
import subprocess
import sys
import time


def run_cmd(cmd):
    """サブプロセスを実行して文字列として返す。失敗したら None。"""
    try:
        out = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL)
        return out
    except Exception:
        return None


def collect_gpu_metrics():
    """
    nvidia-smi から GPU 情報を取得して、辞書のリストを返す。
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = run_cmd(cmd)
    if out is None:
        return []

    metrics = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            idx = int(parts[0])
            util = float(parts[1])        # %
            temp = float(parts[2])        # C
            power = None if parts[3] == "N/A" else float(parts[3])  # W
            mem_used = float(parts[4])    # MiB
            mem_total = float(parts[5])   # MiB
        except ValueError:
            continue

        metrics.append(
            {
                "gpu_index": idx,
                "util": util,
                "temp": temp,
                "power": power,
                "mem_used": mem_used,
                "mem_total": mem_total,
            }
        )
    return metrics


def collect_fan_metrics():
    """
    ipmitool から FAN センサーを取得して、辞書のリストを返す。
    例:
      Front Fan RPM    | 41h | ok  | 29.1 | 8400 RPM
      Rear Fan RPM     | 42h | ok  | 29.1 | 5000 RPM
    """
    out = run_cmd(["ipmitool", "sdr", "type", "fan"])
    if out is None:
        return []

    metrics = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        name = parts[0]

        # 2列目以降から「RPM を含む列」を探す（最後の列にあることが多い）
        reading_field = None
        for p in parts[1:]:
            if "RPM" in p.upper():
                reading_field = p
                break

        if reading_field is None:
            # この行には RPM 情報がなさそう
            continue

        # "8400 RPM" → 8400
        rpm_str = reading_field.split()[0]
        try:
            rpm = int(rpm_str)
        except ValueError:
            continue

        metrics.append({"fan_name": name, "rpm": rpm})

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="GPU / IPMI Fan monitor -> CSV ログ"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="サンプリング間隔（秒, デフォルト: 1.0）",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="logs",
        help="ログ出力ディレクトリ（デフォルト: ./logs）",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ログファイル名にタイムスタンプを付ける
    ts_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_csv_path = os.path.join(args.outdir, f"gpu_metrics_{ts_str}.csv")
    fan_csv_path = os.path.join(args.outdir, f"fan_metrics_{ts_str}.csv")

    print(f"[INFO] GPU log -> {gpu_csv_path}")
    print(f"[INFO] FAN log -> {fan_csv_path}")
    print(f"[INFO] interval = {args.interval} [s]")
    print("[INFO] Ctrl+C で終了します。")

    # CSV を開く
    gpu_file = open(gpu_csv_path, "w", newline="")
    fan_file = open(fan_csv_path, "w", newline="")

    gpu_writer = csv.writer(gpu_file)
    fan_writer = csv.writer(fan_file)

    # ヘッダ
    gpu_writer.writerow(
        ["timestamp", "gpu_index", "util_percent", "temp_C", "power_W", "mem_used_MiB", "mem_total_MiB"]
    )
    fan_writer.writerow(["timestamp", "fan_name", "rpm"])

    gpu_file.flush()
    fan_file.flush()

    stop = False

    def handle_sigint(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop:
            now = dt.datetime.now().isoformat(timespec="seconds")

            gpu_metrics = collect_gpu_metrics()
            fan_metrics = collect_fan_metrics()

            # GPU
            for g in gpu_metrics:
                gpu_writer.writerow(
                    [
                        now,
                        g["gpu_index"],
                        g["util"],
                        g["temp"],
                        g["power"] if g["power"] is not None else "",
                        g["mem_used"],
                        g["mem_total"],
                    ]
                )

            # FAN
            for f in fan_metrics:
                fan_writer.writerow(
                    [
                        now,
                        f["fan_name"],
                        f["rpm"],
                    ]
                )

            gpu_file.flush()
            fan_file.flush()

            # 簡単なコンソール表示（お好みで削除可）
            if gpu_metrics:
                summary = ", ".join(
                    f"GPU{g['gpu_index']}: {g['util']}% {g['temp']}C {g['power']}W {g['mem_used']}/{g['mem_total']}MiB"
                    for g in gpu_metrics
                )
                print(f"{now} | {summary}", end="")
                if fan_metrics:
                    fan_summary = ", ".join(
                        f"{f['fan_name']}={f['rpm']}RPM" for f in fan_metrics
                    )
                    print(" | " + fan_summary)
                else:
                    print()
            else:
                print(f"{now} | (nvidia-smi 取得失敗?)")

            time.sleep(args.interval)

    finally:
        gpu_file.close()
        fan_file.close()
        print("\n[INFO] 終了しました。")


if __name__ == "__main__":
    main()

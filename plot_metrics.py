#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="GPU / FAN CSV ログの簡易プロット（全GPUを同じグラフに表示 & 保存）"
    )
    p.add_argument("--gpu-csv", required=True, help="gpu_metrics_*.csv のパス")
    p.add_argument("--fan-csv", required=False, help="fan_metrics_*.csv のパス（任意）")
    p.add_argument(
        "--outdir",
        type=str,
        default="figs",
        help="画像の保存先ディレクトリ（デフォルト: ./figs）",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="ファイル名のプレフィックス（省略時は gpu-csv のファイル名を流用）",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="ウィンドウを表示せず保存だけ行う",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 出力ディレクトリ作成
    os.makedirs(args.outdir, exist_ok=True)

    # プレフィックス決定
    if args.prefix is not None:
        prefix = args.prefix
    else:
        prefix = Path(args.gpu_csv).stem  # 例: gpu_metrics_20251124_064036

    # GPU CSV 読み込み
    gpu_df = pd.read_csv(args.gpu_csv)
    gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"])
    gpu_df.sort_values("timestamp", inplace=True)

    # Fan CSV（あれば）
    fan_df = None
    if args.fan_csv:
        fan_df = pd.read_csv(args.fan_csv)
        fan_df["timestamp"] = pd.to_datetime(fan_df["timestamp"])
        fan_df.sort_values("timestamp", inplace=True)

    # ---- GPU utilization ----
    fig, ax = plt.subplots()
    for gid, sub in gpu_df.groupby("gpu_index"):
        ax.plot(sub["timestamp"], sub["util_percent"], label=f"GPU{gid}")
    ax.set_xlabel("time")
    ax.set_ylabel("GPU util [%]")
    ax.set_title("GPU utilization")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    util_path = os.path.join(args.outdir, f"{prefix}_gpu_util.png")
    fig.savefig(util_path, dpi=150)
    print(f"[SAVE] {util_path}")

    # ---- GPU temperature ----
    fig, ax = plt.subplots()
    for gid, sub in gpu_df.groupby("gpu_index"):
        ax.plot(sub["timestamp"], sub["temp_C"], label=f"GPU{gid}")
    ax.set_xlabel("time")
    ax.set_ylabel("temperature [C]")
    ax.set_title("GPU temperature")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    temp_path = os.path.join(args.outdir, f"{prefix}_gpu_temp.png")
    fig.savefig(temp_path, dpi=150)
    print(f"[SAVE] {temp_path}")

    # ---- GPU power ----
    fig, ax = plt.subplots()
    for gid, sub in gpu_df.groupby("gpu_index"):
        ax.plot(sub["timestamp"], sub["power_W"], label=f"GPU{gid}")
    ax.set_xlabel("time")
    ax.set_ylabel("power [W]")
    ax.set_title("GPU power draw")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    power_path = os.path.join(args.outdir, f"{prefix}_gpu_power.png")
    fig.savefig(power_path, dpi=150)
    print(f"[SAVE] {power_path}")

    # ---- GPU memory used ----
    fig, ax = plt.subplots()
    for gid, sub in gpu_df.groupby("gpu_index"):
        ax.plot(sub["timestamp"], sub["mem_used_MiB"], label=f"GPU{gid}")
    ax.set_xlabel("time")
    ax.set_ylabel("memory used [MiB]")
    ax.set_title("GPU memory used")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    mem_path = os.path.join(args.outdir, f"{prefix}_gpu_mem.png")
    fig.savefig(mem_path, dpi=150)
    print(f"[SAVE] {mem_path}")

    # ---- FAN ----
    if fan_df is not None and not fan_df.empty:
        fig, ax = plt.subplots()
        for name, sub in fan_df.groupby("fan_name"):
            ax.plot(sub["timestamp"], sub["rpm"], label=name)
        ax.set_xlabel("time")
        ax.set_ylabel("RPM")
        ax.set_title("Fan speeds")
        plt.xticks(rotation=30, ha="right")
        ax.legend()
        plt.tight_layout()
        fan_path = os.path.join(args.outdir, f"{prefix}_fans.png")
        fig.savefig(fan_path, dpi=150)
        print(f"[SAVE] {fan_path}")

    if not args.no_show:
        # 画面にも表示
        plt.show()
    else:
        # 表示せずに終了
        plt.close("all")


if __name__ == "__main__":
    main()

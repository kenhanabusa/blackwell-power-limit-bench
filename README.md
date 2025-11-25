# Blackwell Power Limit Bench

RTX PRO 6000 Blackwell サーバーエディション 4GPU マシンで、
パワーリミット（600 / 500 / 400 / 300 W）を変えながら
Transformer 学習ベンチマークを実行しつつ、GPU / ファン / サーバー電力を記録するためのスクリプト一式です。

ベンチマークでは各 GPU がほぼフルにメモリを使用する設定になっており、
平均で **約 93.0 GiB / 95.6 GiB（約 97%）** を継続的に使用します。
つまり **RTX PRO 6000 Blackwell Server Edition 96GB の容量をほぼフルに使う** 形でベンチマークしています。

関連ブログ記事本体はこちら：

- https://server-gear.com/blog/post/rtx-pro-6000-blackwell-server-4gpu-power-limit-benchmark

---

## ハードウェア構成（検証時）

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 4
- サーバー筐体: Supermicro SYS-322GA-NR
- CPU: Intel Xeon 6944P × 2（合計 144 コア）
- メモリ: DDR5-6400 32GB DIMM × 16（合計 512GB）
- ストレージ: NVMe SSD（KIOXIA KCD8XPUG7T68 7.68TB / KCD8XPUG1T92 1.92TB）
- OS: Ubuntu 24.04.3 LTS 系
- 電源: 冗長 PSU 構成
- NVIDIA Driver: 580 系（例: 580.105.08, CUDA 13.0）

※シリアル番号などの固有情報は README からは省いています。

---

## ソフトウェア構成（例）

- PyTorch + CUDA (torchrun, DistributedDataParallel)
- ipmitool（サーバー電力・ファン情報取得に利用）
- nvidia-smi（GPU 情報取得に利用）
- Python 3.11 系
- 必要 Python パッケージ: pandas, matplotlib など

※PyTorch / CUDA の具体的なインストール方法は、利用する環境に依存するためここでは省略しています。  
　公式ドキュメントの手順に沿って GPU 対応の PyTorch をインストールしてください。

---

## ファイル構成

- run_power_sweep.sh  
  conda 環境を有効化し、GPU メモリ関連の環境変数を設定したうえで  
  `run_all_power_sweep.py` を起動するラッパースクリプトです。

- run_all_power_sweep.py  
  学習ジョブを 1 回だけ起動し、その裏で

  - `nvidia-smi` による GPU 利用率 / 温度 / メモリ使用量 / GPU 電力の取得  
  - `ipmitool dcmi power reading` によるサーバー全体の電力取得  
  - `ipmitool sdr type fan` によるファン RPM 取得  

  を行いつつ、パワーリミットを **600 → 500 → 400 → 300 W** とスイープします。  
  最終的に `summary.csv` といくつかの PNG グラフ（util / temp / power / tokens など）を生成します。

- monitor_gpus_and_fans.py  
  単体で GPU とファンの状態をロギングしたい場合に使えるモニタ用スクリプトです。

- train_ddp_transformer.py  
  4GPU DDP で動作する Transformer 学習ベンチマークです。  
  おおよそ 2.6 億パラメータの GPT 系モデルで、実際の LLM 学習に近い  
  メモリアクセスと通信パターンになるように作ってあります。

- plot_metrics.py  
  取得した CSV ログから、電力・温度・ファン回転数などのグラフを生成するためのスクリプトです。

---

## 実験条件（デフォルト設定）

`run_power_sweep.sh` では以下の条件で `torchrun` を起動します。

- シーケンス長: `seq_len = 1280`
- グローバルバッチサイズ: `global_batch_size = 256`（4GPU なので GPU あたり 64）
- 精度: FP16（AMP + GradScaler）
- 分散: 4GPU データ並列（DDP）
- パワーリミット: 600 / 500 / 400 / 300 W（`--step-watts=100`）
- 各ステップの計測時間: 600 秒（10 分）

この設定では、各 GPU のメモリ使用量はほぼいっぱいになり、
平均で **約 93.0 GiB / 95.6 GiB（約 97%）** 程度を継続的に使用します。  
つまり **RTX PRO 6000 Blackwell Server Edition 96GB の容量をほぼフルに使う** 形でベンチマークしています。

---

## 使い方（Quick Start）

### 1. リポジトリの取得

以下を実行してリポジトリを取得します。

    git clone https://github.com/kenhanabusa/blackwell-power-limit-bench.git
    cd blackwell-power-limit-bench

### 2. Python 環境の準備（例: conda）

すでに PyTorch 環境がある場合は、このステップは読み替えていただいて構いません。

- conda / mamba が入っている前提:

    conda create -n gpu-bench python=3.11 -y
    conda activate gpu-bench

- 依存パッケージのインストール例（あくまで一例です）:

    # PyTorch 本体（CUDA バージョンに応じて公式サイトのコマンドに置き換えてください）
    # 例: CUDA 12.8 (cu128) の場合（2025-11 時点）
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    pip install -U pip
    pip install pandas matplotlib

    # サーバー電力・ファン情報取得に必要なツール
    sudo apt-get update
    sudo apt-get install -y ipmitool

- NVIDIA ドライバ / CUDA Toolkit は、事前にセットアップしておいてください。

### 3. パワーリミットスイープの実行

`run_power_sweep.sh` を実行すると、学習ジョブを起動したうえで  
600 / 500 / 400 / 300 W にパワーリミットを切り替えながら、各種ログを収集します。

    cd blackwell-power-limit-bench

    # 実行権限がなければ付与
    chmod +x run_power_sweep.sh

    # ベンチマーク開始（sudo パスワードを聞かれる場合があります）
    ./run_power_sweep.sh

実行が終わると、`power_sweep_logs_YYYYMMDD_HHMMSS` のようなディレクトリが作成され、
その中に以下が生成されます。

- `summary.csv` : パワーリミットごとの平均値（GPU 利用率 / 温度 / メモリ / 電力 / サーバー電力 / tokens/s など）
- `stepXX_..._gpu.csv` : 生の GPU メトリクスログ
- `stepXX_..._fan.csv` : ファン RPM ログ
- `stepXX_..._server.csv` : サーバー電力ログ
- `summary_*.png` : パワーリミットと各メトリクスの簡易プロット

### 4. ログの確認と追加プロット

一番手軽なのは `summary.csv` を手元の環境で開くことです（Excel や pandas など）。

Python からざっくり中身を確認する例:

    # 例: 直近のログディレクトリが power_sweep_logs_20251124_133840 の場合
    cd blackwell-power-limit-bench

    python - << 'PY'
    import pandas as pd

    run_dir = "power_sweep_logs_20251124_133840"  # 実際のディレクトリ名に置き換えてください
    df = pd.read_csv(f"{run_dir}/summary.csv")
    print(df.to_string(index=False))
    PY

より詳細なグラフを描きたい場合は `plot_metrics.py` を参考にして、
環境に合わせて適宜カスタマイズしてください。

---

## ライセンス / 注意事項

- 本リポジトリのスクリプトは、RTX PRO 6000 Blackwell サーバーエディション 4GPU 構成を前提に検証しています。
  別構成で利用する場合は、パワーリミットの範囲や ipmitool の挙動などを十分確認したうえでご利用ください。
- ライセンス表記や細かいチューニング情報は今後追記予定です。


### 検証時に使用したバージョン（参考）

このリポジトリの結果は、以下の環境で動作確認しています。

- Python 3.11.14
- PyTorch 2.9.1+cu128
- torchvision 0.24.1+cu128
- torchaudio 2.9.1+cu128
- NVIDIA Driver 580.105.08（`nvidia-smi` 上の CUDA Version 表示は 13.0）
- CUDA 12.8 系ライブラリ一式（nvidia-cuda-runtime-cu12 12.8.90 / nvidia-cudnn-cu12 9.10.2.21 など）

Blackwell 世代の GPU を使う場合は、
**少なくとも「PyTorch 2.9 系 + CUDA 12.8 以降」程度の比較的新しいスタック**を使うことを推奨します。

PyTorch 自体のインストールコマンドは OS やパッケージマネージャによって変わるため、  
最新版については PyTorch 公式サイトの「Get Started」から、
お使いの環境（Linux / pip or conda / CUDA 12.8 など）を選んで表示されるコマンドを利用してください。

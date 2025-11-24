# Blackwell Power Limit Bench

RTX PRO 6000 Blackwell サーバーエディション 4GPU マシンで、
パワーリミット（600 / 500 / 400 / 300 W）を変えながら
学習ベンチマークを実行するためのスクリプト一式です。

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 4
- サーバー筐体: Supermicro SYS-322GA-NR
- CPU: Intel Xeon 6944P × 2（計 144 コア）
- メモリ: DDR5-6400 32GB DIMM × 16（計 512GB）
- OS: Ubuntu 24.04 LTS
- フレームワーク: PyTorch + torchrun + DDP

詳細な使い方や再現手順は今後追記予定です。

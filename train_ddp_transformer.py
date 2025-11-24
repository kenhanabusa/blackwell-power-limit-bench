#!/usr/bin/env python
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================
# ランダムトークン列データセット
# ============================
class RandomTextDataset(Dataset):
    """
    実際の運用では、このクラスを自前の Dataset に差し替えればOKです。
    今は vocab_size の中からランダムでトークン列を生成するだけです。
    """
    def __init__(self, vocab_size: int, seq_len: int, dataset_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            dtype=torch.long,
        )
        return tokens


# ============================
# Transformer 言語モデル（約0.76Bパラメータ）
# ============================
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 1024,    # さらに少し縮小（≈0.76B params）
        n_layers: int = 12,
        n_heads: int = 16,      # 1536 / 16 = 96 dim/head
        dim_ff: int = 4096,     # 4 * d_model
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,  # (batch, seq, hidden)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len) のトークンID
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        # 位置ID [0, 1, ..., seq_len-1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        x = self.encoder(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits


# ============================
# DDP 初期化 / 後始末
# ============================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


# ============================
# メイン
# ============================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Transformer LM benchmark (DDP, ~0.76B params)"
    )

    # モデルやデータのサイズ
    parser.add_argument("--vocab-size", type=int, default=50000,
                        help="語彙サイズ")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="シーケンス長")
    parser.add_argument("--global-batch-size", type=int, default=256,
                        help="全GPU合計のバッチサイズ（デフォルト:256）")
    parser.add_argument("--dataset-size", type=int, default=10000,
                        help="エポックあたりのサンプル数（合計）")

    # 学習設定
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学習率")
    parser.add_argument("--warmup-iters", type=int, default=20,
                        help="測定前のウォームアップイテレーション数")
    parser.add_argument("--measure-iters", type=int, default=100,
                        help="実測に使うイテレーション数")
    parser.add_argument("--fp16", action="store_true",
                        help="AMP(FP16)を有効化")

    args = parser.parse_args()

    # ---- DDP セットアップ ----
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if args.global_batch_size < world_size:
        if rank == 0:
            print(
                f"[ERROR] global_batch_size={args.global_batch_size} が "
                f"world_size={world_size} より小さいため、"
                "local_batch_size が 0 になってしまいます。"
            )
        cleanup_distributed()
        return

    if args.global_batch_size % world_size != 0 and rank == 0:
        print(
            f"[Warning] global_batch_size={args.global_batch_size} は "
            f"world_size={world_size} で割り切れません。"
        )

    local_batch_size = max(1, args.global_batch_size // world_size)

    # ---- モデル ----
    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=max(args.seq_len, 4096),
        # d_model / n_layers / n_heads / dim_ff はデフォルト値を使用（1536, 24, 16, 6144）
    )

    # パラメータ数を表示（確認用）
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("===== Model config =====")
        print(f"Parameters      : {total_params} "
              f"({total_params/1e9:.3f} B params)")
        print("========================\n")

    model.to(device)

    # DDP ラッパー
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # AMP (FP16) スケーラ（新API）
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    if rank == 0:
        print("===== Training config =====")
        print(f"world_size        : {world_size}")
        print(f"rank              : {rank}")
        print(f"local_rank        : {local_rank}")
        print(f"global_batch_size : {args.global_batch_size}")
        print(f"local_batch_size  : {local_batch_size}")
        print(f"seq_len           : {args.seq_len}")
        print(f"dataset_size      : {args.dataset_size}")
        print(f"fp16              : {args.fp16}")
        print("===========================\n")

    # ---- データセット / データローダ ----
    dataset = RandomTextDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dataset_size=args.dataset_size,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 1イテレーションの処理 ----
    def run_iteration(batch_tokens: torch.Tensor):
        """
        1 ミニバッチ分の学習ステップを実行し、
        損失と「1イテレーションで処理したトークン数（このGPU分）」を返す。
        """
        batch_tokens = batch_tokens.to(device, non_blocking=True)  # (B, L)

        # 次トークン予測用に1トークンずらす
        inputs = batch_tokens[:, :-1]   # (B, L-1)
        targets = batch_tokens[:, 1:]   # (B, L-1)

        # 新しい torch.amp.autocast API
        with torch.amp.autocast(
            device_type="cuda",
            enabled=args.fp16,
            dtype=torch.float16,
        ):
            logits = model(inputs)      # (B, L-1, vocab)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tokens_in_batch = inputs.numel()  # B * (L-1)
        return loss.item(), tokens_in_batch

    # ---- イテレータヘルパ ----
    model.train()
    epoch = 0
    data_iter = iter(dataloader)

    def next_batch():
        nonlocal data_iter, epoch
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch

    # ---- ウォームアップ ----
    if rank == 0:
        print(f"Warmup {args.warmup_iters} iters ...")

    for i in range(args.warmup_iters):
        batch = next_batch()
        loss, _ = run_iteration(batch)
        if rank == 0 and (i + 1) % 10 == 0:
            print(f"[Warmup] iter {i+1}/{args.warmup_iters} loss={loss:.4f}")

    # ---- 実測区間 ----
    if rank == 0:
        print(f"\nMeasure {args.measure_iters} iters ...")

    torch.cuda.synchronize()
    start = time.time()
    total_tokens = 0.0  # 全GPU合計トークン数

    for i in range(args.measure_iters):
        batch = next_batch()
        loss, tokens_per_gpu = run_iteration(batch)

        # 全GPUのトークン数を集約
        t = torch.tensor([tokens_per_gpu], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        tokens_global = t.item()
        total_tokens += tokens_global

        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - start
            tok_per_sec = total_tokens / max(elapsed, 1e-6)
            print(
                f"[Measure] iter {i+1}/{args.measure_iters} "
                f"loss={loss:.4f} "
                f"throughput={tok_per_sec/1e6:.2f}M tokens/s"
            )

    torch.cuda.synchronize()
    elapsed = time.time() - start
    final_tok_per_sec = total_tokens / max(elapsed, 1e-6)

    if rank == 0:
        print("\n===== Result =====")
        print(f"Total tokens   : {total_tokens:.3e}")
        print(f"Elapsed (s)    : {elapsed:.2f}")
        print(
            f"Throughput     : {final_tok_per_sec:.2f} tokens/s "
            f"({final_tok_per_sec/1e6:.2f}M tokens/s)"
        )
        print("==================")

    cleanup_distributed()


if __name__ == "__main__":
    main()

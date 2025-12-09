import os, time, torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    # 32MiB 程度のテンソル（4096x4096 fp16）
    n = 4096
    a = torch.randn((n, n), device=device, dtype=torch.float16)
    b = torch.randn((n, n), device=device, dtype=torch.float16)

    # warmup
    for _ in range(5):
        c = a @ b
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iters = 20
    t0 = time.time()
    for _ in range(iters):
        c = a @ b
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    t1 = time.time()

    sec = (t1 - t0) / iters
    gb = (c.numel() * c.element_size()) / (1024**3)  # per-GPU payload
    traffic_gb = 2 * (world_size - 1) / world_size * gb  # rough allreduce traffic per GPU

    if local_rank == 0:
        print(f"world_size={world_size}")
        print(f"avg iter time: {sec:.3f} s")
        print(f"payload per iter per GPU: {gb:.3f} GiB (tensor)")
        print(f"estimated allreduce traffic per iter per GPU: {traffic_gb:.3f} GiB")
        print(f"est. allreduce throughput per GPU: {traffic_gb/sec:.3f} GiB/s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

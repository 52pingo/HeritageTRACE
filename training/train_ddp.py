# SPDX-License-Identifier: Apache-2.0
# Modified for HeritageTRACE: parameterised local training and state-disjoint validation.
# -*- coding: utf-8 -*-
"""Laya RLCD 微调训练脚本(单机多卡 DDP,适配 A40 双卡)。

基于官方 notebook(train_ddp.py)改写:
- 路径全部参数化,不依赖 Kaggle
- 单卡/多卡自动适配(单卡时跳过 DDP)
- A40 48GB 显存充裕,梯度检查点默认关闭;显存不够时 --grad-ckpt 开启

用法(在装有 CUDA 版 torch 的机器上):
  torchrun --standalone --nproc_per_node=2 train_ddp.py \
      --model-dir /path/to/laya --items train_items.pt --out /path/to/output
"""
import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos,
        "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def save_checkpoint(model, tok, cfg, out_dir) -> None:
    """保存检查点,对主机内存压力做防护。

    把 322M 参数转 fp16 搬到 CPU 会瞬时占用约 644MB 主机内存;在内存本就紧张的
    机器上(实测 8GB 显卡本机空闲仅 4-6GB),这一步会以
    `DefaultCPUAllocator: not enough memory` 或 safetensors 的 Rust
    `memory allocation of N bytes failed` 崩溃 —— 训练本身跑得好好的,却死在存盘上。

    对策:存前回收 Python 与 CUDA 缓存;失败时做一次更彻底的回收后重试;
    存完立刻释放那份 fp16 副本,避免它和下一轮训练叠加。
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "model": os.path.join(out_dir, "model.safetensors"),
        "cfg": os.path.join(out_dir, "rl_agent_config.json"),
    }
    last_err = None
    for attempt in range(2):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sd = None
        try:
            sd = {kk: vv.half().contiguous().cpu() for kk, vv in model.state_dict().items()}
            save_file(sd, paths["model"])
            model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
            tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
            with open(paths["cfg"], "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            return
        except (MemoryError, RuntimeError) as exc:
            last_err = exc
            if attempt == 0:
                print(f"  [warn] 保存检查点失败({str(exc)[:90]}),回收内存后重试...")
        finally:
            del sd
            gc.collect()


def fit_one_temp(sel):
    """在留出集上按问题类型拟合一个校准温度(LBFGS 最小化 NLL)。"""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--items", required=True, help="prepare_data.py 输出的 train_items.pt")
    ap.add_argument("--out", required=True, help="输出检查点目录")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=16, help="A40 48GB 可用 32;开 --grad-ckpt 后可更大")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true", help="显存不足时开启梯度检查点")
    ap.add_argument("--max-len", type=int, default=1024,
                    help="state 的最大 token 数。默认 1024 与基座一致;"
                         "长记录会在 state 尾部被截断,可调到 1536 覆盖长文")
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    args = ap.parse_args()

    use_dist = "RANK" in os.environ
    if use_dist:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda")

    with open(os.path.join(args.model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["max_tokens_per_batch"] = 4096
    # max_len 会随 checkpoint 一起保存,故推理时的截断长度自动与训练一致
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(args.model_dir, "encoder"))
    weights = load_file(os.path.join(args.model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    if args.grad_ckpt:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()

    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) \
        if use_dist else model

    all_items = torch.load(args.items, weights_only=False)

    # 校准留出集:固定种子,所有 rank 留出相同样本
    CALIB_MAX = 400
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    my_items = train_items[rank::world_size]

    EPOCHS, MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = args.epochs, args.micro_batch, args.grad_accum, 4
    SIGMA_START, SIGMA_END = 0.4, 0.1

    enc_params = [p for n, p in train_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in train_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder},
         {"params": head_params, "lr": args.lr_head}], weight_decay=0.01)

    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        print(f"RLCD training: {len(train_items)} train items ({len(calib_items)} held out) "
              f"| {len(my_items)} per rank | {world_size} GPU(s) | {EPOCHS} epochs")
    t0 = time.time()

    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, EPOCHS - 1)

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = train_model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device))

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. GRPO 基线采样:零均值投影的带噪 logits
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. 适当评分规则奖励
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. 策略梯度 + 软交叉熵
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if rank == 0 and (n_batches % 50) == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | "
                      f"Loss: {loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f}")

        if rank == 0:
            print(f"=== Epoch {epoch+1}/{EPOCHS} done in {time.time()-t0:.1f}s | "
                  f"Avg Loss: {epoch_loss/max(1, n_batches):.4f} ===")
        if use_dist:
            dist.barrier()
        # 滚动检查点:崩溃/断点不丢已有进度
        if rank == 0:
            # rl_agent_config.json 必须一并保存:laya.load() 依赖它读取
            # max_len / head_max_len / 温度等配置。缺失时该检查点无法加载,
            # 也就无法按 README 的说法用它续训。
            save_checkpoint(model, tok, cfg, os.path.join(args.out, "checkpoint_latest"))
            print(f"  rolling checkpoint saved (epoch {epoch+1})")

    if use_dist:
        dist.barrier()

    # 训练后:在留出集上拟合校准温度并导出最终检查点
    if rank == 0:
        print("\nFitting calibration temperatures on held-out slice...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx:c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device), cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                        cb["qtype"].to(device))
                l_np = l_sub.float().cpu().numpy()
                for r_i, it in enumerate(c_chunk):
                    kk = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r_i, :kk], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("Fitted temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
        except Exception as e:
            print("Temperature fitting fallback:", e)

        cfg["fine_tuned"] = True
        cfg["temperature"] = fitted_temps
        cfg.pop("temperature_by_options", None)  # 旧的桶级温度会掩盖新拟合值
        save_checkpoint(model, tok, cfg, args.out)
        print(f"Model saved to {args.out}")

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

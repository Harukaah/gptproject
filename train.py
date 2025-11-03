# train.py
import os, time, argparse, random, math
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import GPT, GPTConfig
from utils import create_dataloaders


def parse_args():
    ap = argparse.ArgumentParser()
    # Data / tokenizer
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--tokenizer_name", type=str, default="airesearch/wangchanberta-base-att-spm-uncased")
    # Model / train
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer steps (global)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    # Runtime / IO
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Logging / saving / eval
    ap.add_argument("--save_every", type=int, default=10_000)
    ap.add_argument("--step_log_every", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=5_000, help="validate every N steps (0 = only end of epoch)")
    ap.add_argument("--throttle_ms", type=float, default=5, help="sleep per step to keep UI responsive (ms)")
    # Resume
    ap.add_argument("--init_from", type=str, default="", help="path to checkpoint to resume from")
    ap.add_argument("--save_full_state", action="store_true", help="save optimizer/scheduler/global_step too")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f">>> TRAINING DEVICE: {device}")
    if device == "cuda":
        print(">>> GPU:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
        # New-style TF32 controls (fallback to old if unavailable)
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        from torch import amp
        scaler = amp.GradScaler("cuda")
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.float32)

    # ---------------- Data ----------------
    train_dl, val_dl, vocab_size, tokenizer = create_dataloaders(
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        tokenizer_name=args.tokenizer_name,
    )
    print(f"len(train_dl) ≈ {len(train_dl):,} steps/epoch | batch={args.batch_size} block={args.block_size}")

    # ---------------- Model ----------------
    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout
    )
    model = GPT(cfg).to(device)

    # ---------------- Optim / Sched ----------------
    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    # T_max as an upper bound; cosine still works if we break early
    total_steps_planned = max(1, args.epochs * len(train_dl))
    sched = CosineAnnealingLR(opt, T_max=total_steps_planned)

    # ---------------- Resume ----------------
    global_step = 0
    if args.init_from:
        print(f"[resume] loading from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            if args.save_full_state:
                if "optimizer" in ckpt: opt.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt: sched.load_state_dict(ckpt["scheduler"])
            if "global_step" in ckpt:
                global_step = int(ckpt["global_step"])
                print(f"[resume] global_step -> {global_step}")
            else:
                print("[resume] no global_step found; continuing from 0 for step counting.")
        else:
            # weights-only
            model.load_state_dict(ckpt, strict=False)
            print("[resume] loaded weights-only checkpoint")

    # ---------------- Helpers ----------------
    throttle_sec = max(0.0, args.throttle_ms / 1000.0)
    best_val = math.inf

    def run_eval():
        model.eval()
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits, loss = model(xb, yb)
                loss_sum += loss.item(); n += 1
        model.train()
        val_loss = loss_sum / max(1, n)
        ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
        print(f"[VAL] loss={val_loss:.4f} | ppl={ppl:.2f}")
        return val_loss

    def save_checkpoint(tag):
        path = os.path.join(args.out_dir, f"gpt_{tag}.pt")
        if args.save_full_state:
            torch.save({
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
                "global_step": global_step,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
            }, path)
        else:
            torch.save(model.state_dict(), path)
        print("Saved:", path)

    # ---------------- Train Loop ----------------
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t_last = time.time()
            for step, (xb, yb) in enumerate(train_dl, start=1):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)

                with autocast_ctx:
                    logits, loss = model(xb, yb)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()

                # LR warmup then cosine
                if global_step < (args.warmup_steps or 0):
                    lr_scale = float(global_step + 1) / float(max(1, args.warmup_steps))
                    for pg in opt.param_groups: pg["lr"] = lr_scale * args.lr
                else:
                    sched.step()

                global_step += 1

                # gentle throttle for desktop responsiveness
                if throttle_sec > 0:
                    time.sleep(throttle_sec)

                # Logging
                if args.step_log_every and (global_step % args.step_log_every == 0):
                    now = time.time()
                    s_per_it = (now - t_last) / float(args.step_log_every)
                    t_last = now
                    print(f"ep {epoch} | step {step}/{len(train_dl)} | gstep {global_step} | loss {loss.item():.4f} | {s_per_it:.3f}s/it")

                # Periodic save
                if args.save_every and (global_step % args.save_every == 0):
                    save_checkpoint(f"step{global_step}")

                # Mid-epoch validation
                if args.eval_every and (global_step % args.eval_every == 0):
                    val_loss = run_eval()
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint("best")

                # Stop early on max_steps
                if args.max_steps is not None and global_step >= args.max_steps:
                    print(f"[STOP] Reached max_steps={args.max_steps}, saving and exiting…")
                    save_checkpoint(f"step{global_step}")
                    # Final val before exit (optional)
                    val_loss = run_eval()
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint("best")
                    return

            # End-of-epoch validation
            val_loss = run_eval()
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint("best")
            print(f"[EPOCH {epoch}] val_loss={val_loss:.4f} (best={best_val:.4f})")

    except KeyboardInterrupt:
        print("\n[STOPPED] KeyboardInterrupt — saving interrupt checkpoint…")
        save_checkpoint("interrupt")


if __name__ == "__main__":
    main()







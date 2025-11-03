# generate.py
import argparse, torch
from model import GPT, GPTConfig
from utils import load_tokenizer, DEF_THAI_TOKENIZER

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_name", type=str, default=DEF_THAI_TOKENIZER)
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--prompt", type=str, default="")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = load_tokenizer(args.tokenizer_name)

    # Encode prompt
    if args.prompt:
        ids = tok.encode(args.prompt, add_special_tokens=False)
    else:
        ids = [tok.bos_token_id] if tok.bos_token_id is not None else [tok.pad_token_id or 0]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens, block_size=cfg.block_size,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
        )

    text = tok.decode(out[0].tolist(), skip_special_tokens=True)
    print("\n===== GENERATED TEXT =====\n")
    print(text)
    print("\n==========================\n")

if __name__ == "__main__":
    main()



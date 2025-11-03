import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Default tokenizer (override via --tokenizer_name in train.py)
DEF_THAI_TOKENIZER = "airesearch/wangchanberta-base-att-spm-uncased"


# --- Tokenizer helpers --------------------------------------------------------
def load_tokenizer(name: str | None = None):
    """
    Load a HF tokenizer and ensure it has a pad token so batching/decoding is safe.
    """
    name = name or DEF_THAI_TOKENIZER
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)

    # Ensure a pad token exists (many sentencepiece models don't define one)
    if getattr(tok, "pad_token", None) is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    return tok


# Backward-compatible name (so `from utils import get_tokenizer` still works)
get_tokenizer = load_tokenizer
# ----------------------------------------------------------------------------


class TextDataset(Dataset):
    def __init__(self, dir_path: str, tokenizer, block_size: int = 512):
        self.tokenizer = tokenizer
        self.block_size = int(block_size)

        # --- Collect .txt files ---
        txt_files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        if not txt_files:
            raise ValueError(f"[utils] No .txt files found in: {dir_path}")

        texts = []
        for fname in txt_files:
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"[utils] WARN: failed to read {fpath}: {e}")

        print(f"[utils] Loaded {len(texts)} text files from {dir_path}")
        if not texts:
            raise ValueError("[utils] No readable text content was loaded.")

        # --- Concatenate and tokenize ---
        data = "\n\n".join(texts)
        enc_ids = self.tokenizer.encode(data, add_special_tokens=False)

        # --- Safety clamp: ensure ids are in range of embedding table ---
        vocab_full = len(self.tokenizer)  # includes special/added tokens
        unk_id = getattr(self.tokenizer, "unk_token_id", 0)
        enc_ids = [tid if 0 <= tid < vocab_full else unk_id for tid in enc_ids]

        if len(enc_ids) < self.block_size + 1:
            raise ValueError(
                f"[utils] Not enough tokens ({len(enc_ids)}) for block_size={self.block_size}. "
                "Use smaller block_size or provide more data."
            )

        self.tokens = torch.tensor(enc_ids, dtype=torch.long)
        print(f"[utils] Total tokens: {len(self.tokens):,}")

    def __len__(self):
        # number of training examples (each yields block_size tokens)
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def _collate(batch):
    xs, ys = zip(*batch)  # tuples of 1D LongTensors
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def create_dataloaders(
    data_dir: str,
    block_size: int = 512,
    batch_size: int = 16,
    val_ratio: float = 0.05,
    num_workers: int = 0,                  # safe on Windows
    tokenizer_name: str = DEF_THAI_TOKENIZER,
):
    """
    Windows-friendly DataLoaders:
      - num_workers: 0 (safest) or small (e.g., 2)
      - Only set prefetch_factor/persistent_workers when num_workers > 0
      - pin_memory=True
    """

    # --- Tokenizer ---
    tokenizer = load_tokenizer(tokenizer_name)
    if not hasattr(tokenizer, "__len__"):
        raise ValueError("[utils] Tokenizer missing __len__ (cannot determine full vocab size).")

    # --- Dataset & split ---
    dataset = TextDataset(data_dir, tokenizer, block_size=block_size)
    total_len = len(dataset)
    if total_len < 2:
        raise ValueError(
            f"[utils] Dataset too small ({total_len} samples). "
            "Use smaller block_size or add more data."
        )

    val_len = max(1, int(total_len * float(val_ratio)))
    train_len = max(1, total_len - val_len)
    if train_len + val_len > total_len:
        val_len = max(1, total_len - 1)
        train_len = max(1, total_len - val_len)

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    print(f"[utils] Train samples: {train_len:,} | Val samples: {val_len:,}")

    # --- Loader kwargs with conditional prefetch/persistent ---
    common = dict(
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        collate_fn=_collate,
        num_workers=num_workers,
    )

    train_kwargs = dict(shuffle=True, **common)
    val_kwargs   = dict(shuffle=False, **common)

    if num_workers > 0:
        train_kwargs.update(persistent_workers=False, prefetch_factor=2)
        val_kwargs.update(persistent_workers=False, prefetch_factor=2)

    train_dl = DataLoader(train_ds, **train_kwargs)
    val_dl   = DataLoader(val_ds,   **val_kwargs)

    vocab_size = len(tokenizer)
    print(f"[utils] Vocab size: {vocab_size:,}")
    print("[utils] DataLoaders ready ✅")

    return train_dl, val_dl, vocab_size, tokenizer




# model.py
# GPT blocks implemented from scratch; tokenizer is external (utils.py).

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """
    Hyperparameters that define model size and training behavior.

    Attributes
    ----------
    vocab_size : int
        Number of distinct token IDs (size of tokenizer vocabulary).
    block_size : int
        Maximum sequence length (context window). Positional embeddings are
        created up to this length, and the attention mask is sized to it.
    n_embd : int
        Width of token/hidden embeddings (channel dimension).
    n_head : int
        Number of attention heads per transformer block.
    n_layer : int
        Number of stacked transformer blocks.
    dropout : float
        Dropout probability used in attention weights and MLP.
    """
    vocab_size: int
    block_size: int = 1024
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2


# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------
class Head(nn.Module):
    """
    Single self-attention head with causal masking.

    Given input x of shape (B, T, C), projects to key/query/value and computes:
        softmax( (Q K^T) / sqrt(d_k) + causal_mask ) V
    to produce a contextualized representation of shape (B, T, head_size).

    Notes
    -----
    - Causal mask (lower-triangular) prevents attending to future tokens.
    - 'scale' improves training stability per "Attention is All You Need".
    """
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        # Linear projections for K, Q, V (bias=False is common in attention)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Precompute a lower-triangular mask of size (block_size, block_size)
        # Stored as a buffer so it moves with the module's device/dtype.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.scale = head_size ** -0.5  # 1/sqrt(d_k)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, C) where:
              B = batch size, T = sequence length, C = embedding width.

        Returns
        -------
        torch.Tensor
            Shape (B, T, head_size).
        """
        B, T, C = x.shape

        # Project to Q, K, V
        k = self.key(x)                  # (B, T, head_size)
        q = self.query(x)                # (B, T, head_size)
        v = self.value(x)                # (B, T, head_size)

        # Attention scores: (B, T, head_size) x (B, head_size, T) -> (B, T, T)
        wei = (q @ k.transpose(-2, -1)) * self.scale

        # Apply causal mask so positions can only attend to <= current index
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Normalize to a distribution across keys for each query position
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted sum of values: (B, T, T) x (B, T, head_size) -> (B, T, head_size)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention: runs several 'Head's in parallel
    and concatenates their outputs, followed by a linear projection.
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head  # assume n_embd divisible by n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate head outputs along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        # Final projection mixes information across heads
        out = self.dropout(self.proj(out))                   # (B, T, n_embd)
        return out


# -----------------------------------------------------------------------------
# MLP / Feed-Forward
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    Position-wise MLP used inside each transformer block.

    Implements: Linear(C -> 4C) -> ReLU -> Linear(4C -> C) -> Dropout
    The 4x expansion is a common design that increases model capacity.
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),                   # GELU is also common in GPT variants
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """
    A single Transformer decoder block:
        x = x + MHA(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    This is the "Pre-LN" layout (norm before sublayers), which stabilizes
    training and is widely used in modern Transformers.
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.sa(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.ff(self.ln2(x))
        return x


# -----------------------------------------------------------------------------
# GPT Model
# -----------------------------------------------------------------------------
class GPT(nn.Module):
    """
    Decoder-only Transformer for next-token prediction (language modeling).

    Forward inputs:
      - idx: LongTensor of token IDs with shape (B, T).
      - targets (optional): LongTensor of target token IDs with shape (B, T).
    Forward outputs:
      - logits: unnormalized scores over vocabulary, shape (B, T, V).
      - loss: cross-entropy over all positions if targets provided, else None.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding layers: tokens and absolute positions
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # (V, C)
        self.pos_emb   = nn.Embedding(cfg.block_size, cfg.n_embd)   # (Tmax, C)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])

        # Final normalization and LM head to vocab logits
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

        # Initialize weights with a small Gaussian as in GPT-style models
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Parameters
        ----------
        idx : torch.LongTensor
            Token IDs of shape (B, T).
        targets : torch.LongTensor, optional
            Next-token targets of shape (B, T). If provided, returns loss.

        Returns
        -------
        logits : torch.Tensor
            Shape (B, T, V) where V = vocab_size.
        loss : torch.Tensor or None
            Scalar loss if targets given, else None.
        """
        B, T = idx.shape

        # Token and position embeddings, summed (standard GPT approach)
        tok = self.token_emb(idx)                              # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device)) # (T, C)
        x = tok + pos                                          # broadcast to (B, T, C)

        # Transformer stack
        x = self.blocks(x)                                     # (B, T, C)
        x = self.ln_f(x)                                       # (B, T, C)

        # Project to vocabulary logits
        logits = self.lm_head(x)                               # (B, T, V)

        loss = None
        if targets is not None:
            # Flatten batch + time so each position is one training example
            B_, T_, V = logits.shape
            loss = F.cross_entropy(logits.view(B_ * T_, V), targets.view(B_ * T_))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size, temperature=1.0, top_k=None, top_p=None):
        """
        Autoregressively sample tokens from the model.

        Parameters
        ----------
        idx : LongTensor
            Initial context tokens, shape (B, T0).
        max_new_tokens : int
            Number of tokens to generate.
        block_size : int
            Context window to feed the model (usually cfg.block_size).
        temperature : float
            >1.0 makes sampling more random; <1.0 makes it sharper.
        top_k : int or None
            Keep only top-k highest-logit tokens (nucleus-like but fixed k).
        top_p : float or None
            Nucleus sampling: keep smallest set of tokens with cumulative
            probability >= top_p (e.g., 0.9).

        Returns
        -------
        LongTensor
            Concatenation of input and generated tokens, shape (B, T0 + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Use only the last `block_size` tokens as context (sliding window)
            idx_cond = idx[:, -block_size:]

            # Forward pass to get logits for the next token
            logits, _ = self(idx_cond)              # (B, t, V)
            logits = logits[:, -1, :]               # (B, V) only last time step
            logits = logits / max(1e-6, temperature)

            # Optional: top-k filtering (set others to -inf to zero them after softmax)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")

            # Optional: top-p (nucleus) filtering
            if top_p is not None:
                # Sort logits descending and compute cumulative probability
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)  # (B, V)
                probs = torch.softmax(sorted_logits, dim=-1)
                cdf = torch.cumsum(probs, dim=-1)

                # Mask tokens past the nucleus (except the first to keep at least one)
                mask = cdf > top_p
                mask[..., 0] = False
                sorted_logits[mask] = -float("inf")

                # Scatter back to the original index order
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)   # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token to the sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)

        return idx

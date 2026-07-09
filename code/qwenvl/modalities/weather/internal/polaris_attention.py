
import torch
from typing import Optional
from torch import nn
import torch.nn.functional as F
from einops import rearrange


__all__ = ["FlashAttention", "precompute_freqs_cis"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, compile: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.rmsnorm_fn = (
            torch.compile(self.compute_rmsnorm, fullgraph=True)
            if compile
            else self.compute_rmsnorm
        )

    @staticmethod
    def compute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        def _norm(x, eps):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(x.float(), eps).type_as(x)
        return output * weight

    def forward(self, x: torch.Tensor):
        return self.rmsnorm_fn(x, self.weight, self.eps)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq, xk = xq.contiguous(), xk.contiguous()
    freqs_cos, freqs_sin = freqs_cos.contiguous(), freqs_sin.contiguous()

    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    seq_len = xq_r.shape[1]
    assert seq_len <= freqs_cos.shape[0], (
        f"seq_len={seq_len} exceeds precomputed freqs buffer size={freqs_cos.shape[0]}"
    )
    freqs_cos = freqs_cos[:seq_len]
    freqs_sin = freqs_sin[:seq_len]

    freqs_cos = rearrange(freqs_cos, "n c -> 1 n 1 c")
    freqs_sin = rearrange(freqs_sin, "n c -> 1 n 1 c")  

    # Apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine real and imaginary parts
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)

    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )



class FlashAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dropout: float = 0.0,
        is_causal: bool = False,
        attn_implementation = None, 
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)


    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor=None,
        freqs_sin: torch.Tensor=None,
        mask: torch.Tensor = None,
    ):
        bsz, seq_len, _ = x.shape
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        if self.max_seq_len is None:
            xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   
            xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        else:
            nseq = self.max_seq_len // seq_len
            xq = xq.view(-1, nseq*seq_len, self.n_heads, self.head_dim)
            xk = xk.view(-1, nseq*seq_len, self.n_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        if not (freqs_cos is None or freqs_sin is None):
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if self.max_seq_len is not None:
            xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        
        # # grouped multiquery attention: expand out keys and values
        # xk = repeat_kv(xk, self.n_rep)  # (bs, seq_len, n_heads, head_dim)
        # xv = repeat_kv(xv, self.n_rep)  # (bs, seq_len, n_heads, head_dim)

        # Repeat KV heads if necessary
        if self.n_heads != self.n_kv_heads:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if mask is not None:
            mask = mask.to(xq).unsqueeze(1).contiguous()
            nW = mask.shape[0]
            if nW < bsz:
                mask = mask.repeat([bsz // nW, 1, 1, 1])        

        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=mask, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=self.is_causal
        )

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output




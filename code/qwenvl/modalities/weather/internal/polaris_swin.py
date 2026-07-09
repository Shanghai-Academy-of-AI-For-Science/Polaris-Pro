import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .polaris_attention import FlashAttention
from .polaris_layers import LayerNorm, RMSNorm
from .helpers import to_2tuple

__all__ = ["SwinBlock", "GeGLU_FFN"]


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

class GeGLU_FFN(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_dim=None, 
        multiple_of: int = 32, 
        dropout=0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.act = GeGLU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=init_std)
        
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SwiGLU_FFN(nn.Module):
    def __init__(
        self, 
        dim: int, 
        hidden_dim = None, 
        multiple_of: int = 32, 
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: (win_h, win_w)

    Returns:
        windows: (num_windows*B, win_h, win_w, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: (win_h, win_w)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, x, mask=None, **kwargs):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class WindowAttentionV2(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        

        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("k_bias", torch.zeros(dim).half())
        # self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x, mask=None, **kwargs):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        k_bias = self.k_bias.to(self.q_bias)
        qkv_bias = torch.cat([self.q_bias, k_bias, self.v_bias])
        # qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))        
        
        logit_scale = self.logit_scale.clamp(max=np.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table.to(x)).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        attn = attn + relative_position_bias.unsqueeze(0)
        
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn).to(v)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

def _compute_attn_mask(H, W, window_size, shift_size, mask_type):
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = (slice(0, -window_size[0]),
                slice(-window_size[0], -shift_size[0]),
                slice(-shift_size[0], None))
    w_slices = (slice(0, -window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            if mask_type == 'h':
                img_mask[:, h, :, :] = cnt
            elif mask_type == 'w':
                img_mask[:, :, w, :] = cnt
            elif mask_type == 'hw':
                img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


from transformers.modeling_layers import GradientCheckpointingLayer
class SwinBlock(GradientCheckpointingLayer):
    def __init__(
            self, dim, num_heads,
            input_size, window_size=7, shift_size=0, embed_dim=None,
            attn_type='v1', mask_type='hw', norm_type="ln", ffn_type="geglu_ffn",
            qk_scale=None, n_kv_heads=None,
            mlp_ratio=4., drop=0., attn_drop=0.,
            attn_implementation="flash_attention_2",
            **kwargs
        ):
        super().__init__()
        if embed_dim is None:
            embed_dim = dim
        self.dim = dim
        self.default_input_size = tuple(int(s) for s in input_size)
        self.num_heads = num_heads
        self.window_size = to_2tuple(window_size)
        self.shift_size = to_2tuple(shift_size)
        self.mlp_ratio = mlp_ratio
        self.norm_type = norm_type
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.mask_type = mask_type
        self._attn_implementation = attn_implementation

        default_full_window = all(
            w == s for w, s in zip(self.window_size, self.default_input_size)
        )

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        if norm_type == "ln":
            self.norm1 = LayerNorm(dim, eps=1e-6)
            self.norm2 = LayerNorm(dim, eps=1e-6)
        elif norm_type == "adaln":
            self.norm1 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            self.norm2 = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            adaln_linear = nn.Linear(embed_dim, 6 * dim, bias=True)
            nn.init.zeros_(adaln_linear.weight)
            nn.init.zeros_(adaln_linear.bias)
            self.adaln = nn.Sequential(nn.SiLU(), adaln_linear)
        elif norm_type == "adarms":
            self.norm1 = RMSNorm(dim, eps=1e-5)
            self.norm2 = RMSNorm(dim, eps=1e-5)
            adaln_linear = nn.Linear(embed_dim, 4 * dim, bias=True)
            nn.init.zeros_(adaln_linear.weight)
            nn.init.zeros_(adaln_linear.bias)
            self.adaln = nn.Sequential(nn.SiLU(), adaln_linear)
        else:
            raise ValueError(f"norm_type {norm_type} not supported")

        if attn_type == 'v1':
            self.attn = WindowAttention(
                dim, window_size=self.window_size, num_heads=num_heads,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif attn_type == 'v2':
            self.attn = WindowAttentionV2(
                dim, window_size=self.window_size, num_heads=num_heads,
                attn_drop=attn_drop, proj_drop=drop)
        elif attn_type == "flash":
            self.attn = FlashAttention(
                dim=dim,
                n_heads=num_heads,
                n_kv_heads=n_kv_heads,
                attn_implementation=self._attn_implementation,
                dropout=attn_drop,
                is_causal=False,
                max_seq_len=None if default_full_window else int(np.prod(self.default_input_size)),
            )
        else:
            raise ValueError(f"attn_type {attn_type} not supported")


        if ffn_type == "geglu_ffn":
            self.mlp = GeGLU_FFN(dim)
        elif ffn_type == "swiglu_ffn":
            self.mlp = SwiGLU_FFN(dim)
        else:
            raise ValueError(f"ffn_type {ffn_type} not supported")

        if not default_full_window and max(self.shift_size) > 0:
            H, W = self.default_input_size
            attn_mask = _compute_attn_mask(H, W, self.window_size, self.shift_size, mask_type)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        self._attn_mask_cache = {}

    def init_weights(self, init_std):
        for norm in (self.norm1, self.norm2):
            norm.reset_parameters()
        self.attn.init_weights(init_std)
        self.mlp.init_weights(init_std)

    def _get_attn_mask(self, H, W, device):
        if max(self.shift_size) == 0:
            return None
        if (H, W) == tuple(self.default_input_size):
            return self.attn_mask
        key = (H, W)
        if key not in self._attn_mask_cache:
            self._attn_mask_cache[key] = _compute_attn_mask(
                H, W, self.window_size, self.shift_size, self.mask_type
            )
        return self._attn_mask_cache[key].to(device)

    def _resolve_input_size(self, L):
        if L == self.default_input_size[0] * self.default_input_size[1]:
            return self.default_input_size
        wh, ww = self.window_size
        sqrt_l = int(L ** 0.5)
        for H in range(sqrt_l, 0, -1):
            if L % H != 0:
                continue
            W = L // H
            if H % wh == 0 and W % ww == 0:
                return (H, W)
        raise ValueError(
            f"Cannot resolve seq_len={L} into (H, W) divisible by window_size={self.window_size}"
        )

    def window_attention(self, x, freqs_cos, freqs_sin, input_size):
        H, W = input_size
        B, L, C = x.shape

        x = x.view(B, H, W, C)

        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        attn_mask = self._get_attn_mask(H, W, x.device)

        if self.attn_type == "flash" and (H, W) != tuple(self.default_input_size):
            old_max_seq = self.attn.max_seq_len
            self.attn.max_seq_len = None
            attn_windows = self.attn(x_windows, freqs_cos=freqs_cos, freqs_sin=freqs_sin, mask=attn_mask)
            self.attn.max_seq_len = old_max_seq
        else:
            attn_windows = self.attn(x_windows, freqs_cos=freqs_cos, freqs_sin=freqs_sin, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        return x


    def forward(
        self,
        x: torch.Tensor,
        embed: torch.Tensor=None,
        freqs_cos: torch.Tensor=None,
        freqs_sin: torch.Tensor=None,
        **kwargs,
    ):
        B, L, C = x.shape
        input_size = self._resolve_input_size(L)
        is_full_window = all(w == s for w, s in zip(self.window_size, input_size))

        shortcut = x

        if self.norm_type == "adaln":
            gamma = self.adaln(embed).unsqueeze(1)
            scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = gamma.chunk(6, dim=-1)
            x = self.norm1(x) * (1 + scale_msa) + shift_msa
        elif self.norm_type == "adarms":
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln(embed).unsqueeze(1).chunk(4, dim=-1)
            x = self.norm1(x) * (1 + scale_msa)
        else:
            x = self.norm1(x)

        if is_full_window:
            x = self.attn(x, freqs_cos, freqs_sin)
        else:
            x = self.window_attention(x, freqs_cos, freqs_sin, input_size)

        if self.norm_type == "adaln":
            x = shortcut + gate_msa * x
            x = x + gate_mlp*self.mlp(self.norm2(x)*(1+scale_mlp)+shift_mlp)
        elif self.norm_type == "adarms":
            x = shortcut + gate_msa * x
            x = x + gate_mlp*self.mlp(self.norm2(x)*(1+scale_mlp))
        else:
            x = shortcut + x
            x = x + self.mlp(self.norm2(x))

        return x
    

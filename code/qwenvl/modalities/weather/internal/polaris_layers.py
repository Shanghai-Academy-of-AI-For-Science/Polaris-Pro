# polaris_layers.py
from __future__ import annotations

import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------
# 基础归一化与嵌入模块 (保留兼容性)
# ---------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x, dim=-1):
        u = x.mean(dim, keepdim=True)
        s = x.var(dim, keepdim=True)   
        return (x - u) / torch.sqrt(s + self.eps) 
    
    def forward(self, x):
        output = self._norm(x.float()).to(x)
        if self.elementwise_affine:
            output = output * self.weight        
        return output

class RMS_norm(nn.Module):
    def __init__(self, dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        # normalize over last dim
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = x / rms * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x, dim=-1):
        return x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)        
        return output * self.weight
        
class PatchEmbedConv(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        patch_size=4,
        norm_func=None, 
        flatten=False,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.flatten = flatten    

        # ⭐ 关键：patch_size=1 时保持老行为；否则按 patch_size 下采样
        if self.patch_size == (1, 1):
            # 保留你原来的设计：kernel_size=(2,1), stride=1
            kernel_size = (2, 1)
            stride = (1, 1)
            padding = (0, 0)
        else:
            # 正常的 patch 下采样：kernel_size=stride=patch_size
            kernel_size = self.patch_size
            stride = self.patch_size
            padding = (0, 0)

        self.proj = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm_func(out_chans) if norm_func else nn.Identity()

    def forward(self, x):
        embed = self.proj(x)
        if self.flatten:
            embed = rearrange(embed, 'n c h w -> n (h w) c')
        return self.norm(embed)
               
class WeatherLayerNorm(nn.Module):
    """
    专为气象 (B, C, H, W) 格式设计的高效 LayerNorm (Channel-first)。
    适用于深层全卷积潜空间演化。
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        
    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class AdaLN(nn.Module):
    def __init__(self, dim, embed_dim=None):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        in_dim = embed_dim if embed_dim else dim
        self.scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, 2 * dim, bias=True),
        )

    def forward(self, x, embed):
        scale, shift = self.scale_shift(embed).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x

def sincos_embedding(x, embed_dim, max_period=10000, is_periodic=False):
    omega = torch.arange(embed_dim//2, dtype=x.dtype, device=x.device)
    if is_periodic:
        x = 2 * torch.pi * x
    else:
        omega /= embed_dim / 2.
        omega = 1. / max_period ** omega  
    out = torch.einsum('m,d->md', x.reshape(-1), omega) 
    emb_sin = torch.sin(out)  
    emb_cos = torch.cos(out)  
    emb = torch.cat([emb_sin, emb_cos], dim=1) 
    return emb

class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size, frequency=256, is_periodic=False, sinusoidal=True, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency = frequency
        self.is_periodic = is_periodic
        self.sinusoidal = sinusoidal
        # Optional output-side dropout. Helps prevent the encoder from
        # latching onto specific (hour, doy) combinations when training
        # data spans many years and most doy values appear repeatedly.
        # ``dropout=0.0`` (default) is a no-op so existing checkpoints
        # see identical behaviour.
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        if not self.sinusoidal:
            return self.dropout(self.mlp(x))
        embed = sincos_embedding(
            x.float(), self.frequency, is_periodic=self.is_periodic
        ).type_as(x)
        embed = self.mlp(embed)
        return self.dropout(embed)

# ---------------------------------------------------------
# 气象定制：潜空间大核卷积 Block
# ---------------------------------------------------------
class GlobalWeatherConvBlock(nn.Module):
    """
    经度循环大核 CNN Block，专为地球流体物理推演设计。
    替代原有的 Window Attention，解决网格伪影并大幅提升感受野。
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        # 7x7 大核深度可分离卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, groups=dim, bias=False)
        self.norm = WeatherLayerNorm(dim)
        
        # Inverted Bottleneck: 1x1 卷积放大 4 倍通道进行特征融合
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        
        x = F.pad(x, pad=(3, 3, 3, 3), mode='replicate')  

        x = x.contiguous()
        
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        return shortcut + self.drop_path(x)

# ---------------------------------------------------------
# Encoder: 单帧气象特征提取网络
# ---------------------------------------------------------
class CubeEmbedConv(nn.Module):
    """
    重构后的单帧推演 Encoder。
    包含三阶段：前置抗混叠 -> Patch下采样 -> 潜空间大核深层演化。
    """
    def __init__(
        self,
        in_chans: int = 70,
        out_chans: int = 2048,
        in_frames: int = 1,          # 现固定为单帧输入
        patch_size: int = 6,         # 默认 6 倍下采样 (120x240)
        depth: int = 2,              # 潜空间推演层数 (建议 8-16)
        flatten: bool = False,     
        norm_func: Optional[nn.Module] = None,
        # 兼容旧签名的冗余参数，防止外部调用报错
        temporal_pad_to: int = 2,  
        attn_heads: int = 8,
        attn_dropout: float = 0.0,
        window_size: int = 8,
        keep_time_dim: bool = False,
        feat_chans: int = 2048,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.flatten = flatten

        # 阶段一：高分辨率前置特征融合 (抗混叠，避免下采样丢失高频气象细节)
        inter_chans = 256
        self.stage1_smooth = nn.Sequential(
            nn.Conv2d(in_chans, inter_chans, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            WeatherLayerNorm(inter_chans),
            nn.GELU()
        )

        # 阶段二：Patch 强力下采样 (无重叠，直接拉升到目标潜空间通道数)
        self.stage2_patchify = nn.Sequential(
            nn.Conv2d(inter_chans, out_chans, kernel_size=patch_size, stride=patch_size),
            WeatherLayerNorm(out_chans)
        )

        # 阶段三：潜空间大核全局演化 (替代 Window Attention)
        self.stage3_evolution = nn.Sequential(
            *[GlobalWeatherConvBlock(dim=out_chans) for _ in range(depth)]
        )
        
        self.norm = norm_func(out_chans) if norm_func is not None else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        lead_hour: Optional[int] = None,       # 兼容签名
        lead_hour_nums: Optional[int] = None,  # 兼容签名
    ) -> torch.Tensor:
        """
        x: (B, T, C, H, W) 或 (B, C, H, W). 期望 T==1。
        """
        # 1. 处理维度，兼容 T 维度
        if x.ndim == 5:
            assert x.shape[1] == 1, f"CubeEmbedConv expects single frame (T=1), got {x.shape}"
            x = x.squeeze(1)  # 转换为 (B, C, H, W)
        
        # print("x0",x.shape)
        # 2. 三阶段推演
        x = self.stage1_smooth(x)
        # print("x1",x.shape)
        x = self.stage2_patchify(x)
        # print("x2",x.shape)
        z = self.stage3_evolution(x)
        # print("z",z.shape)
        
        # 3. 输出格式化
        if self.flatten:
            z_tok = rearrange(z, "b c h w -> b (h w) c")
            return self.norm(z_tok)
            
        return self.norm(z)

# ---------------------------------------------------------
# Decoder: 气象场重构网络
# ---------------------------------------------------------
def remove_small_scales(x, scale_factor=0.5, mode="bilinear", random_scale=False):
    import numpy as np
    if random_scale:
        scale_factor = np.random.choice([1 / s for s in range(1, 13)])
    if scale_factor < 1:
        down = F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
        x = F.interpolate(down, size=x.shape[-2:], mode=mode, align_corners=False)
    return x

class SmoothDeconv(nn.Module):
    """
    数学精确匹配 patch_size 的平滑反卷积。
    彻底修复原版本中存在的空间偏移(Phase Shift)和形状裁切问题。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple],
        bias: bool = True, 
    ):
        super().__init__()
        # 直接使用与 stride 完全相等的 kernel_size，避免 checkerboard artifacts
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,        # 完美对齐，不需要人为 padding
            output_padding=0, 
            bias=bias
        )
        # 可选：后接一层普通卷积做进一步平滑
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x, output_size=None):
        x = self.deconv(x)
        x = self.smooth(x)
        # 如果有极微小的边界舍入误差，进行安全裁切
        if output_size is not None and x.shape[-2:] != output_size:
            x = x[..., :output_size[0], :output_size[1]]
        return x


class ProgressiveUpsample(nn.Module):
    """
    阶梯式渐进上采样模块。
    完美平衡“信息保留”与“防显存溢出(OOM)”的矛盾，严格遵循流体力学的连续性。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # 阶段一：空间放大 2 倍，通道 2048 -> 512
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU()
        )
        
        # 阶段二：空间放大 3 倍，通道 512 -> 128
        self.stage2_conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU()
        )
        
        # 最终映射：128 -> 具体的物理变量通道数 (如 upper_chans)
        self.head = nn.Conv2d(128, out_channels, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x, target_size=(721, 1440)):
        # x 初始状态: (B, 2048, 120, 240)
        
        # --- Stage 1: 先插值放大空间，再卷积融合降维 (保留高频信息) ---
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.stage1_conv(x)  # 此时形状: (B, 512, 240, 480)
        
        # --- Stage 2: 再次插值放大空间，再降维 ---
        x = F.interpolate(x, scale_factor=3.0, mode='bilinear', align_corners=False)
        x = self.stage2_conv(x)  # 此时形状: (B, 128, 720, 1440)
        
        # --- 解决 720 与 721 的地球极点拓扑问题 ---
        current_h, current_w = x.shape[-2:]
        target_h, target_w = target_size
        
        if current_h != target_h or current_w != target_w:
            pad_h = target_h - current_h  # 721 - 720 = 1
            pad_w = target_w - current_w  # 1440 - 1440 = 0
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
            
        # --- 最终映射输出 ---
        x = self.head(x)
        return x


class DoubleDeconvHead(nn.Module):
    def __init__(
        self, 
        in_chans: int, 
        upper_chans: int,
        lower_chans: int,
        patch_size: int,
    ):
        super().__init__()
        self.lower_chans = lower_chans
        self.upper_chans = upper_chans

        # 换用阶梯式渐进上采样
        self.upper_head = ProgressiveUpsample(in_channels=in_chans, out_channels=upper_chans)

        if self.lower_chans > 0:
            self.lower_head = ProgressiveUpsample(in_channels=in_chans, out_channels=lower_chans)

    def forward(self, h, residual=None, input_size=(721, 1440), lead_hour=None, target_frames: int = 1):
        
        # ==========================================
        # 🚨 修复 1：安全处理 Batch Tensor 级别的 lead_hour
        # ==========================================
        if lead_hour is not None:
            # lead_hour shape: (B,)
            # 等价于 max(lead_hour // 6, 1) 的 Tensor 写法
            lead_step = torch.clamp(lead_hour // 6, min=1.0) 
            # 广播到 (B, 1, 1, 1) 以便与图像特征相乘
            lead_step = lead_step.view(-1, 1, 1, 1).to(h.dtype)
        else:
            lead_step = 1.0
        
        # ==========================================
        # 2. 解码预测变化量 (Delta)
        # ==========================================
        pred = self.upper_head(h, target_size=input_size)
        output = pred * lead_step
        
        if self.lower_chans > 0:
            lower = self.lower_head(h, target_size=input_size)
            # 如果下层也需要乘时间步，取消下面的注释：
            # lower = lower * lead_step
            output = torch.cat([output, lower], dim=1)
        
        # ==========================================
        # 🚨 修复 2：残差相加必须在 FP32 精度下进行！
        # ==========================================
        if residual is not None:

            res_frame = residual[:, -1] if residual.ndim == 5 else residual
            res_frame = remove_small_scales(res_frame)
            

            # 强制提升至 FP32 抵抗 bf16 的精度吞噬
            output = output.to(torch.float32)
            res_frame = res_frame.to(torch.float32)
            
            output = output + res_frame
        
        # 返回 fp32 结果，直接交给外面的 MSE Loss 计算（Loss也需要 fp32 保证稳定）
        return output
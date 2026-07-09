import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from einops import rearrange

__all__ = [
    "make_seq", "crop_dataarray", "filter_dataset_channels", "load_meteorological_buffers",
    "TYPHOON_PAD_MULTIPLE", "pad_spatial_to_multiple", "crop_spatial_pad",
    "build_channel_map", "pad_channels", "extract_valid_channels",
]

TYPHOON_PAD_MULTIPLE = 120


def pad_spatial_to_multiple(x: torch.Tensor, multiple: int = TYPHOON_PAD_MULTIPLE):
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (pl, pr, pt, pb), mode="constant", value=0)
    return x, (pt, pb, pl, pr)


def crop_spatial_pad(x: torch.Tensor, pad_info):
    pt, pb, pl, pr = pad_info
    hs = slice(pt, x.shape[-2] - pb if pb > 0 else x.shape[-2])
    ws = slice(pl, x.shape[-1] - pr if pr > 0 else x.shape[-1])
    return x[..., hs, ws]


def build_channel_map(src_names, ref_names):
    ref_idx = {n: i for i, n in enumerate(ref_names)}
    src_to_ref = np.array([ref_idx.get(str(n), -1) for n in src_names], dtype=np.int64)
    mask = np.zeros(len(ref_names), dtype=bool)
    for i in src_to_ref:
        if 0 <= i < len(ref_names):
            mask[i] = True
    return src_to_ref, mask


def pad_channels(x_src, src_to_ref, ref_c):
    c = x_src.shape[0]
    out = np.zeros((ref_c,) + x_src.shape[1:], dtype=np.float32)
    for j in range(c):
        i = int(src_to_ref[j])
        if 0 <= i < ref_c:
            out[i] = x_src[j]
    return out


def extract_valid_channels(x_ref, src_to_ref, src_c):
    out = np.zeros((src_c,) + x_ref.shape[1:], dtype=np.float32)
    for j in range(src_c):
        i = int(src_to_ref[j])
        if 0 <= i < x_ref.shape[0]:
            out[j] = x_ref[i]
    return out

# 🚨 需要剔除的通道
DEFAULT_REMOVE_CHANNELS = [
    "q2m", "d2m", "sst", "ws100m", "u100m", "v100m",
    "lcc", "mcc", "hcc", "tcc", "ssr", "ssrd", "fdir", "ttr", "tcw", "tp"
]

def make_seq(ds, total_frames, frame_interval, frame_step=1, ignore_times=[]):
    i = 0
    inds = []
    times = ds.time.values
    ignore_timestamps = {pd.to_datetime(t) for t in ignore_times}
    
    while i < len(times) - total_frames:
        cur_sequence = []
        for j in range(i, i + total_frames - 1):
            if times[j+1] - times[j] == frame_interval:
                current_time = pd.to_datetime(times[j])
                if current_time in ignore_timestamps:
                    continue
                cur_sequence.append(j)

        if len(cur_sequence) == total_frames - 1:
            inds.append(i)

        i += frame_step

    return np.array(inds, dtype=np.int32) 

def crop_dataarray(ds, image_size=None, latlon_range=None):
    if "lat" in ds.dims:
        ilats = np.arange(ds.lat.size)
        if latlon_range is not None:
            lat_min, lat_max, lon_min, lon_max = latlon_range
            ilats = np.where((ds.lat >= lat_min) & (ds.lat <= lat_max))[0]
        if image_size is not None:
            ilats = ilats[:image_size[0]]
        ds = ds.isel(lat=ilats)

    if "lon" in ds.dims:
        ilons = np.arange(ds.lon.size)
        if latlon_range is not None:
            lat_min, lat_max, lon_min, lon_max = latlon_range
            ilons = np.where((ds.lon >= lon_min) & (ds.lon <= lon_max))[0]
        if image_size is not None:
            ilons = ilons[:image_size[1]]
        ds = ds.isel(lon=ilons)
    return ds

def filter_dataset_channels(ds, remove_channels=None):
    """
    统一的通道过滤函数：Dataset 和 Buffer 读取都调用这个函数。
    """
    if remove_channels is None:
        remove_channels = DEFAULT_REMOVE_CHANNELS
        
    if isinstance(ds, xr.Dataset):
        ds = ds[list(ds.data_vars)[0]]
        
    all_channels = ds.channel.values.tolist()
    keep_channels = [c for c in all_channels if c not in remove_channels]
    keep_inds = [i for i, c in enumerate(all_channels) if c in keep_channels]
    
    out_ds = ds.sel(channel=keep_channels)
    return out_ds, keep_channels, keep_inds

def load_meteorological_buffers(
    data_path: str, 
    image_size=None,
    latlon_range=None,
    remove_channels=None,
    buffer_types=None,
    index_names=None
):
    """
    统一的数据提取。
    """
    if buffer_types is None:
        buffer_types = [
            "mean", "std", "diff_mean", "diff_std", "climate",
            "const", "weight", "channel_mask", "land_mask", "station_mask", "rps_mask", "dry_mask"
        ]
    if index_names is None:
        index_names = dict(logid=[], uid=["u10m"], vid=["v10m"], accumid=[])

    ds = xr.open_zarr(data_path)
    if "level" in ds.dims:
        ds = ds.rename({"level": "channel"})
    ds = crop_dataarray(ds, image_size, latlon_range)

    # 🚨 调用统一过滤函数
    ds, keep_channels, keep_inds = filter_dataset_channels(ds, remove_channels)

    indices = {}
    for k, prefixes in index_names.items():
        inds = [i for i, name in enumerate(keep_channels) if name in prefixes]
        if len(inds) > 0:
            indices[k] = inds
    
    coords = dict(lat=ds.lat.values.tolist(), lon=ds.lon.values.tolist())
    
    buffers = {}
    for k in buffer_types:
        file_name = os.path.join(data_path, f"{k}.nc")
        if not os.path.exists(file_name):
            continue

        da = xr.open_dataarray(file_name)
        da = crop_dataarray(da, image_size, latlon_range)

        if k in ["mean", "std", "diff_mean", "diff_std"]:
            if "channel" in da.dims:
                da = da.sel(channel=keep_channels)
            else:
                da_vals = da.values[keep_inds]
                da = xr.DataArray(da_vals, coords={"channel": keep_channels}, dims=("channel",))

        values = da if k == "climate" else da.values

        if k == "const":
            values = rearrange(values, "c h w -> 1 c h w")
        elif k == "weight" and values.ndim == 1:
            values = rearrange(values, "h -> 1 h 1")
        elif k in ["mean", "std", "diff_mean", "diff_std"]:
            values = rearrange(values, "c -> c 1 1")

        buffers[k] = values

    return keep_channels, indices, coords, buffers
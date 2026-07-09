import collections.abc
from itertools import repeat
from inspect import isfunction
from torch.nn import functional as F

__all__ = ["to_2tuple", "round_to_multiple", "exists", "default", "append_dims"]

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def round_to_multiple(x: int, multiple_of: int = 32) -> int:
    """Round up x to the nearest multiple of multiple_of."""
    return int(multiple_of * ((x + multiple_of - 1) // multiple_of))


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]



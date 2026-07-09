"""Loss helpers for the med_seg (SAM3) decoder.

Adapted from Sam_uni reference implementation. Pure functions — no module
state — so they're cheap to import and easy to unit-test.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def dice_loss_sum_from_logits(logits: Tensor, targets: Tensor, eps: float = 1.0) -> Tensor:
    probs = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(-1)
    denom = probs.sum(-1) + targets.sum(-1)
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).sum()


def xyxy_norm_to_cxcywh_norm(boxes_xyxy: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1).clamp(min=0.0)
    bh = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, bw, bh], dim=-1)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1_xyxy: Tensor, boxes2_xyxy: Tensor) -> Tensor:
    """GIoU between two sets of boxes (paired, same length).

    Numerical-stability notes (matter in bf16 + small/degenerate boxes):
      * normalised box widths/heights can underflow to 0 in bf16
        (anything below ~1/256 rounds to 0). Cast to fp32 internally so
        the eps in `union + eps` and `c_area + eps` actually shields the
        division — in bf16 a 1e-6 eps is itself rounded to 0.
      * use a relatively large eps (1e-4) because dx/(union+eps)^2 is the
        backward gradient: with eps=1e-6 and union≈eps you get a 1e12
        gradient blow-up, which is the dominant grad_norm source on
        small/empty box queries early in training.
    """
    boxes1_xyxy = boxes1_xyxy.float()
    boxes2_xyxy = boxes2_xyxy.float()
    eps = 1e-4
    x1 = torch.max(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
    y1 = torch.max(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
    x2 = torch.min(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
    y2 = torch.min(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    a1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(min=0) \
        * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]).clamp(min=0)
    a2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(min=0) \
        * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]).clamp(min=0)
    union = a1 + a2 - inter
    iou = inter / (union + eps)

    cx1 = torch.min(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
    cy1 = torch.min(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
    cx2 = torch.max(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
    cy2 = torch.max(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
    return iou - (c_area - union) / (c_area + eps)


def generalized_box_iou_pairwise(boxes1_xyxy: Tensor, boxes2_xyxy: Tensor) -> Tensor:
    """Pairwise GIoU [N, M] — used by the matcher."""
    a1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(min=0) \
        * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]).clamp(min=0)
    a2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(min=0) \
        * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]).clamp(min=0)

    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - inter
    iou = inter / (union + 1e-6)

    lt_c = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb_c = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    c_area = wh_c[..., 0] * wh_c[..., 1]
    return iou - (c_area - union) / (c_area + 1e-6)


def clip_xyxy(box, w: int, h: int):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def xyxy_abs_to_cxcywh_norm(box_xyxy, w: int, h: int):
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = max(0.0, (x2 - x1) / w)
    bh = max(0.0, (y2 - y1) / h)
    return [
        min(1.0, max(0.0, cx)),
        min(1.0, max(0.0, cy)),
        min(1.0, max(0.0, bw)),
        min(1.0, max(0.0, bh)),
    ]


def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]

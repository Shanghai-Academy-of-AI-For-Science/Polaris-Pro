import torch
from typing import Dict, Optional, Sequence, List, Tuple


def _build_pad_id_sets(bio_token_ids: Optional[Dict] = None):
    """Build sets of start/pad token IDs for modality-aware rope computation.

    Returns ``(all_start_ids, image_like_pad_ids, video_like_pad_ids)``
    where:
    - *all_start_ids*: token IDs that mark the beginning of a modality segment.
    - *image_like_pad_ids*: pad token IDs whose segments use ``image_grid_thw``.
    - *video_like_pad_ids*: pad token IDs whose segments use ``video_grid_thw``.
    """
    vision_start_token_id = 151652
    image_token_id = 151655
    video_token_id = 151656

    all_start_ids = {vision_start_token_id}
    image_like_pad_ids = {image_token_id}
    video_like_pad_ids = {video_token_id}

    if bio_token_ids:
        # RNA / DNA / protein / mol / weather latent tokens are 1-D
        # (h=w=1) -> image-like grid.
        # Any future / unknown bio modality defaults to video-like (2-D) routing.
        _IMAGE_LIKE = {"rna", "dna", "protein", "mol", "weather"}
        for name, ids in bio_token_ids.items():
            if name.startswith("_"):
                continue
            start_id = ids.get("start")
            pad_id = ids.get("pad")
            if start_id is not None:
                all_start_ids.add(start_id)
            if pad_id is not None:
                if name in _IMAGE_LIKE:
                    image_like_pad_ids.add(pad_id)
                else:
                    video_like_pad_ids.add(pad_id)

    return all_start_ids, image_like_pad_ids, video_like_pad_ids


def _find_first_in_set(tokens: List[int], start: int, id_set: set, sentinel: int) -> int:
    """Return index of the first token in *id_set* starting from *start*."""
    for idx in range(start, len(tokens)):
        if tokens[idx] in id_set:
            return idx
    return sentinel


def _reorder_image_like_grids_by_text_pos(
    input_ids: torch.Tensor,
    bio_grids: Dict[str, torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    bio_token_ids: Optional[Dict],
    image_token_id: int = 151655,
) -> Optional[torch.Tensor]:
    """Concatenate image-like grids in TEXT-order so ``get_rope_index_3``
    consumes them in the same order it walks pad tokens.

    Without this reordering, when one sample contains pad tokens from multiple
    sources whose registration order differs from text-appearance order
    (e.g. text has ``<mol>...<protein>...`` but the modality registry was
    built as ``[rna, dna, protein, mol]``), the grids fed to RoPE are
    permuted relative to the pad blocks they describe, producing silently
    wrong position ids.

    Walks each sample independently, since cross-modal layout can differ
    per sample, and concatenates the per-sample reordered grids in
    sample-major × text-position-minor order — which matches how the
    Qwen3-VL rope helpers iterate batches.

    Args:
        input_ids: ``[B, L]`` LLM token ids.
        bio_grids: mapping ``{modality_name -> [N_mod_total, 3]}`` where the
            rows are ordered as the modality's encoder produces them
            (sample-major × text-position-minor within a sample).
        image_grid_thw: ``[N_img_total, 3]`` real-image grids, or ``None``.
        bio_token_ids: ``config.bio_token_ids`` — used to map modality name
            back to its pad token id.
        image_token_id: the LLM token id for image patches (default Qwen3-VL).

    Returns:
        ``[N_all, 3]`` tensor combining all image-like grids in text order,
        or ``None`` when there is nothing to scatter.
    """
    if not bio_grids and image_grid_thw is None:
        return None

    bio_token_ids = bio_token_ids or {}

    # pad_id → ("bio", mod_name) | ("img", None).
    src: Dict[int, Tuple[str, Optional[str]]] = {}
    for mod in bio_grids:
        pad = bio_token_ids.get(mod, {}).get("pad")
        if pad is not None:
            src[int(pad)] = ("bio", mod)
    if image_grid_thw is not None:
        src[int(image_token_id)] = ("img", None)

    if not src:
        # No pad ids known — fall back to plain concatenation in registration
        # order, preserving legacy behaviour for callers that haven't wired
        # ``bio_token_ids`` through.
        rows = list(bio_grids.values())
        if image_grid_thw is not None:
            rows.append(image_grid_thw)
        return torch.cat(rows, dim=0) if rows else None

    bio_offsets = {mod: 0 for mod in bio_grids}
    img_offset = 0
    out_rows: List[torch.Tensor] = []

    pad_id_tensor = torch.tensor(sorted(src.keys()), device=input_ids.device)

    for b in range(input_ids.shape[0]):
        ids_b = input_ids[b]
        # Build a boolean "is pad" mask over the whole sample.
        is_pad = torch.isin(ids_b, pad_id_tensor)
        if not is_pad.any():
            continue
        # Detect block starts: positions where ids_b changes value AND lands
        # on a pad token.  A pad block is K identical pad ids; only the FIRST
        # of each run counts as a new sequence to consume.
        prev = torch.cat(
            [torch.full((1,), -1, dtype=ids_b.dtype, device=ids_b.device), ids_b[:-1]]
        )
        block_start = is_pad & (ids_b != prev)
        positions = block_start.nonzero(as_tuple=True)[0].tolist()
        for pos in positions:
            pid = int(ids_b[pos].item())
            kind, mod = src[pid]
            if kind == "bio":
                grid = bio_grids[mod]
                idx = bio_offsets[mod]
                if idx >= grid.shape[0]:
                    raise ValueError(
                        f"_reorder_image_like_grids_by_text_pos: text contains "
                        f"more '{mod}' pad blocks than the encoder produced "
                        f"({grid.shape[0]} grids). Sample {b}, pos {pos}."
                    )
                out_rows.append(grid[idx])
                bio_offsets[mod] = idx + 1
            else:
                if image_grid_thw is None or img_offset >= image_grid_thw.shape[0]:
                    raise ValueError(
                        f"_reorder_image_like_grids_by_text_pos: text contains "
                        f"more image pad blocks than image_grid_thw rows. "
                        f"Sample {b}, pos {pos}."
                    )
                out_rows.append(image_grid_thw[img_offset])
                img_offset += 1

    # Sanity: every grid row must have been consumed.
    for mod, idx in bio_offsets.items():
        if idx != bio_grids[mod].shape[0]:
            raise ValueError(
                f"_reorder_image_like_grids_by_text_pos: '{mod}' has "
                f"{bio_grids[mod].shape[0]} encoder grids but only {idx} "
                f"pad blocks were found in input_ids."
            )
    if image_grid_thw is not None and img_offset != image_grid_thw.shape[0]:
        raise ValueError(
            f"_reorder_image_like_grids_by_text_pos: image_grid_thw has "
            f"{image_grid_thw.shape[0]} rows but only {img_offset} image "
            f"pad blocks were found in input_ids."
        )

    return torch.stack(out_rows, dim=0) if out_rows else None


def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    bio_token_ids: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL timestamp-based rope index with per-modality token support.

    Modality segments are identified by ``(start_token, pad_token)`` pairs.
    The *pad_token* determines whether the segment uses ``image_grid_thw``
    (image-like) or ``video_grid_thw`` (video-like) for spatial positions.
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    all_start_ids, image_like_pad_ids, video_like_pad_ids = _build_pad_id_sets(bio_token_ids)

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]

            # Find all modality start positions and classify by pad type
            start_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for sid in all_start_ids:
                start_mask |= (input_ids == sid)
            start_indices = start_mask.nonzero(as_tuple=True)[0]

            image_nums, video_nums = 0, 0
            if len(start_indices) > 0:
                next_tokens = input_ids[torch.clamp(start_indices + 1, max=len(input_ids) - 1)]
                for nt in next_tokens:
                    nt_val = nt.item()
                    if nt_val in image_like_pad_ids:
                        image_nums += 1
                    elif nt_val in video_like_pad_ids:
                        video_nums += 1

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            sentinel = len(input_tokens) + 1

            for _ in range(image_nums + video_nums):
                ed_image = (
                    _find_first_in_set(input_tokens, st, image_like_pad_ids, sentinel)
                    if remain_images > 0 else sentinel
                )
                ed_video = (
                    _find_first_in_set(input_tokens, st, video_like_pad_ids, sentinel)
                    if remain_videos > 0 else sentinel
                )

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                is_latent = (h.item() == 1 and w.item() == 1)
                merge = 1 if is_latent else spatial_merge_size
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // merge,
                    w.item() // merge,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

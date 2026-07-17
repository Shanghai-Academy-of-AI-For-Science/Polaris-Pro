# Medical-image segmentation

神珍 performs **text-prompted medical-image segmentation**: given an
image and a text description of a target region, the backbone conditions a
**SAM 3** decoder to output the mask. This modality has its own script.

## ⚠️ SAM 3 weights are embedded — under Meta's SAM License

The decoder is a **fine-tuned SAM 3** backbone whose weights are embedded in the
released `model.safetensors`; its topology and processor config ship in
`model/sam3/`. These SAM 3 weights are governed by **Meta's SAM License**
(`SAM_LICENSE.txt`), not Apache-2.0, and their use is subject to that license's
acceptable-use restrictions (no military / weapons / illegal uses; Trade-Control
compliance). Everything else in 神珍 is Apache-2.0.

## Requirements
- `transformers==5.0.0` (provides `Sam3Model`), `opencv-python-headless`.
- `scipy` — only for the optional mask post-processing (connected-component
  filtering / closing); everything else runs without it.
- A GPU with ≥ 24 GB memory for the 2016×2016 SAM 3 input. Test-time
  augmentation (tiling / multi-scale, below) raises memory and runtime.

## Run

### Single image (most common)

Segment one image by describing the target region in text:

```bash
export PYTHONPATH=$PWD/code

python code/scripts/medseg/infer_med_seg_qwen3vl.py \
    --ckpt_path model \
    --image path/to/image.png \
    --prompt "left heart ventricle in cardiac MRI" \
    --results_root out/medseg \
    --method mkb \
    --save_vis
```

This writes the predicted mask to `out/medseg/<image>_mask.png` (and, with
`--save_vis`, an overlay `<image>_overlay.png`). No ground truth or metrics are
needed in this mode. The mask comes from the `argmax` read-out head by default;
use `--single_image_head semantic` (or `union`) for multi-region targets — see
[Heads and test-time augmentation](#heads-and-test-time-augmentation) below.

### Dataset folder (batch + metrics)

For batch evaluation against ground-truth masks, point `--data_root` at a
dataset in **BiomedParse layout**; the script computes Dice per unit. The
defaults (full + tile views, all three heads, light post-processing) give a
good speed/accuracy balance on one GPU:

```bash
python code/scripts/medseg/infer_med_seg_qwen3vl.py \
    --ckpt_path model \
    --data_root <dataset_root> \
    --results_root out/medseg \
    --method mkb \
    --save_pred_masks
```

- **Faster:** add `--no-use_tile` for a single full-image view.
- **Higher accuracy:** add `--use_hflip --multi_scale_image_sizes 1680`
  (slower, more memory — see below).

**What `--data_root` must contain.** The script recursively finds every
`test.json` beneath `--data_root`; each directory holding a `test.json` is one
"unit" and must look like:

```
<unit>/
├── test.json          # index: images + annotations (schema below)
├── test/              # the input images (referenced by file_name)
│   └── case001.png
└── test_mask/         # the ground-truth masks, one per annotation
    └── case001_left+ventricle.png
```

`--data_root` may point at a single unit, or at a parent folder containing many
units (each is evaluated separately). Image folders named `test/`, `images/`, or
`imgs/` are accepted; mask folders named `test_mask/`, `masks/`, etc.

**`test.json` schema** (COCO-style; only the fields below are read):

```jsonc
{
  "images": [
    { "id": 25, "file_name": "case001.png", "height": 1024, "width": 1024 }
  ],
  "annotations": [
    {
      "image_id": 25,                                  // links to images[].id
      "file_name": "case001.png",                      // input image (in test/)
      "mask_file": "case001_left+ventricle.png",       // GT mask (in test_mask/)
      "sentences": [ { "sent": "left heart ventricle in cardiac MRI" } ]  // the text prompt
    }
  ]
}
```

One annotation = one (image, text-prompt, GT-mask) instance. The prompt is taken
from `sentences[].sent` (or `sent`/`raw`/`text`/`phrase`/`caption`, or the
`category_id` name as a fallback).

**Outputs.** Per unit, under `out/medseg/<unit>/`: predicted masks in
`pred_masks/<head>/` (with `--save_pred_masks`), optional overlays in
`vis/<head>/` (with `--save_vis`), and one metrics file **per read-out head**
`mkb_<unit>_dataset_metrics_<head>.json` (e.g. `..._argmax.json`,
`..._semantic.json`, `..._union.json`) containing Dice/IoU scores plus
per-instance results. Compare the heads and pick the best for your data — see
below.

## Heads and test-time augmentation

The decoder emits several candidate segmentation queries per image. The script
reduces them into **read-out heads** and can fuse **multiple views** of the
image before thresholding. These only change how predictions are read out — the
model weights are untouched.

**Read-out heads** (`--heads argmax,semantic,union`, all three by default):

| Head | How the mask is formed | Best for |
|------|------------------------|----------|
| `argmax` | Single highest-presence query. | One clear target per image (organs: CT / MRI / X-ray / dermoscopy). |
| `semantic` | Pixel-wise max of (presence × mask-prob) over queries. | Diffuse / multi-region targets (vessels, cells, lesions: DRIVE / GlaS / OCT). |
| `union` | Hard OR of all queries scoring above `--score_thr`. | Several discrete instances of the same class. |

On single-target data `semantic` can legitimately collapse to empty — that is
expected; use `argmax` there. Pick the head whose `_<head>.json` Dice is best
for your modality.

**Test-time augmentation.** The image is run through several views, fused at the
probability level, then heads are read from the fused tensor:

- `--use_full` / `--use_tile` — full-image and sliding-window tile views. Both
  **on by default**; tiling (`--tile_size 512 --tile_overlap 0.5`) helps small /
  thin structures. Pass `--no-use_tile` (or `--no-use_full`) to disable.
- `--use_hflip` — add a horizontally-flipped view (off by default).
- `--multi_scale_image_sizes 1680,1344` — also run SAM 3 at other input sizes
  and average. Most accurate, but each extra size loads another ~0.8 B-param
  SAM 3 backbone (more memory, slower). Sizes must be clean multiples of the
  SAM 3 patch grid (e.g. 1680, 1344).
- `--fuse_full_tile mean|max` — how the full and tile views are combined.

**Post-processing** (per-head; needs `scipy`). Each flag takes a single value
for all heads, or a `head:value,...` spec:

- `--pp_min_cc_prob 0.55` — drop connected components whose mean probability is
  below the threshold. **On by default** at 0.55; set `0` to disable.
- `--pp_min_area_frac 1e-4` — drop components smaller than this fraction of the
  image (off by default).
- `--pp_keep_largest_cc 1` — keep only the largest component (single-target only).
- `--pp_close_iters 1` — morphological closing; smooths borders, but avoid on
  vessels / cells since it merges adjacent objects.

## Parameters

| Argument | Meaning |
|----------|---------|
| `--ckpt_path` | The model directory, e.g. `model`. |
| `--image` + `--prompt` | Single-image mode (mutually exclusive with `--data_root`). |
| `--single_image_head` | Which head's mask to save in single-image mode (`argmax` default / `semantic` / `union`). |
| `--data_root` | Dataset root in BiomedParse layout (folder mode). |
| `--results_root` | Where masks / overlays / metrics are written. |
| `--method` | A short run label; used only in output filenames and log paths. |
| `--heads` | Read-out heads to compute, comma-separated (default `argmax,semantic,union`). |
| `--score_thr` | Per-query presence-score threshold for the `union` / gated `semantic` heads (default 0.5). |
| `--mask_thr` | Per-pixel mask-probability threshold (default 0.5). |
| `--use_full` / `--use_tile` | Full-image / tiled views; both on by default (`--no-...` to disable). |
| `--use_hflip` | Add a horizontally-flipped view (off by default). |
| `--tile_size` / `--tile_overlap` / `--fuse_full_tile` | Tile geometry and full-vs-tile fusion (`mean`/`max`). |
| `--multi_scale_image_sizes` | Comma-separated extra SAM 3 input sizes for multi-scale TTA. |
| `--pp_min_cc_prob` / `--pp_min_area_frac` / `--pp_keep_largest_cc` / `--pp_close_iters` | Per-head mask post-processing (needs `scipy`); `--pp_min_cc_prob` defaults to 0.55. |
| `--save_pred_masks` | Save predicted mask PNGs (folder mode). |
| `--save_vis` | Also save red-overlay visualizations. |
| `--sam3_image_size` | Primary SAM 3 input resolution; default 2016 (this checkpoint's training size — leave as is). |
| `--sam3_model_path` | Optional: a different SAM 3 config dir. Default resolves `model/sam3/`. |
| `--dtype` | `bf16` (default) / `fp16` / `fp32`. |

The SAM 3 topology/processor is loaded from `model/sam3/` and its weights from
`model.safetensors`. Run `--help` for the full option list.

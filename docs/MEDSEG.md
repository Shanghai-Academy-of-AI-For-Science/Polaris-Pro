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
- `opencv-python-headless`, `transformers==5.0.0` (provides `Sam3Model`).
- A GPU with ≥ 24 GB memory for the 2016×2016 SAM 3 input.

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
    --method shenzhen \
    --save_vis
```

This writes the predicted mask to `out/medseg/<image>_mask.png` (and, with
`--save_vis`, an overlay `<image>_overlay.png`). No ground truth or metrics are
needed in this mode.

### Dataset folder (batch + metrics)

For batch evaluation against ground-truth masks, point `--data_root` at a
dataset in **BiomedParse layout** and the script computes Dice per unit:

```bash
python code/scripts/medseg/infer_med_seg_qwen3vl.py \
    --ckpt_path model \
    --data_root <dataset_root> \
    --results_root out/medseg \
    --method shenzhen \
    --save_pred_masks
```

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
`pred_masks/` (with `--save_pred_masks`), optional overlays in `vis/` (with
`--save_vis`), and a metrics file `shenzhen_<unit>_dataset_metrics.json`
containing Dice/IoU scores plus per-instance results.

## Parameters

| Argument | Meaning |
|----------|---------|
| `--ckpt_path` | The model directory, e.g. `model`. |
| `--image` + `--prompt` | Single-image mode (mutually exclusive with `--data_root`). |
| `--data_root` | Dataset root in BiomedParse layout (folder mode). |
| `--results_root` | Where masks / overlays / metrics are written. |
| `--method` | A short run label; used only in output filenames and log paths. |
| `--save_pred_masks` | Save predicted mask PNGs (folder mode). |
| `--save_vis` | Also save red-overlay visualizations. |
| `--mask_threshold` | Probability threshold for the binary mask (default 0.5). |
| `--sam3_image_size` | SAM 3 input resolution; default 2016 (this checkpoint's training size — leave as is). |
| `--sam3_model_path` | Optional: a different SAM 3 config dir. Default resolves `model/sam3/`. |
| `--dtype` | `bf16` (default) / `fp16` / `fp32`. |

The SAM 3 topology/processor is loaded from `model/sam3/` and its weights from
`model.safetensors`. Run `--help` for tiling and other advanced options.

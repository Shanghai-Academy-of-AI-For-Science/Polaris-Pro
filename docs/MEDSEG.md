# Medical-image segmentation

Polaris-Pro performs **text-prompted medical-image segmentation**: given an
image and a text description of a target region, the backbone conditions a
**SAM 3** decoder to output the mask. This modality has its own script.

## ⚠️ SAM 3 is required and NOT included

The decoder relies on Meta's **Segment Anything Model 3 (SAM 3)**, distributed
under Meta's **gated, non-commercial SAM license**. It is **not** bundled here
and nothing in this repo grants any rights to it. Download it from Meta (accept
their license) and point the script at it. Your use of SAM 3 is governed solely
by Meta's license.

## Requirements
- `opencv-python-headless`, `transformers==5.0.0` (provides `Sam3Model`).
- A downloaded SAM 3 directory.

## Run
```bash
export PYTHONPATH=$PWD/code

python code/scripts/medseg/infer_med_seg_qwen3vl.py \
    --ckpt_path model \
    --sam3_model_path <PATH_TO_SAM3> \
    --data_root <dataset_root> \
    --results_root out/medseg \
    --method polaris_pro \
    --save_pred_masks
```

SAM 3 is resolved from `<ckpt_path>/sam3/` if present, else `--sam3_model_path`.
Run `--help` for tiling / threshold / visualization options.

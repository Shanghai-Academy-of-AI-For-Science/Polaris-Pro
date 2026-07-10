# Medical-image segmentation

Polaris-Pro performs **text-prompted medical-image segmentation**: given an
image and a text description of a target region, the backbone conditions a
**SAM 3** decoder to output the mask. This modality has its own script.

## ⚠️ SAM 3 weights are embedded — under Meta's SAM License

The decoder is a **fine-tuned SAM 3** backbone whose weights are embedded in the
released `model.safetensors`; its topology and processor config ship in
`model/sam3/`. These SAM 3 weights are governed by **Meta's SAM License**
(`SAM_LICENSE.txt`), not Apache-2.0, and their use is subject to that license's
acceptable-use restrictions (no military / weapons / illegal uses; Trade-Control
compliance). Everything else in Polaris-Pro is Apache-2.0.

## Requirements
- `opencv-python-headless`, `transformers==5.0.0` (provides `Sam3Model`).
- A GPU with ≥ 24 GB memory for the 2016×2016 SAM 3 input.

## Run
```bash
export PYTHONPATH=$PWD/code

python code/scripts/medseg/infer_med_seg_qwen3vl.py \
    --ckpt_path model \
    --data_root <dataset_root> \
    --results_root out/medseg \
    --method polaris_pro \
    --save_pred_masks
```

The SAM 3 topology/processor is loaded from `model/sam3/` and its weights from
`model.safetensors`. To use a different SAM 3 config, pass `--sam3_model_path`.
Run `--help` for tiling, threshold, and visualization options.

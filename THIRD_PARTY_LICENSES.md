# Third-Party Licenses & Attributions

Polaris-Pro itself is released under the **Apache License 2.0** (see `LICENSE`).
It builds on, vendors, or links to the third-party works below. Each retains
its own license; where model weights are merged into `model.safetensors`, the
corresponding upstream license still governs that portion of the weights.

---

## 1. Qwen3-VL — base multimodal backbone

- **Upstream:** Qwen3-VL-8B-Instruct, Alibaba Cloud / Qwen team.
- **License:** Apache License 2.0.
- **Where:** the LLM + vision tower (`model.language_model.*`, `visual.*`) and
  the `qwen3_vl` modeling classes this repo extends.
- **Link:** https://github.com/QwenLM/Qwen3-VL , https://huggingface.co/Qwen

## 2. ESM-2 — protein sequence encoder

- **Upstream:** ESM-2 (`esm2_t30_150M_UR50D`), Meta AI / `facebookresearch/esm`.
- **License:** MIT.
- **Where:** the protein encoder backbone. Weights are **merged** into
  `model.safetensors` under `model.modality_router.encoders.protein.*`. The
  tokenizer is re-implemented (`qwenvl/modalities/protein/esm2_tokenizer.py`).
- **Link:** https://github.com/facebookresearch/esm

## 3. Swin Transformer — used inside the weather encoder

- **Upstream:** Swin Transformer, Microsoft.
- **License:** MIT.
- **Where:** window-attention / patch-merging blocks used by the Polaris-derived
  weather encoder (`qwenvl/modalities/weather/internal/polaris_swin.py`,
  `polaris_attention.py`).
- **Link:** https://github.com/microsoft/Swin-Transformer

## 4. Polaris — weather encoder/decoder architecture

- **Upstream:** Polaris meteorological forecasting model (research codebase).
- **Where:** the weather modality encoder, decoder, and Swin/RoPE support
  modules under `qwenvl/modalities/weather/` are adapted (vendored + trimmed)
  from the Polaris codebase. Weights are merged into `model.safetensors` under
  `model.modality_router.encoders.weather.*` / `decoders.weather.*`.
- **License:** released here under Apache-2.0 with the authors' permission.
  If you redistribute the weather component separately, retain this attribution.

## 5. Suiren — molecular GNN encoder

- **Upstream:** "Suiren" molecular graph pretraining (research codebase).
- **Where:** the molecule GNN (`qwenvl/modalities/mol/graph_NN.py`,
  `org_mol2d.py`). The pretrained GNN weights are merged into
  `model.safetensors` under `model.modality_router.encoders.mol.*`.
- **License:** released here under Apache-2.0 with the authors' permission.
- **Depends on:** RDKit (BSD-3-Clause) and PyTorch-Geometric (MIT).

## 6. SAM 3 — medical-image segmentation backbone (NOT redistributed)

- **Upstream:** Segment Anything Model 3 (SAM 3), Meta.
- **License:** **Meta's SAM license — gated, non-commercial, click-through.**
- **Status:** **NOT included** in this repository or on the model hub. The
  medical-segmentation code (`qwenvl/modalities/med_seg/`) can attach a SAM 3
  model at runtime, but you must obtain SAM 3 yourself from Meta and accept
  their license. See `docs/MEDSEG.md`.
- **Link:** https://huggingface.co/facebook (SAM 3, gated)

---

## Python dependencies

Standard PyPI packages (see `requirements.txt`) retain their own licenses,
e.g. PyTorch (BSD-3-Clause), transformers/tokenizers/accelerate/safetensors
(Apache-2.0), NumPy/SciPy (BSD), RDKit (BSD-3-Clause), PyTorch-Geometric (MIT),
xarray/netCDF4 (BSD/Apache), OpenCV (Apache-2.0).

If you believe any attribution here is incomplete or incorrect, please open an
issue so it can be fixed.

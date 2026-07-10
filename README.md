# Polaris-Pro

<div align="center">

[🤗 Model](https://huggingface.co/sais-org/Polaris_Pro) &nbsp;•&nbsp; [💻 GitHub](https://github.com/Shanghai-Academy-of-AI-For-Science/Polaris-Pro) &nbsp;•&nbsp; [📜 Technical Report (coming soon)](#) &nbsp;•&nbsp; [⚖️ License: Apache-2.0](./LICENSE)

</div>

**Polaris-Pro is a unified scientific multimodal foundation model** that
supports scientific **understanding and generation** across Earth science,
proteins, RNA, DNA, and small molecules within a single **8B** model.

## Introduction

Polaris-Pro pairs **domain-specific encoders and decoders with a shared
Qwen3-VL-8B backbone**, so heterogeneous scientific data — biological sequences,
molecular graphs, gridded physical fields, and medical images — are understood
*and* generated within one model through a natural-language interface. Each
modality keeps its native encoder (ESM-2 for protein, ConvFormers for RNA/DNA, a
molecular graph encoder, a Swin-ViT weather tower, a SAM-based image path) and
decodes back into its native form: a class, a number, a designed sequence, a
SMILES string, a global forecast, or a segmentation mask.

> 📜 **Technical report coming soon.**

## Key features

- **Unified understanding *and* generation.** One model classifies, regresses,
  designs sequences, generates molecules, forecasts weather, and segments
  medical images — through a single natural-language interface.
- **Seven modalities, one 8B backbone.** Protein / RNA / DNA / molecule /
  weather / medical-image / text share the Qwen3-VL-8B Transformer via a
  modality router; no per-task model swapping.
- **Native scientific encoders/decoders.** Domain-specific modules preserve
  sequence motifs, molecular graphs, dense physical fields, and image structure
  that a generic text tokenizer would destroy — the source of its edge over
  text-token scientific LLMs at the same scale.
- **Efficient 8B scale.** Delivers strong scientific results — including
  end-to-end weather forecasting and medical-image segmentation — from a single
  8B model.

## Capabilities

| Modality      | Understanding | Generation |
|:--------------|:-------------:|:----------:|
| Protein       | ✅            | —          |
| RNA           | ✅            | ✅         |
| DNA           | ✅            | —          |
| Molecule      | ✅            | ✅         |
| Weather       | —             | ✅         |
| Medical image | —             | ✅         |
| Text          | ✅            | ✅         |

<sub>**Understanding** = classification / regression / scientific QA. **Generation**: RNA sequence design · Molecule text → SMILES · Weather 10-day global ERA5 0.25° forecast · Medical-image text-prompted segmentation (needs SAM 3).</sub>

## Benchmarks

**Polaris-Pro** (**8B**) vs **Biology-Instructions** (Llama-3.1-**8B**, text-token,
no scientific encoders) and **Intern-S1-Pro** (**~1T** MoE scientific model).
**Bold** = best; <ins>underline</ins> = second-best.

### Biological sequence understanding

| Task | Metric | Polaris-Pro (8B) | Biology-Instructions (8B) | Intern-S1-Pro (~1T) |
|:-----|:------:|:----------------:|:-------------------------:|:-------------------:|
| DNA · Epigenetic marks (EMP) | MCC | **71.99** | 3.64 | <ins>14.02</ins> |
| DNA · Promoter det. 300bp (PD300) | MCC | **91.17** | 58.18 | <ins>82.65</ins> |
| DNA · Core-promoter (CPD) | MCC | **66.35** | 44.54 | <ins>54.60</ins> |
| DNA · Enhancer activity (EA) | PCC | 52.64 | <ins>53.28</ins> | **55.16** |
| RNA · ncRNA function | Acc | **91.46** | <ins>63.09</ins> | 34.50 |
| RNA · Modification | AUC | **96.03** | <ins>59.06</ins> | 57.77 |
| RNA · APA isoform | R² | <ins>79.87</ins> | 59.01 | **82.95** |
| RNA · CRISPR on-target | Spearman ρ | **28.76** | -0.02 | <ins>15.69</ins> |
| Protein · Stability | Spearman ρ | **70.63** | 60.25 | <ins>60.82</ins> |
| Protein · Fluorescence | Spearman ρ | <ins>70.12</ins> | 2.57 | **78.14** |
| Protein · Enzyme Commission | Fmax | <ins>68.65</ins> | 19.79 | **72.70** |
| Protein · Solubility | Acc | <ins>67.26</ins> | 63.02 | **67.60** |
| Cross-modal · RPI (RNA–protein) | MCC | **76.49** | <ins>74.26</ins> | 58.51 |
| Cross-modal · AAN (antibody–antigen) | MCC | <ins>42.96</ins> | 1.06 | **44.76** |
| Cross-modal · EPI (enhancer–promoter) | MCC | <ins>-0.03</ins> | **3.37** | -1.30 |

<sub>Aggregate over 20 biological-understanding benchmarks: Polaris-Pro matches or beats the ~1T Intern-S1-Pro on 10/20 and the same-scale 8B text-token baseline on 16/20.</sub>

### Molecule understanding (SMolInstruct)

| Task | Metric | Polaris-Pro (8B) | LlaSMol |
|:-----|:------:|:----------------:|:-------:|
| BBBP | Acc | **96.95** | 74.60 |
| HIV | Acc | **97.00** | 96.70 |
| SIDER | Acc | **71.00** | 70.70 |
| ClinTox | Acc | 92.36 | **93.10** |
| ESOL | RMSE ↓ | **0.550** | 1.150 |
| Lipophilicity | RMSE ↓ | **0.628** | 1.010 |

### Earth-science forecasting — vs ECMWF HRES (day-10, global ERA5 0.25°)

| Variable | Metric | Polaris-Pro (8B) | ECMWF HRES (NWP) |
|:---------|:------:|:----------------:|:----------------:|
| Z500 | RMSE ↓ | **≈740** | ≈810 |
| T2M | RMSE ↓ (K) | **≈2.65** | ≈2.90 |
| MSL | RMSE ↓ (Pa) | **≈680** | ≈745 |

<sub>Polaris-Pro tracks or beats the operational physics-based HRES system, with the advantage growing at longer lead times.</sub>

### Medical-image segmentation

Mean Dice (%) on the BiomedParse test splits, 102,855 image–prompt pairs across
nine imaging modalities, versus six modality-native segmentation specialists.

| Modality | # Samples | Polaris-Pro | BiomedParse | MedSAM | SAM | SAM3 | DINO+MedSAM | DINO+SAM |
|:---------|----------:|:-----------:|:-----------:|:------:|:---:|:----:|:-----------:|:--------:|
| **All**    | 102,855 | **91.20** | <ins>90.73</ins> | 83.55 | 71.29 | 35.40 | 15.37 | 15.10 |
| CT         |  45,306 | **93.36** | <ins>92.25</ins> | 83.87 | 74.10 | 28.93 |  9.59 | 10.34 |
| MRI        |  30,990 | **85.29** | <ins>85.25</ins> | 75.90 | 68.34 | 53.64 | 13.28 | 12.39 |
| OCT        |     283 | <ins>85.31</ins> | **86.63** | 56.26 | 55.99 |  8.69 |  6.68 |  6.98 |
| X-ray      |  13,840 | <ins>98.02</ins> | **98.28** | 97.75 | 81.35 | 39.96 | 37.22 | 30.63 |
| Dermoscopy |      65 | **98.08** | 97.11 | <ins>97.35</ins> | 88.23 | 51.47 | 81.28 | 78.29 |
| Endoscopy  |     410 | **97.39** | 96.77 | <ins>97.05</ins> | 92.88 | 38.82 | 25.01 | 24.54 |
| Fundus     |     800 | <ins>91.33</ins> | **91.50** | 88.06 | 57.16 | 18.58 |  3.19 |  2.73 |
| Pathology  |     977 | **87.29** | <ins>81.57</ins> | 43.44 | 42.06 | 26.08 | 25.38 | 24.69 |
| Ultrasound |  10,184 | <ins>90.54</ins> | **91.03** | 89.76 | 57.47 |  5.23 | 17.12 | 22.91 |

<sub>Best overall Dice (All), and best on CT, MRI, pathology, dermoscopy, and endoscopy. On X-ray, Fundus, and Ultrasound the gap to BiomedParse is ≤ 0.5 Dice; on the smallest split (OCT, 283 samples) it is 1.3. Polaris-Pro reuses the SAM3 image branch as its dense-feature source, lifting off-the-shelf SAM3 (35.40) to 91.20.</sub>

## Setup

Python 3.10, an NVIDIA GPU (≥ 48 GB recommended), CUDA 12.x.

```bash
conda create -n polaris python=3.10 -y && conda activate polaris
pip install torch==2.6.0 torchvision==0.21.0     # match your host CUDA
pip install -r requirements.txt
```

`transformers==5.0.0` is a hard pin. `flash-attn` is **not** installed by
default (it must be compiled against your torch/CUDA); without it the model
automatically falls back to eager attention — identical outputs, just slower.
To enable it: `pip install flash-attn==2.7.4.post1 --no-build-isolation`.

## Download weights

```bash
hf download sais-org/Polaris_Pro --local-dir ./model
```

All scientific encoder/decoder weights (ESM-2, the Suiren molecular graph
encoder, the RNA/DNA ConvFormers, and the Swin-ViT weather tower) are embedded
in `model.safetensors` — nothing else to fetch. The **only** separate download
is **SAM 3** for medical-image segmentation, which ships under Meta's gated
license (see [docs/MEDSEG.md](docs/MEDSEG.md)).

## Quick start

```bash
export PYTHONPATH=$PWD/code

bash run_examples.sh                 # one example per task type
GPU=1 bash run_examples.sh mol_gen   # a single example on a chosen GPU
```

## Usage

Sequences are passed via `--rna/--dna/--protein/--mol` and referenced in the
prompt with placeholders `<rna>/<dna>/<protein>/<mol>`. Each task also has a
`--system` prompt that fixes the output format — use it to match the benchmark
setting:

```bash
export PYTHONPATH=$PWD/code
python code/inference.py --model_path model --greedy --max_new_tokens 64 \
  --rna "GGATGCGATCATGTCTGCACTAACACACCGGATCCCATCAGAACTCCGAAGTTAAGCGTGCTTGGGCGGGAGTAGTACTAGGATGGGCGACCCCTTAGGAAGTACTCGTGTTGCATCCC" \
  --system "You are a non-coding RNA family classifier. Output only the family name, no other text." \
  --prompt $'<rna>\nWhich family does this non-coding RNA sequence belong to?'

# molecule generation (text -> SMILES; note the task flag, no <mol> input)
python code/inference.py --model_path model --task mol_generation --max_new_tokens 128 \
  --system "You are a molecule generation expert. Output only the SMILES string, with no additional text." \
  --prompt $'Generate a molecule that matches the following description:\n...\nOutput only the canonical SMILES string.'
```

- `--greedy` for classification/regression (deterministic).
- `--system` sets the per-task output-format instruction; `run_examples.sh` has the one for every task.
- `--task mol_generation` for molecule generation; `--task generation` for RNA design.
- Batch mode: `--input_file samples.jsonl --output_file out.jsonl`, one JSON record per line
  with `conversations` (system/human turns) and the sequence field. See `run_examples.sh`.

Weather forecasting and medical-image segmentation use dedicated scripts —
see [docs/WEATHER.md](docs/WEATHER.md) and [docs/MEDSEG.md](docs/MEDSEG.md)
(the latter needs SAM 3, downloaded separately from Meta).

## License

Apache-2.0 (`LICENSE`). Third-party components and the separately-downloaded,
gated SAM 3 are documented in `THIRD_PARTY_LICENSES.md` and `NOTICE`.

## Citation

A technical report is coming soon. For now, please cite:

```bibtex
@misc{polarispro2026,
  title  = {Polaris-Pro: A Unified Scientific Multimodal Foundation Model},
  author = {Hesen Chen and Xinyu Su and Xiaomeng Yang and Yuetan Lin and Zixiong Yang and Zhiyu Tan and Hao Li},
  year   = {2026},
  note   = {https://huggingface.co/sais-org/Polaris_Pro}
}
```

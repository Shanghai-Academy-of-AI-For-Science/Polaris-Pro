#!/usr/bin/env bash
# ============================================================================
# Polaris-Pro — minimal inference launcher with one example per task type.
#
# Usage:
#   bash run_examples.sh                # run ALL examples below
#   bash run_examples.sh rna_cls        # run a single example by name
#   GPU=2 bash run_examples.sh mol_gen  # pick a GPU
#
# Example names: rna_cls  rna_reg  rna_gen  dna_cls  dna_reg
#                protein_cls  protein_reg  protein_ec  mol_cls  mol_reg  mol_gen
#                text   (plain scientific-text QA, no bio input)
#
# Weather forecasting and medical-image segmentation use dedicated entry
# scripts (see docs/WEATHER.md and docs/MEDSEG.md); they are not part of this
# text-style CLI because their I/O is gridded netCDF / image+mask.
# ============================================================================
set -euo pipefail

# ---- Resolve package root (this script's directory) ----
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$PKG_ROOT/model}"
PY="${PY:-python}"                       # set PY=/path/to/conda/envs/xxx/bin/python if needed
GPU="${GPU:-0}"
MAX_NEW="${MAX_NEW:-64}"

export PYTHONPATH="$PKG_ROOT/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
# flash-attn is used if installed; otherwise the model auto-falls back to eager.

INFER="$PKG_ROOT/code/inference.py"

# Note: medical-image segmentation (med_seg) needs the SAM3 backbone, which is
# NOT bundled (Meta gated license). Download it yourself and pass it to the
# med_seg script with --sam3_model_path; see docs/MEDSEG.md. The examples below
# do not touch med_seg.

run() {  # name, then inference.py args
  local name="$1"; shift
  echo ""
  echo "================ [$name] ================"
  $PY "$INFER" --model_path "$MODEL" --greedy --max_new_tokens "$MAX_NEW" "$@"
}

WANT="${1:-all}"
do_run() { [ "$WANT" = "all" ] || [ "$WANT" = "$1" ]; }

# Each task ships a specific system prompt that constrains the output format;
# passing it (via --system) reproduces the benchmark setting. The sequences and
# prompts below are taken from the evaluation test sets.

# ---------------------------------------------------------------------------
# RNA
# ---------------------------------------------------------------------------
# (1) RNA classification — non-coding RNA family
do_run rna_cls && run rna_cls \
  --rna "GGATGCGATCATGTCTGCACTAACACACCGGATCCCATCAGAACTCCGAAGTTAAGCGTGCTTGGGCGGGAGTAGTACTAGGATGGGCGACCCCTTAGGAAGTACTCGTGTTGCATCCC" \
  --system "You are a non-coding RNA family classifier. Output only the family name, no other text." \
  --prompt $'<rna>\nWhich family does this non-coding RNA sequence belong to?'

# (2) RNA regression — mean ribosome loading (translation efficiency)
do_run rna_reg && run rna_reg \
  --rna "CCCCCCAAGCAACACGCGCGGGCCTATCGGCAGCACCATGGCCGCATATACCGCATATA" \
  --prompt $'<rna>\nWhat is the expected translation efficiency associated with the sequence?'

# (3) RNA generation — toehold switch design (trigger + linker -> switch)
do_run rna_gen && run rna_gen --task generation --max_new_tokens 128 \
  --rna "CUAAAUUAACAAUAGUAGUAAUUUUUUUUU" "AACCUGGCGGCAGCGCAAAAGAUGCG" \
  --system "You are a specialized RNA design model. Output the generated sequence tag directly." \
  --prompt $'<rna>\n<rna>\nThe first RNA is the trigger and the second is the linker. Design a high-performance toehold switch sequence.'

# ---------------------------------------------------------------------------
# DNA
# ---------------------------------------------------------------------------
# (4) DNA classification — promoter detection, 300 bp (Yes/No)
do_run dna_cls && run dna_cls \
  --dna "GCAATAAAAGGCTTAGCCACATAGTGCATGCATGTACACAGCATGTACAC" \
  --system "You are a DNA sequence analysis expert. Read the DNA sequence(s) and the question carefully. Respond with a single token: exactly 'Yes' or 'No'. Do not add any explanation, punctuation, reasoning, or additional text." \
  --prompt $'<dna>\nIs this 300 bp DNA sequence a promoter region (all promoters, TATA and non-TATA combined)? Answer Yes or No.'

# (5) DNA regression — enhancer activity (float)
do_run dna_reg && run dna_reg \
  --dna "AACATACCCTGCTCTAGCGTATTGCTTTTTGGCAGCTACGTAGCTAGCTAGCTTTTCGTTTGG" \
  --system "You are a DNA sequence analysis expert. Read the DNA sequence and the question carefully. Respond with a single floating-point number only. Do not add units, explanations, reasoning, or any additional text." \
  --prompt $'<dna>\nPredict the quantile-normalized developmental enhancer (Dev) log2 enrichment activity score of this DNA sequence. Answer with a float number.'

# ---------------------------------------------------------------------------
# Protein
# ---------------------------------------------------------------------------
# (6) Protein classification — solubility (0/1)
do_run protein_cls && run protein_cls \
  --protein "MLSVRIAAAVARALPRRAGLVSKNALGSSFIAARNFHASNTHLQKTGTAEMSSILEERILGADTSVDLEETGRVLSIGDGIARVHGLRNVQAEEMVEFSSGLKGMSLNLEP" \
  --system "You are a protein solubility predictor. This is a binary classification task. Output only one digit: 1 for soluble, 0 for insoluble. Do not output any other text." \
  --prompt $'<protein>\nSolubility prediction involves forecasting if a protein can dissolve. What is the solubility status of this protein? Output only one digit: 1 for soluble, 0 for insoluble.'

# (7) Protein regression — stability score
do_run protein_reg && run protein_reg \
  --protein "TTIKVNGQEYTVPLSPEQAAKAAKKRWPDYEVQIHGNTVKVTR" \
  --system "You are a protein stability predictor. Output only the stability score as a number, no other text." \
  --prompt $'<protein>\nHow is the stability of this protein sequence calculated?'

# (8) Protein multi-label — Enzyme Commission (EC) numbers
do_run protein_ec && run protein_ec \
  --protein "MHHHHHHSSGVDLGTENLYFQSNAMDFPQQLEACVKQANQALSRFIAPLPFQNTPVVETMQYGALLGGKRLRPFLVYATGHMFGVSTNTLDAPAAAVELIHAYSLIHDDLPAMDDDDLRRGLPTCHVKFGEANAILAGDALQTLAFSILSDADLADYIIQRNK" \
  --system "You are a protein function predictor. Output only the EC number(s), comma-separated, no other text." \
  --prompt $'<protein>\nPredict the Enzyme Commission (EC) number(s) of this protein. Output only the EC numbers, comma-separated.'

# ---------------------------------------------------------------------------
# Molecule
# ---------------------------------------------------------------------------
# (9) Mol classification — ADMET (Ames mutagenicity, 0/1)
do_run mol_cls && run mol_cls \
  --mol "CC(=O)Nc1ccc2c(=O)c(=O)c3cccc4ccc1c2c43" \
  --system "You are a molecular property prediction expert; given a molecule's SMILES string and an ADMET endpoint description, respond with only 0 or 1 to indicate whether the molecule possesses that property." \
  --prompt $'<mol>\nGiven the SMILES representation of a molecule, predict whether it is mutagenic (1) or non-mutagenic (0) based on the Ames test.'

# (10) Mol regression — physicochemical property (dipole moment)
do_run mol_reg && run mol_reg \
  --mol "CC(=O)Nc1ccccc1" \
  --system "You are a molecular property prediction expert. Based on the input molecular representations and instructions, answer with the specific molecular property values." \
  --prompt $'<mol>\nWhat is the dipole moment value of this molecular?'

# (11) Mol generation — text description -> SMILES (no <mol> input)
do_run mol_gen && run mol_gen --task mol_generation --max_new_tokens 128 \
  --system "You are a molecule generation expert. Given a natural-language molecular description, generate one molecule as a valid canonical SMILES string. Output only the SMILES string, with no additional text." \
  --prompt $'Generate a molecule that matches the following description:\nThe molecule is a long-chain fatty acid that is henicosane in which one of the methyl groups has been oxidised to give the corresponding carboxylic acid. It is a straight-chain saturated fatty acid and a long-chain fatty acid. It is a conjugate acid of a henicosanoate.\nOutput only the canonical SMILES string.'

# ---------------------------------------------------------------------------
# Scientific text (no bio input) — plain multiple-choice / QA
# ---------------------------------------------------------------------------
do_run text && run text --max_new_tokens 256 \
  --prompt $'The following is a multiple choice question about biology. Think step by step and then finish your answer with "the answer is (X)".\nQuestion:\nWhich molecule carries amino acids to the ribosome during translation?\nOptions:\nA. mRNA\nB. tRNA\nC. rRNA\nD. snRNA\nAnswer:'

echo ""
echo "================ done ================"

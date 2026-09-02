# Hyperstition-Awareness

# Durability of hyperstition-imparted model alignment — code

Pipeline for conditions A–E′ of the meta-awareness study. Everything runs on Google Colab
(A100 for evaluation, fine-tuning and probes; no GPU for corpus generation or analysis).

## Pipeline

| Step | Notebook | What it does | Writes to Drive |
|---|---|---|---|
| 1 | `01_generate_corpora.ipynb` | Generates the three 6,000-pair Q&A corpora (D, E, E′) with GPT-5 Mini via `scripts/generate_corpus.py` | `experiment/synthetic_data/{meta_awareness,benign_control,ai_control}_corpus.jsonl` + `*_stats.json` |
| 2 | `02_finetune_lora.ipynb` | Rank-32 RSLoRA fine-tune of the Tice et al. DPO model on each corpus; identical hyperparameters; seed 42 | `experiment/adapters/condition_{d_meta,e_benign,eprime_ai_control}/` + `hparams.json` |
| 3 | `03_evaluate.ipynb` | 8-variant log-prob evaluation on the 4,174-question suite for A, B, C (plain model) and D, E, E′ (adapter loaded, fresh runtime each) | `meta_awareness_v3/results_{cond}_{split}.json` |
| 4 | `04_analysis_prereg.ipynb` | Pre-registered paired analysis: bootstrap CI, McNemar on hard flips, paired t as sensitivity check | `meta_awareness_v3/prereg_analysis_all_comparisons.json` |
| 5 | `05_freetext_probe.ipynb` | Twelve-question free-generation probe on base, D, E′ (fresh runtime each) | `meta_awareness_v3/freetext_probes/probe_{cond}.json` |

`scripts/finetune_lora.py` is the same code as the training cell in notebook 02, kept as a file for
reference. `scripts/generate_corpus.py` must be uploaded to `/content/` before running notebook 01.

## Conditions

| | Route | Content |
|---|---|---|
| A | — | HHH system prompt only |
| B | in-context | HHH + 182-word hyperstitional meta-awareness passage |
| C | in-context | HHH + 182-word matched benign passage |
| D | LoRA | meta-awareness corpus |
| E | LoRA | benign-science corpus |
| E′ | LoRA | AI-training/alignment corpus, engineering framing, no hyperstition |

## Fixed settings

Model `geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo`; fp16 on A100;
evaluation seed 56, fine-tuning seed 42, bootstrap seed 0; 4 phrasings × 2 orderings;
answer read as log-prob of `A` vs `B` after `Answer: (` inside the assistant turn.

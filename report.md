# Small-model ticket triage — assessment report

## Data audit and method
Training-only audit: 1445 raw rows, 1360 retained, 85 dropped, 175 fixed. Finding counts overlap: {"priority_rule_recomputed": 60, "legacy_priority_key_and_value": 70, "legacy_date_to_iso": 30, "invalid_json_label": 25, "normalize_transaction_id": 20, "rupees_stored_as_paise": 25, "exact_duplicate_input": 60}. The row-level log records every changed/dropped row. JSON/schema checks, duplicate grouping, decimal currency parsing excluding balances/limits/fees and quoted text, and deterministic priority rules produce these findings. Decimal rupees stored as truncated whole rupees are recovered from an unambiguous explicit amount. No LLM generated labels. Category/language labels and semantic dates beyond format conversion are not exhaustively relabelled; that remains an audit limitation.

## Training and controlled comparisons
Base: Qwen/Qwen2.5-0.5B-Instruct; revision recorded in final_config.json. Transformers/PEFT with an explicit PyTorch loop keeps masking and exposure visible. The original system/user messages use the native chat template. Loss includes only assistant tokens, including end-of-message, with prompt/pad tokens masked. Full examples are retained without truncation; measured lengths are in length_statistics.json. LoRA targets attention and MLP projections. Seed 42, dropout 0.05, AdamW, 2e-4 peak LR with warmup/cosine schedule; 100 updates × 16 sampled rows, identical exposure budget across runs. Mixed precision is FP16; QLoRA uses NF4/double quantization with FP16 computation. Alpha/rank is fixed at 2. E samples existing rows within 10% of ₹5,000/₹50,000 boundaries at 3× weight, with the same total sample count. All comparisons have one declared intervention; single-seed results do not establish statistical significance.

| Run | Exact match | Mean field | Peak GB | Train min |
|---|---:|---:|---:|---:|
| baseline_fixed | 0.0% | 0.0% | — | — |
| baseline_rules | 0.0% | 2.3% | — | — |
| A_raw | 52.5% | 90.5% | 4.95 | 8.07 |
| B_clean | 53.5% | 91.5% | 5.03 | 8.32 |
| C_fp16 | 55.0% | 91.2% | 4.93 | 6.01 |
| D_rank32 | 67.0% | 94.2% | 5.16 | 8.45 |
| E_boundary | 52.0% | 90.7% | 5.03 | 8.40 |

A/B isolates cleaning, B/C precision, B/D rank, B/E sampling. Baselines load the untuned instruction checkpoint in FP16; rules are added only in the reference baseline. Peak GB is allocated CUDA memory during training, not total device usage; reserved memory and actual optimizer updates are also logged. Chosen run: **D_rank32**, using dev exact match, then mean field accuracy, then lower rank. Test never influences selection.

## Error analysis
Weakest fields: priority 72.5%, needs_human 89.0%, category 96.5%. 66 dev rows fail exact match. Full per-field/category/slice scores and per-row diagnostic hypotheses are in results/. The table shows actual failures, not invented examples.

| ID | Failed fields | Expected → actual (first failed field) |
|---|---|---|
| dv-00005 | priority | priority: 'P2' → 'P3' |
| dv-00006 | txn_date | txn_date: '2026-06-30' → '2026-06-28' |
| dv-00011 | priority, needs_human | priority: 'P1' → 'P3' |
| dv-00012 | priority, needs_human | priority: 'P1' → 'P3' |
| dv-00016 | priority | priority: 'P2' → 'P3' |
| dv-00017 | priority, needs_human | priority: 'P1' → 'P3' |
| dv-00019 | priority, needs_human | priority: 'P1' → 'P3' |
| dv-00021 | priority | priority: 'P2' → 'P3' |
| dv-00023 | priority | priority: 'P2' → 'P3' |
| dv-00028 | priority | priority: 'P2' → 'P3' |

Amount/date errors suggest checking copying, normalization and relative-date reasoning; priority/escalation errors may propagate from those upstream errors. These are hypotheses, not causal proof. The detailed error file gives an explanation for each listed case. With one more week: inspect ambiguous training semantics, add deterministic schema-derived training augmentation, repeat key contrasts with additional seeds, and test a larger model within the same hardware constraints. Avoid repeatedly tuning on the same small dev set without reporting that limitation.

## Reproduction, limitations and AI disclosure
Inference uses untouched system/user inputs, greedy model.generate, max_new_tokens=200, native chat formatting and whitespace stripping only. No grammar constraints, output repair or teacher-model labels. Reload matching, environment, timing and configuration are saved. Notebook reproduction must be checked on a fresh T4 before claiming the three-hour condition is met. No production deployment or business savings are claimed.

OpenAI Codex assisted with code drafting, deterministic audit checks, experiment design, debugging, research and report drafting. Numbers come from executed code and the supplied scorer. The candidate must review and understand the implementation before submission and live defense.

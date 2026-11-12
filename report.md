# Small-model ticket triage

## Data audit

I audited the training split only. It had 1,445 rows; I retained 1,360, dropped 85, and fixed 175. Finding counts overlap. `cleaning_log.csv` records each changed or dropped row.

The deterministic checks covered malformed labels, legacy priority fields, date and transaction-ID formats, rupees recorded as paise, duplicate inputs, and priority rules. I repaired an amount only when the ticket text made the unit correction clear; I dropped ambiguous amount conflicts rather than guessing. I left category and language labels alone, did not build a full semantic date parser, and used an incomplete repeated-contact pattern check. Those are limits of this audit.

| Finding | Rows |
|---|---:|
| Priority rule recomputed | 60 |
| Legacy priority key/value | 70 |
| Legacy date format | 30 |
| Invalid JSON label | 25 |
| Transaction ID normalized | 20 |
| Rupees/paise unit corrected | 25 |
| Exact duplicate input removed | 60 |

These counts overlap; they do not sum to the number of changed rows. I did not use dev or test labels to clean training data, and I did not use a second model to create labels.

## Training and results

I used `Qwen/Qwen2.5-0.5B-Instruct` with Transformers and PEFT. The training loop uses the model’s chat template and computes loss on assistant tokens only; prompt and padding tokens are masked. Examples are not truncated. Fine-tuning used seed 42, 100 update steps, an effective batch of 16, AdamW, and a peak learning rate of 2e-4. QLoRA runs use NF4 with double quantization and FP16 compute. The FP16 run uses an unquantized base. Each comparison changes one setting; results are from one seed, so they are not evidence of statistical significance.

| Run | Change | Dev exact match | Mean field accuracy | Peak allocated GB | Train min |
|---|---|---:|---:|---:|---:|
| Base, fixed prompt | No fine-tuning | 0.0% | 0.0% | — | — |
| Base, rules prompt | Reference baseline | 0.0% | 2.31% | — | — |
| A_raw | Raw data, 4-bit, rank 16 | 52.5% | 90.50% | 4.95 | 8.07 |
| B_clean | Clean data, 4-bit, rank 16 | 53.5% | 91.50% | 5.03 | 8.32 |
| C_fp16 | FP16 LoRA, rank 16 | 55.0% | 91.25% | 4.93 | 6.01 |
| D_rank32 | Rank 32 | **67.0%** | **94.25%** | 5.16 | 8.45 |
| E_boundary | Upweight boundary examples | 52.0% | 90.69% | 5.03 | 8.40 |

A/B compares cleaning, B/C compares precision, B/D compares rank, and B/E compares sampling. For E, I upweighted existing examples within 10% of the ₹5,000 and ₹50,000 thresholds at 3×, keeping the sample count fixed. I selected D using dev exact match, then mean field accuracy, then lower rank. Test data did not affect model selection.

## Error analysis

The weakest fields were priority (72.5%), `needs_human` (89.0%), and category (96.5%). There were 66 exact-match failures. In the table, paired values are `priority/needs_human`. These ten examples show the main patterns:

| ID | Expected → predicted | What went wrong |
|---|---|---|
| dv-00005 | P2 → P3 | The ₹49,680 amount was extracted correctly, but priority was too low near the ₹50,000 rule. |
| dv-00006 | 2026-06-30 → 2026-06-28 | “Day before yesterday” was resolved two days too early. |
| dv-00011 | P1/true → P3/false | The ₹84,137 urgent refund was not escalated. |
| dv-00012 | P1/true → P3/false | The ₹237,415 refund amount was right; priority and escalation were missed. |
| dv-00016 | P2 → P3 | The failed ₹25,647 UPI top-up was under-prioritized. |
| dv-00017 | P1/true → P3/false | The delayed ₹180,827.25 refund was not escalated. |
| dv-00019 | P1/true → P3/false | The ₹259,419 faulty-product refund was not escalated. |
| dv-00021 | P2 → P3 | The ₹15,775 refund was under-prioritized. |
| dv-00023 | P2 → P3 | The ₹23,752.75 failed payment in Hinglish was under-prioritized. |
| dv-00028 | P2 → P3 | The ₹30,274 failed transfer was under-prioritized. |

In most of these rows, amount extraction was correct but the priority or escalation output was not. That points to rule application as a likely weakness; it does not prove the cause. The date example is a separate relative-date error. With another week, I would inspect more threshold and escalation cases, add only schema-grounded training examples, repeat key comparisons with more seeds, and then try a larger model if the T4 budget allows.

## Reproduction and limits

The recorded selected-run reproduction took 63.13 minutes inside the runner on a Colab T4, including training, a 16-row adapter reload check, and generation of all 400 test predictions. Notebook setup and package installation are not included in that timing. Inference uses the original system and user messages, greedy decoding, `max_new_tokens=200`, and whitespace stripping only. The adapter is not a production deployment, and I have not measured business savings.

**AI assistance:** I used OpenAI Codex to draft and revise parts of the code, help build deterministic audit checks, debug the runs, and edit this report. It did not create labels or test predictions; the fine-tuned model generated the test outputs. I am responsible for the submitted work and for explaining the code and results.

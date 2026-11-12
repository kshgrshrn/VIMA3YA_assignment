# Small-model ticket triage

I fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` with 4-bit QLoRA for the VIMA3YA take-home. My best run was `D_rank32`: **67.0% exact match** and **94.25% mean field accuracy** on the 200-row dev set.

This repository contains the adapter, code, notebook, evaluation results, cleaning log, and test predictions. It does not contain the supplied ticket dataset or base-model weights. This is an assessment project, not a production system.

## Reproduce

Open `notebook.ipynb` in a free Colab T4 runtime and upload the original `training_assessment.zip`. The notebook downloads the runner files and the saved selection metadata from this repository. Its default path runs a small smoke check, retrains the selected recipe, reloads the adapter, and writes predictions for all 400 test inputs. To rerun the full experiment matrix, set `RUN_FULL_EXPERIMENTS = True` in its optional cell; the saved metrics and per-run logs are in `results/`.

The recorded selected-run reproduction took 63.13 minutes on a T4. That timing starts inside the runner and does not include notebook setup or package installation. The notebook and model downloads require an internet connection.

## Attribution

Some ticket text is adapted from [PolyAI Banking77](https://github.com/PolyAI-LDN/task-specific-datasets), as described in the assignment brief. The task, categories, and labels were supplied for this assessment. The base model is licensed under Apache-2.0; see its [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). The supplied dataset is not included here.

See [report.md](report.md) for the audit, experiments, error analysis, limitations, and tool disclosure.

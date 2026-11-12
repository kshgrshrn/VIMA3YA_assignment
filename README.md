# NimbusPay small-model triage

Base model: `Qwen/Qwen2.5-0.5B-Instruct`. Best run: `D_rank32`. Loaded in 4-bit: **True**.

Measured dev exact match: **67.0%**; mean field accuracy: **94.2%**.

An independent hiring-assessment project, not production work or employment. See report.md for AI-assistance disclosure and limitations.

## Reproduce
Open notebook.ipynb in a fresh free Colab T4. Upload the employer's original training_assessment.zip when prompted. The notebook embeds the code, uses the saved final configuration, retrains the selected adapter and produces test predictions. No supplied training dataset or base model is included here. Full experiment code is retained; enable experiment mode only to rerun the entire matrix.

The exact model revision, adapter configuration, generation settings, library versions, loss logs, audit and scorer outputs are retained. The provided scorer is unchanged. Adapters use standard PEFT safetensors. Keep this repository private until the agreed deadline; make no commits after the deadline.

## Attribution
Some supplied ticket text is adapted from PolyAI Banking77 (CC-BY-4.0), per the assignment brief: https://github.com/PolyAI-LDN/task-specific-datasets . Task, categories and labels are supplied by VIMA3YA. No rights to redistribute the full supplied dataset are asserted. Model license: Apache-2.0, see its upstream model card.

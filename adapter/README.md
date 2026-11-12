# NimbusPay ticket-triage adapter

This PEFT LoRA adapter was trained for the VIMA3YA take-home assignment.

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter:** rank-32 LoRA trained with a 4-bit NF4 base (QLoRA)
- **Best dev result:** 67.0% exact match and 94.25% mean field accuracy on 200 rows
- **Intended use:** assessment and experimentation only; this has not been deployed
- **Known weakness:** priority and `needs_human` are less reliable than the other fields

The base model is Apache-2.0 licensed. The supplied assessment data is not included in this repository. For the training setup and evaluation details, see the top-level `README.md` and `report.md`.

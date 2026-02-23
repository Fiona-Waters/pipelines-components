# Fine-Tuning Components

> ⚠️ **DEPRECATED** — The unified `train_model` component in this directory is deprecated and will be removed in the next release.
>
> Use specialized components instead:
>
> - `components.training.osft` for OSFT training
> - `components.training.sft` for SFT training

> **Deprecated** — This unified training component is deprecated and will be removed in the next release.
> Use the specialized components instead: [OSFT](../osft/), [SFT](../sft/), or [LoRA](../lora/).

## Overview 🧾

- **[LoRA](lora/)** - Low-Rank Adaptation fine-tuning with Unsloth backend
- **[OSFT](../osft/)** - Orthogonal Subspace Fine-Tuning with mini-trainer backend
- **[SFT](../sft/)** - Supervised Fine-Tuning with instructlab-training backend

All components share common utilities from `../shared/` for dataset handling, model persistence, and metrics extraction.

## Adding New Components

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `str` |  |

## Metadata 🗂️

- **Name**: finetuning
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: HuggingFace Datasets, Version: >=2.14.0
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - fine_tuning
  - finetuning
  - osft
  - sft
  - llm
  - language_model
  - deprecated
- **Last Verified**: 2026-01-09 00:00:00+00:00
- **Owners**:
  - Approvers:
    - briangallagher
    - Fiona-Waters
    - kramaranya
    - MStokluska
    - szaher

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)

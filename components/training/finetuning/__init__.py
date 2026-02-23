"""Finetuning Components Subcategory.

This subcategory contains specialized fine-tuning components:
- osft: Orthogonal Subspace Fine-Tuning
- sft: Supervised Fine-Tuning

The unified train_model component is deprecated. Use specialized components instead.
"""

from .component import train_model  # noqa: F401 (deprecated, kept for backwards compatibility)

__all__ = ["train_model"]

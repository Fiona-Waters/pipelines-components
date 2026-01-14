"""OSFT Pipeline - Placeholder.

A full version of the OSFT pipeline with additional stages.
This is a placeholder for future implementation.
"""

from kfp import dsl

# Import reusable components
from components.data_processing.dataset_download import dataset_download
from components.deployment.model_registry import model_registry
from components.evaluation.lm_eval import universal_llm_evaluator
from components.training.finetuning import train_model

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "10Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "osft-pipeline"
# =============================================================================


@dsl.pipeline(
    name="osft-pipeline",
    description="OSFT Pipeline - Placeholder for future implementation",
)
def osft_pipeline():
    """OSFT pipeline placeholder.

    A full version of the OSFT pipeline with additional stages.
    To be implemented in a follow-up PR.
    """
    placeholder_task()

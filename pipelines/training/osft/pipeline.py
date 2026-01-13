"""OSFT Pipeline - Placeholder.

A full version of the OSFT pipeline with additional stages.
This is a placeholder for future implementation.
"""

from kfp import dsl


# Import pipeline-specific (non-reusable) components
from pipelines.training.osft.components.dataset_download import dataset_download
from pipelines.training.osft.components.eval import universal_llm_evaluator
from pipelines.training.osft.components.model_registry import model_registry

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

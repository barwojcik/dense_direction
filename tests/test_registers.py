import pytest
from mmengine import DATASETS, FUNCTIONS, TRANSFORMS
from mmseg.registry import METRICS, MODELS, VISUALIZERS

import dense_direction  # noqa: F401

registry_map = {
    "DATASETS": DATASETS,
    "FUNCTIONS": FUNCTIONS,
    "TRANSFORMS": TRANSFORMS,
    "METRICS": METRICS,
    "MODELS": MODELS,
    "VISUALIZERS": VISUALIZERS,
}


@pytest.mark.parametrize(
    "registry_name, name",
    [
        # Models
        # - backbones
        ("MODELS", "Dino2TorchHub"),
        ("MODELS", "Dino3TorchHub"),
        # - blocks
        ("MODELS", "DPTDecoderBlock"),
        # - heads
        ("MODELS", "DPTDirectionHead"),
        ("MODELS", "DualDecodeHead"),
        ("MODELS", "DummyDirectionHead"),
        ("MODELS", "LinearDirectionHead"),
        ("MODELS", "LinearHead"),
        ("MODELS", "MultiscaleLossDirectionHead"),
        # - losses
        ("MODELS", "DirectionalLoss"),
        ("MODELS", "EfficientDirectionalLoss"),
        ("MODELS", "SmoothnessLoss"),
        # - meta-architectures
        ("MODELS", "Directioner"),
        ("MODELS", "SegmentoDirectioner"),
        # Metrics
        ("METRICS", "CenterlineDirectionMetric"),
        ("METRICS", "DirectionalLossMetric"),
        ("METRICS", "DumpSamples"),
        # Datasets
        ("DATASETS", "ConcreteCracksDataset"),
        ("DATASETS", "OttawaRoadsDataset"),
        # Transforms
        ("TRANSFORMS", "BinarizeAnnotations"),
        ("TRANSFORMS", "CenterlineToDirections"),
        ("TRANSFORMS", "PackCustomInputs"),
        ("TRANSFORMS", "MaskGuidedRandomCrop"),
        # Visualizers
        ("VISUALIZERS", "SegDirLocalVisualizer"),
        # Functions
        ("FUNCTIONS", "circular_point_kernel"),
        ("FUNCTIONS", "radial_line_kernel"),
        ("FUNCTIONS", "polar_kernel"),
        ("FUNCTIONS", "polar_wedge_kernel"),
        ("FUNCTIONS", "polar_disc_kernel"),
        ("FUNCTIONS", "polar_sector_kernel"),
    ],
)
def test_registry_registration(registry_name, name):
    assert name in registry_map[registry_name]

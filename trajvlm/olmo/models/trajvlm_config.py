"""Shared TrajVLM schemas and idempotent checkpoint type registration."""
import dataclasses as _dc
import sys
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.nn.siglip2_trajgroup_vision_backbone import (
    SigLip2TrajGroupVisionBackboneConfig,
)
from olmo.preprocessing.trajvit_preprocessor import TrajVitImageConfig, TrajVitVideoConfig
from olmo.models import model_config as _model_config


@_dc.dataclass
class TrajVlmPreprocessorConfig(Molmo2PreprocessorConfig):
    image: TrajVitImageConfig = _dc.field(default_factory=TrajVitImageConfig)
    video: TrajVitVideoConfig = _dc.field(default_factory=TrajVitVideoConfig)


@_dc.dataclass
class TrajVlmConfig(Molmo2Config):
    vision_backbone: SigLip2TrajGroupVisionBackboneConfig = _dc.field(
        default_factory=SigLip2TrajGroupVisionBackboneConfig,
    )
    mm_preprocessor: TrajVlmPreprocessorConfig = _dc.field(default_factory=TrajVlmPreprocessorConfig)


# Keep the same _model_name so checkpoints saved/loaded as 'video_olmo' resolve to us.
TrajVlmConfig._model_name = Molmo2Config._model_name  # "video_olmo"


def _patched_get_model_types():
    from olmo.models.molmo.molmo import MolmoConfig
    from olmo.models.molmo_point.molmo_point import MolmoPointConfig
    return {
        MolmoConfig._model_name: MolmoConfig,
        TrajVlmConfig._model_name: TrajVlmConfig,       # overrides Molmo2Config
        MolmoPointConfig._model_name: MolmoPointConfig,
        "token_indexing_video_molmo2": MolmoPointConfig,
    }


_model_config.get_model_types = _patched_get_model_types


# trainer_config may already be imported by an embedding caller.
if "olmo.train.trainer_config" in sys.modules:
    sys.modules["olmo.train.trainer_config"].get_model_types = _patched_get_model_types

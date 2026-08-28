import os
import sys

# The repository keeps the DiffSynth plugin next to ComfyUI. Add it lazily so
# installing this custom-node folder does not require a separate package build.
try:
    import diffsynth  # Prefer the package installed with `pip install diffsynth`.
except ImportError:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _diffsynth_root = os.path.join(_repo_root, "DiffSynth-Studio")
    if os.path.isdir(_diffsynth_root) and _diffsynth_root not in sys.path:
        sys.path.insert(0, _diffsynth_root)
    try:
        import diffsynth
    except ImportError as exc:
        raise ImportError(
            "DiffSynthStudioNodes requires DiffSynth-Studio. Install it with `pip install diffsynth`."
        ) from exc

from .nodes import (VRAMConfigNode, QuantizationConfigNode, ModelConfigNode, VRAMLimitNode,
                    MergeModelConfigsNode, PipelineOthersBuilderNode, PipelineLoaderNode,
                    LoRAClearNode, LoRALoadNode, generate_inference_nodes)
from .pipeline_registry import PIPELINE_REGISTRY

NODE_CLASS_MAPPINGS = {
    "DiffSynthVRAMConfig": VRAMConfigNode,
    "DiffSynthQuantizationConfig": QuantizationConfigNode,
    "DiffSynthModelConfig": ModelConfigNode,
    "DiffSynthVRAMLimit": VRAMLimitNode,
    "DiffSynthMergeModelConfigs": MergeModelConfigsNode,
    "DiffSynthPipelineOthersBuilder": PipelineOthersBuilderNode,
    "DiffSynthPipelineLoader": PipelineLoaderNode,
    "DiffSynthLoRAClear": LoRAClearNode,
    "DiffSynthLoRALoad": LoRALoadNode,
}
NODE_CLASS_MAPPINGS.update(generate_inference_nodes())

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffSynthVRAMConfig": "DiffSynth: VRAM Config",
    "DiffSynthQuantizationConfig": "DiffSynth: Quantization Config",
    "DiffSynthModelConfig": "DiffSynth: ModelConfig",
    "DiffSynthVRAMLimit": "DiffSynth: VRAM Limit",
    "DiffSynthMergeModelConfigs": "DiffSynth: Merge ModelConfigs",
    "DiffSynthPipelineOthersBuilder": "DiffSynth: Pipeline Others Builder",
    "DiffSynthPipelineLoader": "DiffSynth: Pipeline Loader",
    "DiffSynthLoRAClear": "DiffSynth: LoRA Clear",
    "DiffSynthLoRALoad": "DiffSynth: LoRA Load",
}
for type_name, meta in PIPELINE_REGISTRY.items():
    NODE_DISPLAY_NAME_MAPPINGS[f"DiffSynth{type_name}Inference"] = f"DiffSynth: {meta.display_name} Inference"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

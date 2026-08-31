import os
import sys
import diffsynth


from .nodes import (VRAMConfigNode, QuantizationConfigNode, MixedQuantizeConfigNode,
                    ModelConfigNode, VRAMLimitNode, MergeModelConfigsNode,
                    PipelineOthersBuilderNode, PipelineLoaderNode,
                    LoRAClearNode, LoRALoadNode, generate_inference_nodes)
from .pipeline_registry import PIPELINE_REGISTRY

NODE_CLASS_MAPPINGS = {
    "DiffSynthVRAMConfig": VRAMConfigNode,
    "DiffSynthQuantizationConfig": QuantizationConfigNode,
    "DiffSynthMixedQuantizeConfig": MixedQuantizeConfigNode,
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
    "DiffSynthMixedQuantizeConfig": "DiffSynth: Mixed Quantize Config",
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

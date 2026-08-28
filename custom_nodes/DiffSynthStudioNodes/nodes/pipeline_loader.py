import hashlib
import json
import torch
from ..pipeline_registry import get_pipeline_class, get_pipeline_type_names, get_from_pretrained_config_params
from ..type_defs import MODEL_CONFIG_LIST, PIPE, ANY


def _dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


class PipelineLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline_type": (get_pipeline_type_names(),),
            "torch_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            "device": (["cuda", "cpu"], {"default": "cuda"}),
            "model_configs": (MODEL_CONFIG_LIST,),
            "vram_limit": ("FLOAT", {"default": 0.0}),
        }, "optional": {"others": (ANY,)}}

    RETURN_TYPES = (PIPE,)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/loader"

    @staticmethod
    def IS_CHANGED(**kwargs):
        """Return a stable cache key without hashing mutable ModelConfig objects."""
        def normalize(value):
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
            fields = ("model_id", "origin_file_pattern", "path", "download_source",
                      "local_model_path", "clear_parameters", "quantize")
            return {field: normalize(getattr(value, field, None)) for field in fields}
        payload = normalize(kwargs)
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def execute(self, pipeline_type, torch_dtype, device, model_configs, vram_limit, others=None):
        cls = get_pipeline_class(pipeline_type)
        kwargs = {"torch_dtype": _dtype(torch_dtype), "device": device,
                  "model_configs": model_configs, "vram_limit": vram_limit}
        config_names = get_from_pretrained_config_params(pipeline_type)
        if isinstance(others, dict):
            kwargs.update(others)
        elif others is not None:
            values = others if isinstance(others, list) else [others]
            for name, value in zip(config_names, values):
                kwargs[name] = value
        pipe = cls.from_pretrained(**kwargs)
        # Useful for diagnostics and downstream extensions; LoRA loading does
        # not depend on this marker and auto-detects the target module.
        pipe._diffsynth_pipeline_type = pipeline_type
        return (pipe,)

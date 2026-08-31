import inspect
from ..pipeline_registry import get_pipeline_type_names, get_pipeline_class, get_from_pretrained_config_params
from ..type_defs import MODEL_CONFIG, ANY


class PipelineOthersBuilderNode:
    @classmethod
    def INPUT_TYPES(cls):
        options = {f"config_{i}": (MODEL_CONFIG,) for i in range(1, 5)}
        return {"required": {"pipeline_type": (get_pipeline_type_names(),)},
                "optional": options}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("others",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, pipeline_type, **kwargs):
        names = get_from_pretrained_config_params(pipeline_type)
        result = {}
        for index, name in enumerate(names, 1):
            value = kwargs.get(f"config_{index}")
            if value is not None:
                result[name] = value
        signature = inspect.signature(get_pipeline_class(pipeline_type).from_pretrained)
        common = {"cls", "torch_dtype", "device", "model_configs", "vram_limit"}
        for name, parameter in signature.parameters.items():
            if name in common or name.endswith("_config") or name in result:
                continue
            if name in kwargs and kwargs[name] is not None:
                result[name] = kwargs[name]
        return (result,)

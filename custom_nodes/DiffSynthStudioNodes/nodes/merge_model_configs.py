from ..type_defs import MODEL_CONFIG, MODEL_CONFIG_LIST


class MergeModelConfigsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_config_1": (MODEL_CONFIG,)},
                "optional": {f"model_config_{i}": (MODEL_CONFIG,) for i in range(2, 7)}}

    RETURN_TYPES = (MODEL_CONFIG_LIST,)
    RETURN_NAMES = ("model_configs",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, model_config_1, **kwargs):
        return ([model_config_1] + [kwargs[name] for name in sorted(kwargs) if kwargs[name] is not None],)

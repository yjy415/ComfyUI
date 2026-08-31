from ..type_defs import MODEL_CONFIG, PIPE


class LoRAClearNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": (PIPE,)}}
    RETURN_TYPES = (PIPE,)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/LoRA"
    def execute(self, pipe):
        pipe.clear_lora()
        return (pipe,)


class LoRALoadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": (PIPE,), "lora_config": (MODEL_CONFIG,)},
                "optional": {"alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1})}}
    RETURN_TYPES = (PIPE,)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/LoRA"
    def execute(self, pipe, lora_config, alpha=1.0):
        module = next((getattr(pipe, name) for name in ("dit", "unet", "video_dit")
                       if getattr(pipe, name, None) is not None), None)
        if module is None:
            raise ValueError("Pipeline has no supported LoRA target module")
        pipe.load_lora(module, lora_config, alpha=alpha)
        return (pipe,)

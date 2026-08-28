from ..type_defs import QUANT_CONFIG

METHODS = [
    "bitsandbytes_nf4", "bitsandbytes_fp4", "torchao_int8_w8a16", "torchao_int4_w4a16",
    "torchao_fp8_w8a16", "torchao_int8_w8a8", "torchao_fp8_w8a8", "torchao_int4_w4a8",
    "torchao_mxfp8_w8a8", "torchao_mxfp4_w4a4", "torchao_nvfp4_w4a4", "torchao_nvfp4_w4a16",
    "comfy_kitchen_int8_w8a8", "comfy_kitchen_fp8_w8a8",
]


def _modules(value):
    if value is None:
        return None
    result = [part.strip() for part in str(value).replace("\n", ",").split(",") if part.strip()]
    return result or None


class QuantizationConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "method": (METHODS, {"default": METHODS[0]}),
            "mode": (["dynamic", "dequant_once"], {"default": "dynamic"}),
            "enable_mixed": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "target_modules": ("STRING", {"default": "", "multiline": True}),
            "exclude_modules": ("STRING", {"default": "", "multiline": True}),
            "load_prequantized": ("BOOLEAN", {"default": False}),
            "method_2": (METHODS, {"default": METHODS[2]}),
            "target_modules_2": ("STRING", {"default": "", "multiline": True}),
            "exclude_modules_2": ("STRING", {"default": "", "multiline": True}),
        }}

    RETURN_TYPES = (QUANT_CONFIG,)
    RETURN_NAMES = ("quant_config",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, method, mode, enable_mixed=False, target_modules="", exclude_modules="",
                load_prequantized=False, method_2=METHODS[2], target_modules_2="", exclude_modules_2=""):
        from diffsynth.core.quant import QuantizeConfig, MixedQuantizeConfig
        first = QuantizeConfig(method=method, mode=mode, target_modules=_modules(target_modules),
                               exclude_modules=_modules(exclude_modules),
                               load_prequantized=False if enable_mixed else load_prequantized)
        if enable_mixed:
            second = QuantizeConfig(method=method_2, mode=mode, target_modules=_modules(target_modules_2),
                                    exclude_modules=_modules(exclude_modules_2), load_prequantized=False)
            return (MixedQuantizeConfig(configs=[first, second], load_prequantized=load_prequantized),)
        return (first,)

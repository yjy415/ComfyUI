import torch


class VRAMLimitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "device": ("STRING", {"default": "cuda"}),
            "buffer_size": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                        "tooltip": "Reserved VRAM in GB. Use 8 GB or more for MiniMax pipelines."}),
        }}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("vram_limit",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, device="cuda", buffer_size=4.0):
        try:
            if not torch.cuda.is_available():
                return (float("inf"),)
            total = torch.cuda.mem_get_info(device)[1] / (1024 ** 3)
            return (max(total - float(buffer_size), 0.0),)
        except Exception:
            return (float("inf"),)

VRAM_CONFIG = "DIFFSYNTH_VRAM_CONFIG"
QUANT_CONFIG = "DIFFSYNTH_QUANT_CONFIG"
MODEL_CONFIG = "DIFFSYNTH_MODEL_CONFIG"
MODEL_CONFIG_LIST = "DIFFSYNTH_MODEL_CONFIG_LIST"
PIPE = "DIFFSYNTH_PIPE"


class AnyType(str):
    """A ComfyUI wildcard type that accepts any connected value."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")

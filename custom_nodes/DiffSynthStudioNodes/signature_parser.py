import inspect
import types
from typing import Literal, Union, get_args, get_origin

try:
    from PIL import Image
except Exception:
    Image = None

_EXCLUDED = {"self", "progress_bar_cmd", "tqdm", "positive_only_lora", "negative_lora", "lora"}
_OVERRIDES = {
    ("ZImage", "num_inference_steps"): {"default": 8, "max": 50},
    ("ZImage", "cfg_scale"): {"default": 1.0},
    ("Flux", "embedded_guidance"): {"default": 3.5},
    ("StableDiffusion", "num_inference_steps"): {"default": 50},
    ("AceStep", "duration"): {"default": 60, "max": 300},
    ("MiniMaxH3", "num_frames"): {"default": 124},
    ("MiniMaxMusic3", "max_audio_duration"): {"default": 60.0, "max": 300.0},
}


def _is_image(annotation):
    if Image is not None and annotation is Image.Image:
        return True
    origin = get_origin(annotation)
    return origin in (list, tuple) and any(_is_image(arg) for arg in get_args(annotation))


def _input_for(name, annotation, default):
    if get_origin(annotation) is Literal:
        values = list(get_args(annotation))
        return (values, {"default": default if default is not inspect.Parameter.empty else values[0]})
    args = get_args(annotation)
    if get_origin(annotation) in (Union, types.UnionType):
        if str in args:
            annotation = str
        elif args:
            annotation = args[0]
    if annotation is str or annotation is inspect.Parameter.empty:
        value = "" if default is inspect.Parameter.empty else default
        if name == "rand_device":
            value = "cuda"
        return ("STRING", {"default": value, "multiline": True})
    if annotation is bool:
        return ("BOOLEAN", {"default": False if default is inspect.Parameter.empty else default})
    if annotation is int:
        options = {"default": 0 if default is inspect.Parameter.empty else default, "min": 0, "max": 2**32 - 1, "step": 1}
        if name in ("width", "height"):
            options.update({"min": 64, "max": 16384, "step": 64})
        elif name == "num_frames":
            options.update({"min": 1, "max": 1000})
        elif name == "num_inference_steps":
            options.update({"min": 1, "max": 200})
        return ("INT", options)
    if annotation is float:
        options = {"default": 0.0 if default is inspect.Parameter.empty else default, "min": 0.0, "max": 1000.0, "step": 0.1}
        if name in {"cfg_scale", "alpha", "strength", "scale", "denoising_strength"}:
            options["max"] = 20.0
        return ("FLOAT", options)
    if _is_image(annotation):
        return ("IMAGE", {})
    return ("*", {})


def parse_call_signature(pipeline_class, pipeline_type):
    required, optional = {}, {}
    for name, parameter in inspect.signature(pipeline_class.__call__).parameters.items():
        if name in _EXCLUDED or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        spec = _input_for(name, parameter.annotation, parameter.default)
        spec_type, options = spec
        options.update(_OVERRIDES.get((pipeline_type, name), {}))
        if parameter.default is None or _is_image(parameter.annotation) or spec_type == "*":
            optional[name] = spec
        else:
            required[name] = spec
    return required, optional

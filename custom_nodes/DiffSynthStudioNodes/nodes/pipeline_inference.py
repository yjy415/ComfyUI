import inspect
import numpy as np
import torch
from PIL import Image
from ..pipeline_registry import PIPELINE_REGISTRY, get_pipeline_class
from ..signature_parser import parse_call_signature
from ..type_defs import PIPE


def _to_pil(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 4:
            value = value[0]
        return Image.fromarray((value.clamp(0, 1).numpy() * 255).astype(np.uint8))
    return value


def _to_image(value):
    if isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Image.Image):
        arrays = [np.asarray(item.convert("RGB"), dtype=np.float32) / 255.0 for item in value]
        return torch.from_numpy(np.stack(arrays))
    if isinstance(value, torch.Tensor):
        return value.float()
    return value


def _to_audio(value, sample_rate=44100):
    if isinstance(value, dict):
        return value
    if isinstance(value, torch.Tensor):
        # ComfyUI's AUDIO contract is waveform [B,C,T] plus integer rate.
        return {"waveform": value, "sample_rate": int(sample_rate)}
    return value


def _convert_output(value, output_type, sample_rate=44100):
    if output_type == "image" or output_type == "video":
        return _to_image(value)
    if output_type == "audio":
        return _to_audio(value, sample_rate)
    if output_type == "audio_video":
        if isinstance(value, tuple):
            return (_to_image(value[0]), _to_audio(value[1], sample_rate))
        if isinstance(value, dict):
            return (_to_image(value.get("video", value.get("images"))), _to_audio(value.get("audio"), sample_rate))
    return value


def generate_inference_nodes():
    nodes = {}
    for type_name, meta in PIPELINE_REGISTRY.items():
        try:
            required, optional = parse_call_signature(get_pipeline_class(type_name), type_name)
        except Exception:
            required, optional = {}, {}
        required = {"pipe": (PIPE,)} | required

        def execute(self, _meta=meta, **kwargs):
            pipe = kwargs.pop("pipe")
            signature = inspect.signature(pipe.__call__)
            for name, value in list(kwargs.items()):
                if value is None:
                    kwargs.pop(name)
                elif name in signature.parameters and _parameter_is_image(signature.parameters[name].annotation):
                    kwargs[name] = _to_pil(value)
            result = pipe(**kwargs)
            sample_rate = getattr(pipe, "audio_sample_rate_output",
                           getattr(pipe, "audio_sample_rate", _meta.audio_sample_rate))
            if _meta.output_type == "audio_video":
                return _convert_output(result, _meta.output_type, sample_rate)
            return (_convert_output(result, _meta.output_type, sample_rate),)

        def is_changed(self, **kwargs):
            return kwargs.get("seed", float("nan"))

        def input_types(cls, _required=required, _optional=optional):
            return {"required": _required, "optional": _optional}

        node_name = f"DiffSynth{type_name}Inference"
        node_class = type(node_name, (), {
            "INPUT_TYPES": classmethod(input_types),
            "RETURN_TYPES": ("IMAGE",) if meta.output_type in ("image", "video") else (("AUDIO",) if meta.output_type == "audio" else ("IMAGE", "AUDIO")),
            "RETURN_NAMES": ("image",) if meta.output_type == "image" else (("video",) if meta.output_type == "video" else (("audio",) if meta.output_type == "audio" else ("video", "audio"))),
            "FUNCTION": "execute", "CATEGORY": "DiffSynth/inference", "OUTPUT_NODE": True,
            "execute": execute, "IS_CHANGED": is_changed,
        })
        nodes[node_name] = node_class
    return nodes


def _parameter_is_image(annotation):
    from ..signature_parser import _is_image
    return _is_image(annotation)

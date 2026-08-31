from dataclasses import dataclass
import importlib
import inspect


@dataclass(frozen=True)
class PipelineMeta:
    type_name: str
    display_name: str
    pipeline_class_path: str
    module_attr: str
    output_type: str
    default_buffer_size: float = 0.5
    audio_sample_rate: int = 44100


PIPELINE_REGISTRY = {
    item.type_name: item for item in [
        PipelineMeta("ZImage", "Z-Image", "diffsynth.pipelines.z_image.ZImagePipeline", "dit", "image"),
        PipelineMeta("Flux", "FLUX", "diffsynth.pipelines.flux_image.FluxImagePipeline", "dit", "image"),
        PipelineMeta("Flux2", "FLUX.2", "diffsynth.pipelines.flux2_image.Flux2ImagePipeline", "dit", "image"),
        PipelineMeta("QwenImage", "Qwen Image", "diffsynth.pipelines.qwen_image.QwenImagePipeline", "dit", "image"),
        PipelineMeta("ErnieImage", "ERNIE Image", "diffsynth.pipelines.ernie_image.ErnieImagePipeline", "dit", "image"),
        PipelineMeta("StableDiffusion", "Stable Diffusion", "diffsynth.pipelines.stable_diffusion.StableDiffusionPipeline", "unet", "image"),
        PipelineMeta("StableDiffusionXL", "Stable Diffusion XL", "diffsynth.pipelines.stable_diffusion_xl.StableDiffusionXLPipeline", "unet", "image"),
        PipelineMeta("AnimaImage", "Anima", "diffsynth.pipelines.anima_image.AnimaImagePipeline", "dit", "image"),
        PipelineMeta("BooguImage", "Boogu Image", "diffsynth.pipelines.boogu_image.BooguImagePipeline", "dit", "image"),
        PipelineMeta("JoyAIImage", "JoyAI Image", "diffsynth.pipelines.joyai_image.JoyAIImagePipeline", "dit", "image"),
        PipelineMeta("Krea2", "Krea 2", "diffsynth.pipelines.krea2.Krea2Pipeline", "dit", "image"),
        PipelineMeta("Ideogram4", "Ideogram 4", "diffsynth.pipelines.ideogram4.Ideogram4Pipeline", "dit", "image"),
        PipelineMeta("HiDreamO1", "HiDream O1", "diffsynth.pipelines.hidream_o1_image.HiDreamO1ImagePipeline", "dit", "image"),
        PipelineMeta("WanVideo", "Wan Video", "diffsynth.pipelines.wan_video.WanVideoPipeline", "video_dit", "video"),
        PipelineMeta("LingBotVideo", "LingBot Video", "diffsynth.pipelines.lingbot_video.LingBotVideoPipeline", "dit", "video"),
        PipelineMeta("LTX2AudioVideo", "LTX-2 Audio Video", "diffsynth.pipelines.ltx2_audio_video.LTX2AudioVideoPipeline", "dit", "audio_video"),
        PipelineMeta("MiniMaxH3", "MiniMax H3", "diffsynth.pipelines.minimax_h3_audio_video.MiniMaxH3Pipeline", "dit", "audio_video", 2.0),
        PipelineMeta("MiniMaxMusic3", "MiniMax Music 3", "diffsynth.pipelines.minimax_music3.MiniMaxMusic3Pipeline", "dit", "audio", 2.0, 32000),
        PipelineMeta("MovaAudioVideo", "MOVA Audio Video", "diffsynth.pipelines.mova_audio_video.MovaAudioVideoPipeline", "video_dit", "audio_video"),
        PipelineMeta("AceStep", "ACE-Step", "diffsynth.pipelines.ace_step.AceStepPipeline", "dit", "audio", 4.0, 48000),
    ]
}


def get_pipeline_class(type_name):
    meta = PIPELINE_REGISTRY[type_name]
    module_name, class_name = meta.pipeline_class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def get_pipeline_type_names():
    return list(PIPELINE_REGISTRY)


def get_from_pretrained_signature(type_name):
    return inspect.signature(get_pipeline_class(type_name).from_pretrained)


def get_from_pretrained_config_params(type_name):
    signature = get_from_pretrained_signature(type_name)
    return [name for name in signature.parameters if name.endswith("_config")]


def get_module_attr(type_name):
    return PIPELINE_REGISTRY[type_name].module_attr

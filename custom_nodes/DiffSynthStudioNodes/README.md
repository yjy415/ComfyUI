# DiffSynth-Studio ComfyUI Nodes

This custom-node package exposes DiffSynth-Studio configuration, model loading,
LoRA, and automatically generated inference nodes in ComfyUI.

## Install

Place this directory at `ComfyUI/custom_nodes/DiffSynthStudioNodes`. The
repository layout used by this checkout already places it beside
`plugin/DiffSynth-Studio`; the node entry point adds that source tree to
`sys.path` at startup.

Install the package in the same environment used by ComfyUI:

```bash
pip install diffsynth
```

Optional quantization/audio backends are installed separately when needed.

## Basic workflow

Use `DiffSynth: VRAM Config`, one `DiffSynth: ModelConfig` node per model file,
and `DiffSynth: Merge ModelConfigs` to feed `DiffSynth: Pipeline Loader`.
Connect a tokenizer or processor ModelConfig directly to `others`, or use
`DiffSynth: Pipeline Others Builder` when several config parameters are needed.
Then connect the matching `DiffSynth: <Pipeline> Inference` node to ComfyUI's
native STRING, Load Image, Preview Image, and Save Image nodes.

LoRAs are chained with `DiffSynth: LoRA Clear` followed by one or more
`DiffSynth: LoRA Load` nodes.

## Z-Image wiring

`VRAM Config` -> `ModelConfig` nodes -> `Merge ModelConfigs` -> `Pipeline
Loader` -> `ZImage Inference` -> native `SaveImage`. Connect the tokenizer
ModelConfig to `Pipeline Others Builder` and its output to `Pipeline Loader.others`.
Use a native Primitive node for `prompt` and `seed`. The same pattern applies
to video and audio pipelines; connect their IMAGE/AUDIO outputs to ComfyUI's
native preview, video, or audio nodes.

## FAQ

**Why does loading take time again?** Loader caching is keyed by model and
loader inputs. Changing prompt or seed does not reload models.

**How much VRAM should MiniMax use?** Set the VRAM buffer to at least 8 GB in
the VRAM Limit node; larger values are appropriate when other GPU workloads run.

**How do I use multiple LoRAs?** Chain `LoRA Clear` and multiple `LoRA Load`
nodes. Each load receives the previous node's pipe output.

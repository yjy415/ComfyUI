# 在 ComfyUI 中使用 DiffSynth-Studio Workflow 推理

本文说明如何理解、导入和运行 `DiffSynthStudioNodes` 提供的默认 workflow，以及如何处理显存、量化、LoRA、图片、视频和音频输入输出。

## 1. 先理解执行链路

插件没有把 DiffSynth 模型拆成 ComfyUI 的 UNet、CLIP、VAE 和 KSampler，而是直接包装 DiffSynth-Studio 的 Pipeline API：

```text
配置节点
  -> ModelConfig 列表
  -> DiffSynth Pipeline.from_pretrained(...)
  -> Pipeline 对象
  -> pipe(**inference_parameters)
  -> IMAGE / AUDIO
  -> ComfyUI 原生保存节点
```

两套系统的职责如下：

| 层 | 职责 |
|---|---|
| ComfyUI | 节点图调度、连线检查、执行缓存、图片和音频输入、结果保存 |
| DiffSynthStudioNodes | 将节点输入转换为 DiffSynth 配置和 Pipeline 调用 |
| DiffSynth-Studio | 模型下载与加载、显存管理、量化、LoRA、实际扩散推理 |

因此，这些 workflow 使用的是 DiffSynth-Studio 的完整 Pipeline，不是 ComfyUI 原生 `KSampler` 路径。不要把 `MODEL`、`CLIP` 或 `LATENT` 类型直接连接到 `DIFFSYNTH_PIPE`。

## 2. 环境准备

### 2.1 确认 Python 环境

必须在启动 ComfyUI 的同一个 Python 环境中安装依赖。使用系统 Python 启动时：

```bash
cd ComfyUI
python -m pip install diffsynth
python -m pip install -r custom_nodes/DiffSynthStudioNodes/requirements.txt
```

使用虚拟环境时，先激活该环境；使用 ComfyUI Portable 时，应调用 Portable 自带的 Python。判断环境是否一致的简单方式是用启动 ComfyUI 的 Python 执行：

```bash
python -c "import diffsynth; print(diffsynth.__file__)"
```

量化后端是可选依赖。只有 workflow 中连接了 `DiffSynthQuantizationConfig` 时，才需要安装所选方法对应的 `bitsandbytes`、`torchao` 或 `comfy-kitchen`。

### 2.2 安装自定义节点

目录应为：

```text
ComfyUI/custom_nodes/DiffSynthStudioNodes/
```

重启 ComfyUI 后，在启动日志中确认没有以下错误：

- `No module named 'diffsynth'`
- `IMPORT FAILED: DiffSynthStudioNodes`
- 缺少量化或音频后端

在节点搜索中输入 `DiffSynth`，应能看到配置、Loader、LoRA 和各 Pipeline 的 Inference 节点。

### 2.3 模型下载条件

默认 workflow 中的 `ModelConfig` 使用 ModelScope 或 Hugging Face 仓库 ID。首次运行会下载模型，需确保：

- ComfyUI 进程能够访问相应模型站点；
- 磁盘空间足够；
- 受限 Hugging Face 模型已完成许可确认和登录；
- 下载缓存目录对 ComfyUI 进程可写。

首次执行可能长时间停留在模型下载或加载阶段，这是正常现象。应观察 ComfyUI 终端日志，而不是反复点击 Queue。

## 3. 选择并导入 Workflow

默认 workflow 位于：

```text
custom_nodes/DiffSynthStudioNodes/examples/
```

文件按 Pipeline 和模型命名，例如：

| 目标 | Workflow |
|---|---|
| Z-Image Turbo 文生图 | `z_image_turbo.json` |
| FLUX.2 dev 文生图 | `flux2_dev.json` |
| SDXL 文生图 | `stable_diffusion_xl_base_1_0.json` |
| Wan 2.1 文生视频 | `wan_video_t2v_1_3b.json` |
| Wan 2.1 图生视频 | `wan_video_i2v_14b_720p.json` |
| ACE-Step 音乐生成 | `ace_step_v15_base.json` |
| LTX-2 文生音视频 | `ltx2_t2av_one_stage.json` |
| MOVA 图生音视频 | `mova_360p_i2av.json` |

导入方式：

1. 打开 ComfyUI 页面。
2. 将目标 JSON 文件拖到画布中，或者使用 ComfyUI 的 `Workflow -> Open`。
3. 等待全部节点显示。
4. 检查是否有红色的未知节点。若有，先解决插件加载问题，不要开始推理。
5. 检查连线是否完整，尤其是 `Pipeline Loader -> Inference -> Save...`。

导入 workflow 本身不会下载或加载模型。只有点击 Queue 后，节点进入执行链才会调用 DiffSynth。

## 4. 阅读默认节点图

### 4.1 VRAM Config

`DiffSynthVRAMConfig` 产生每个 ModelConfig 使用的四阶段设备和 dtype 配置：

| 阶段 | 含义 |
|---|---|
| offload | 模块闲置时保存在哪个设备和 dtype |
| onload | 模块被加载时使用的设备和 dtype |
| preparing | 推理前准备阶段的设备和 dtype |
| computation | 实际计算时的设备和 dtype |

常规低显存配置是 offload/onload 使用 CPU，preparing/computation 使用 CUDA。不要在不了解模型支持情况时把 dtype 改成 `float16`；官方 workflow 中的 BF16/FP32 已按 example 设置。

### 4.2 ModelConfig

每个 `DiffSynthModelConfig` 描述一组模型文件：

- `model_id`：模型仓库 ID；
- `origin_file_pattern`：仓库内文件或 glob；
- `vram_config`：可选显存策略；
- `quant_config`：可选量化策略；
- `path`：可选本地路径；
- `download_source`：ModelScope 或 Hugging Face。

一个 Pipeline 通常包含 transformer/DiT、text encoder、VAE 等多个 ModelConfig。默认 workflow 已按官方 example 填好，第一次运行不应随意删除或调整顺序。

要使用本地模型时，可填写 `path`。本地文件必须和该 ModelConfig 所代表的组件相匹配；不能把整个模型目录无条件填到所有节点。

### 4.3 Merge ModelConfigs

`DiffSynthMergeModelConfigs` 按连接顺序生成 `model_configs` 列表，并交给 Pipeline Loader。顺序来自官方 example。修改模型组件时应保留其原有 slot，避免 Loader 将组件识别错位。

### 4.4 Pipeline Others Builder

tokenizer、processor、audio processor 等 `from_pretrained` 特有参数不属于主 `model_configs` 列表，而是通过 `DiffSynthPipelineOthersBuilder` 构造 `others` 字典。

该节点的 `pipeline_type` 必须与 Pipeline Loader 一致。多个 config 的顺序有意义，例如 SDXL 的 `config_1` 和 `config_2` 分别对应两个 tokenizer。默认 workflow 已完成映射。

### 4.5 VRAM Limit

`DiffSynthVRAMLimit` 查询 GPU 总显存并减去 `buffer_size`，将结果传给 Loader：

```text
vram_limit = GPU total memory - buffer_size
```

常规模型默认预留 4 GB，MiniMax 默认预留 8 GB。这里的 buffer 是给 ComfyUI、CUDA 上下文和其他进程留下的余量，不是模型可以使用的显存。

遇到 OOM 时，优先增大 buffer、启用 offload 或降低分辨率/帧数，而不是把 buffer 改成 0。

### 4.6 Pipeline Loader

`DiffSynthPipelineLoader` 执行：

```python
PipelineClass.from_pretrained(
    torch_dtype=...,
    device=...,
    model_configs=...,
    vram_limit=...,
    **others,
)
```

它是最重的节点。Loader 对输入生成稳定缓存键：只修改 prompt、seed 或推理步数时，ComfyUI 通常复用已加载的 Pipeline；修改模型 ID、dtype、量化配置或 Loader 参数会触发重新加载。

### 4.7 Inference

每种 Pipeline 有独立节点，例如：

- `DiffSynthZImageInference`
- `DiffSynthWanVideoInference`
- `DiffSynthAceStepInference`
- `DiffSynthLTX2AudioVideoInference`

节点参数由 Pipeline 的 `__call__` 签名自动生成。执行时等价于：

```python
result = pipe(prompt=..., seed=..., height=..., width=..., ...)
```

默认 workflow 使用对应官方 example 第一次 `pipe(...)` 调用的显式参数。可以直接修改 Inference 节点内的 prompt、negative prompt、seed、尺寸、帧数、步数和 CFG。

## 5. 执行一次文生图

以 `z_image_turbo.json` 为例：

1. 导入 workflow。
2. 找到 `DiffSynthZImageInference`。
3. 修改 `prompt`；不需要的 `negative_prompt` 可保持空字符串。
4. 确认 `height`、`width` 符合显存能力。
5. 保持 Turbo 的 `num_inference_steps=8` 和 `cfg_scale=1.0` 作为起点。
6. 确认末端连接到 `SaveImage`。
7. 点击 `Queue Prompt`。
8. 在终端观察下载和模型加载进度。
9. 完成后，从 ComfyUI 输出目录或 SaveImage 节点结果中查看图片。

第二次只修改 prompt 或 seed 时，Loader 应被缓存，主要耗时应来自推理本身。

## 6. 执行视频 Workflow

### 6.1 文生视频

例如 `wan_video_t2v_1_3b.json`：

1. 修改 Inference 的 `prompt` 和 `negative_prompt`。
2. 初次测试使用默认 `480x832` 和较少帧数，确认流程可运行后再提高规格。
3. 检查 `seed`、`num_frames`、`num_inference_steps` 和 `tiled`。
4. 运行后，Inference 输出的是 ComfyUI IMAGE batch，每个 batch 元素是一帧。
5. 默认连接到 `SaveAnimatedPNG`。需要 MP4 时，可改接已安装视频扩展的合成节点。

`SaveAnimatedPNG` 的 fps 只影响保存播放速度，不改变 Pipeline 实际生成的帧内容。应按 example 的帧率设置。

### 6.2 图生视频

例如 `wan_video_i2v_14b_720p.json` 或 `mova_360p_i2av.json`：

1. 找到未选择文件的 `LoadImage` 节点。
2. 上传或选择输入图片。
3. 确认它连接到 Inference 的 `input_image`、`edit_image` 或 `input_images`。
4. 修改 prompt。
5. 检查目标尺寸和输入图宽高比。差异过大会发生裁剪或拉伸。
6. Queue 执行。

默认 workflow 不写入虚假的图片文件名，因此首次运行前必须手动选择图片。

对于 `references: list[dict]` 等复杂参数，单个 ComfyUI IMAGE 不能直接代替 DiffSynth 请求结构。此类 socket 会保留为高级输入，并在 workflow 注释中说明；需要能够产生对应 Python 数据结构的节点。

## 7. 执行音频和音视频 Workflow

### 7.1 文生音乐

以 `ace_step_v15_base.json` 或 `minimax_music3.json` 为例：

1. 修改 `prompt`，描述曲风、编曲、速度、情绪和人声。
2. 修改 `lyrics`。无歌词任务按 Pipeline 要求留空或使用结构化歌词。
3. 检查 `duration` 或 `max_audio_duration`，长音频会显著增加耗时和显存。
4. 检查 BPM、调性、拍号、语言等模型特有参数。
5. 运行后，AUDIO 输出连接到 ComfyUI `SaveAudio`。

插件输出 ComfyUI 标准 AUDIO 字典：

```python
{"waveform": tensor, "sample_rate": integer}
```

采样率优先使用 Pipeline 返回或公开的采样率属性，再使用注册表回退值。

### 7.2 同时生成音频和视频

LTX2、MiniMaxH3 和 MOVA 的 Inference 有两个输出：

```text
video -> IMAGE batch -> SaveAnimatedPNG
audio -> AUDIO       -> SaveAudio
```

两个保存节点属于同一次 Pipeline 调用。不要把 video 输出连接到 AUDIO，也不要把 audio 输出连接到图片节点。

对于 A2V/S2V 等需要音频输入的任务：

1. 在 `LoadAudio` 中选择文件；默认 workflow 不预设文件。
2. 确认采样率参数与 example 一致，例如部分 S2V workflow 使用 16000 Hz。
3. 如果 Inference 接受的是 numpy array 或其他 AnyType，而非 ComfyUI AUDIO dict，需要额外的音频解包/转换节点；不要强行连接不兼容类型。

## 8. 修改推理参数的思考顺序

建议按以下顺序调整，便于判断问题来源：

1. 先保持模型和 Loader 配置不变。
2. 只修改 prompt 和 seed，验证基本推理。
3. 调整步数和 CFG，观察质量变化。
4. 调整 height、width、num_frames 或 duration，观察资源占用。
5. 最后再修改显存策略、量化或模型组件。

常见参数影响：

| 参数 | 主要影响 |
|---|---|
| seed | 随机结果；相同环境下用于复现 |
| num_inference_steps | 速度和质量，Turbo 模型不可照搬全模型步数 |
| cfg_scale | 提示词服从度；蒸馏/Turbo 模型通常使用较低值 |
| height/width | 图像尺寸和显存占用 |
| num_frames | 视频长度和显存/耗时 |
| rand_device | 随机噪声生成设备，不等同于模型计算设备 |
| tiled | 分块 VAE 等低显存路径，可能增加耗时 |

## 9. 加入量化

默认 workflow 通常保留 BF16/全精度模型。要按需量化：

1. 添加 `DiffSynthQuantizationConfig`。
2. 选择已安装后端支持的 method。
3. 单量化保持 `enable_mixed=false`。
4. 两种量化混合可设置 `enable_mixed=true`，并填写第二组 method 和模块规则。
5. 三种及以上混合量化时，将前一个 Quantization Config 节点的输出连接到下一个节点的
   `previous_config`；可继续串联任意数量。每组模块规则必须互不重叠，并在最后一个节点设置
   `load_prequantized`。
6. 将输出连接到需要量化的 ModelConfig 的 `quant_config`，通常是 transformer/DiT，而不是 tokenizer。
7. 重新 Queue。因为模型配置改变，Pipeline Loader 会重新加载。

`target_modules` 和 `exclude_modules` 使用逗号或换行分隔。模块集合必须与模型实际名称匹配，混合量化的各组目标不能重叠。

预量化 checkpoint 应启用 `load_prequantized`；普通 BF16 checkpoint 不应仅靠打开该选项假装成预量化模型。

## 10. 加入 LoRA

推荐拓扑：

```text
Pipeline Loader
  -> LoRA Clear
  -> LoRA Load 1
  -> LoRA Load 2
  -> Inference
```

操作步骤：

1. 添加一个 ModelConfig 描述 LoRA 仓库文件或本地 path。
2. 在 Loader 后添加 `DiffSynthLoRAClear`，避免缓存的 Pipeline 残留上次 LoRA。
3. 添加 `DiffSynthLoRALoad`，连接 pipe 和 LoRA ModelConfig。
4. 设置 alpha。
5. 多个 LoRA 按加载顺序串联。
6. 将最后一个 LoRA Load 的 pipe 输出连接到 Inference。

LoRA Load 会自动探测 `dit`、`unet` 或 `video_dit` 目标模块，不需要重复选择 pipeline_type。并非所有模型或 LoRA 格式都能互相兼容。

## 11. 缓存和重新执行

可以用以下规则判断是否会重新加载模型：

| 修改内容 | 通常结果 |
|---|---|
| prompt、negative_prompt、seed | 复用 Pipeline，只重新推理 |
| steps、CFG、尺寸、帧数 | 复用 Pipeline，只重新推理 |
| model_id、file pattern、本地 path | 重新加载 Pipeline |
| dtype、device、vram_limit | 重新加载 Pipeline |
| quant_config、others | 重新加载 Pipeline |

LoRA 会原地修改 Pipeline 对象，因此应使用明确的 `Clear -> Load -> Inference` 依赖链，不能只依赖节点在画布上的视觉位置。

## 12. 常见错误排查

### 12.1 找不到节点

查看 ComfyUI 启动日志。通常是插件目录错误、`diffsynth` 未安装在正确环境、依赖导入失败或未重启 ComfyUI。

### 12.2 模型下载失败

检查 model_id、网络、鉴权、磁盘空间和 download_source。不要把网络错误误判为显存错误。

### 12.3 CUDA OOM

依次尝试：

1. 增大 VRAM Limit 的 buffer；
2. 使用 CPU offload；
3. 降低 height/width；
4. 降低 num_frames 或 duration；
5. 启用 tiled；
6. 对主要 transformer 使用受支持的量化。

修改显存配置后 Loader 需要重新执行。

### 12.4 输出为空或类型错误

检查 Inference 的 output_type：image/video 都以 IMAGE 输出，audio 以 AUDIO 输出，audio_video 有两个输出。检查媒体输入 socket 是否接到了正确参数名。

### 12.5 图生任务仍按文生执行

确认 LoadImage 已选择有效文件、连线连接的是目标 Inference 节点，并且参数名符合该 Pipeline，例如 `edit_image` 和 `input_image` 不能随意互换。

### 12.6 修改 prompt 后模型再次加载

检查是否同时修改了上游 ModelConfig、VRAM Limit、Others Builder 或量化节点。ComfyUI 重启、缓存清理或节点结构变化也会使内存中的 Pipeline 消失。

### 12.7 自定义 scheduler 无法复现

少数官方 Python example 会在加载后直接替换 scheduler，或执行其他节点 API 无法表示的 Python 操作。workflow 的 `extra.ds_notes` 会记录此类限制。此时 workflow 只能复现可由当前节点暴露的部分，不能宣称与脚本逐行等价。

## 13. 推荐验证流程

为新环境或新 workflow 做分阶段验证：

1. 启动 ComfyUI并确认插件注册成功。
2. 导入最小图像 workflow，例如 Z-Image Turbo。
3. 使用较小尺寸完成一次推理。
4. 修改 seed 再运行，确认 Loader 缓存。
5. 再测试目标视频或音频 Pipeline。
6. 最后加入量化、LoRA 和高分辨率参数。

出现问题时记录以下信息，便于定位：

- workflow 文件名；
- ComfyUI 和 `diffsynth` 版本；
- GPU 型号与显存；
- Pipeline Loader 参数；
- 完整终端 traceback；
- 首次加载失败还是推理阶段失败。

按照这条执行链分析，可以把问题快速归类为节点注册、模型配置、下载、Pipeline 加载、推理参数、媒体转换或结果保存，而不需要在整个 ComfyUI 图中盲目排查。

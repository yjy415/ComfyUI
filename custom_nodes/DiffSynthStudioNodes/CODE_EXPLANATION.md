# DiffSynthStudioNodes 代码说明

本文档说明当前 `DiffSynthStudioNodes` 插件的实际实现。插件位于
`plugin/ComfyUI/custom_nodes/DiffSynthStudioNodes/`，作用是把 DiffSynth-Studio
的 Python Pipeline API 适配为 ComfyUI V1 自定义节点。

文档中的“当前代码”指已经写入仓库的实现；方案中提出但尚未实现的能力会在最后列出。

## 1. 总体架构

DiffSynth-Studio 原本通过普通 Python 代码完成推理：

```python
vram_config = {...}
model_config = ModelConfig(...)
pipe = ZImagePipeline.from_pretrained(...)
pipe.load_lora(pipe.dit, lora_config)
image = pipe(prompt="...", seed=42)
```

ComfyUI 的执行模式是“节点输入 -> `execute()` -> 节点输出 -> 下一个节点”。本插件负责三步转换：

```text
ComfyUI 节点参数
    -> DiffSynth 配置对象
    -> Pipeline.from_pretrained() / Pipeline.__call__()
    -> ComfyUI 类型（IMAGE / AUDIO 等）
```

插件不重新实现扩散模型。模型加载、显存管理、量化、LoRA 和采样逻辑仍由 DiffSynth-Studio 负责。

## 2. 文件结构

```text
DiffSynthStudioNodes/
├── __init__.py
├── pipeline_registry.py
├── signature_parser.py
├── type_defs.py
├── README.md
├── CODE_EXPLANATION.md
└── nodes/
    ├── __init__.py
    ├── vram_config.py
    ├── quant_config.py
    ├── model_config.py
    ├── vram_limit.py
    ├── merge_model_configs.py
    ├── pipeline_others_builder.py
    ├── pipeline_loader.py
    ├── lora.py
    └── pipeline_inference.py
```

| 文件 | 职责 |
|---|---|
| `__init__.py` | 设置 DiffSynth import 路径并注册节点 |
| `type_defs.py` | 定义节点端口的自定义类型字符串 |
| `pipeline_registry.py` | 保存 20 个 Pipeline 的元数据 |
| `signature_parser.py` | 将 `__call__` 签名转换成 ComfyUI 输入定义 |
| `nodes/vram_config.py` | 构建 8 字段显存配置 |
| `nodes/quant_config.py` | 构建单量化或混合量化配置 |
| `nodes/model_config.py` | 构建 `ModelConfig` |
| `nodes/vram_limit.py` | 查询显存限制 |
| `nodes/merge_model_configs.py` | 聚合多个模型配置 |
| `nodes/pipeline_others_builder.py` | 构建 Pipeline 专用 config 参数 |
| `nodes/pipeline_loader.py` | 统一调用 `from_pretrained()` |
| `nodes/lora.py` | 清除和加载 LoRA |
| `nodes/pipeline_inference.py` | 动态生成 20 个推理节点 |

## 3. ComfyUI V1 节点协议

节点遵循 `INPUT_TYPES`、`RETURN_TYPES`、`FUNCTION`、`CATEGORY` 这组 V1 API：

```python
class ExampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}, "optional": {...}}

    RETURN_TYPES = (...,)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, ...):
        return (...,)
```

`INPUT_TYPES()` 控制前端控件和端口；`required` 是必需输入；`optional` 是可不连接的输入；
`RETURN_TYPES` 是端口类型；`FUNCTION` 是执行方法名。`execute()` 必须返回 tuple。

入口文件导出 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS`，ComfyUI 启动时读取这两个字典。

## 4. 自定义类型

[`type_defs.py`](./type_defs.py) 定义：

```python
VRAM_CONFIG = "DIFFSYNTH_VRAM_CONFIG"
QUANT_CONFIG = "DIFFSYNTH_QUANT_CONFIG"
MODEL_CONFIG = "DIFFSYNTH_MODEL_CONFIG"
MODEL_CONFIG_LIST = "DIFFSYNTH_MODEL_CONFIG_LIST"
PIPE = "DIFFSYNTH_PIPE"
```

这些是端口兼容性字符串，不是 Python 类名。运行时对象分别是 `dict`、量化配置、`ModelConfig`、
`list[ModelConfig]` 和 Pipeline 实例。

文件中的 `AnyType("*")` 用于 `others` 和复杂高级参数。它通过 `__ne__` 返回 `False` 放宽前端
连线检查，但不会自动转换对象；实际转换仍由节点执行函数完成。

## 5. 插件入口

入口 [`__init__.py`](./__init__.py) 计算当前插件目录到 `plugin/DiffSynth-Studio` 的路径，
将其加入 `sys.path`，使节点可以直接执行 `from diffsynth.core import ModelConfig`。

随后注册 9 个手写节点：VRAM Config、Quantization Config、ModelConfig、VRAM Limit、MergeModelConfigs、
PipelineOthersBuilder、Pipeline Loader、LoRA Clear、LoRA Load。

最后调用 `generate_inference_nodes()`，加入 20 个动态推理节点，当前总数为 29 个。

## 6. Pipeline 注册表

[`pipeline_registry.py`](./pipeline_registry.py) 使用 `PipelineMeta` 保存：

```python
type_name, display_name, pipeline_class_path,
module_attr, output_type, default_buffer_size
```

`type_name` 用于 Combo 和节点 ID；`pipeline_class_path` 用于动态 import；`module_attr` 指 LoRA
目标模块，例如 `dit` 或 `unet`；`output_type` 是 `image`、`video`、`audio` 或 `audio_video`。

`get_pipeline_class()` 用 `importlib.import_module()` 加载类。`get_from_pretrained_config_params()`
通过 `inspect.signature()` 筛选以 `_config` 结尾的参数。例如 ZImage 得到 `tokenizer_config`，
Flux 得到四个 tokenizer/processor config 参数。

## 7. 配置节点

### VRAM Config

[`nodes/vram_config.py`](./nodes/vram_config.py) 提供四组 device/dtype Combo：

```text
offload, onload, preparing, computation
```

设备选项为 `cpu`、`cuda`；dtype 选项为 `float16`、`bfloat16`、`float32`。执行时通过字典将
字符串转成 `torch.dtype`，输出八字段 dict。设备保留字符串，因为 `ModelConfig` 接受这种形式。

### Quantization Config

[`nodes/quant_config.py`](./nodes/quant_config.py) 提供 14 个 DiffSynth 量化方法、`mode`、模块
字符串、`load_prequantized` 和 `enable_mixed`。模块字符串会按逗号和换行切分为空白过滤后的 list。

单量化模式构建 `QuantizeConfig`。混合模式构建两个子 `QuantizeConfig`，再构建
`MixedQuantizeConfig(configs=[first, second])`。代码遵守 DiffSynth 约束：混合模式的子配置共用
一个 mode，`load_prequantized` 只传到外层包装对象。

### ModelConfig

[`nodes/model_config.py`](./nodes/model_config.py) 接收 `model_id`、`origin_file_pattern`，可选
接收 VRAM/量化配置、本地 path、下载源和 `clear_parameters`。它只创建配置对象，不下载模型。
下载发生在 Pipeline Loader 调用 DiffSynth 的 `download_and_load_models()` 时。

### VRAM Limit

[`nodes/vram_limit.py`](./nodes/vram_limit.py) 调用：

```python
total = torch.cuda.mem_get_info(device)[1] / (1024 ** 3)
limit = max(total - buffer_size, 0.0)
```

CUDA 不可用或查询失败时返回 `float("inf")`，表示不设置显存上限。

## 8. 两个辅助节点

### MergeModelConfigs

[`nodes/merge_model_configs.py`](./nodes/merge_model_configs.py) 提供一个必填的
`model_config_1` 和五个可选配置，按顺序组成 `list[ModelConfig]`，输出给 Loader 的 `model_configs`。

### PipelineOthersBuilder

[`nodes/pipeline_others_builder.py`](./nodes/pipeline_others_builder.py) 根据 Pipeline 的
`from_pretrained()` 签名，将 `config_1` 到 `config_4` 映射成对应的 `*_config` 参数名，返回 dict。
例如 Flux 的多个 tokenizer/processor 配置会被构造成：

```python
{"tokenizer_1_config": mc1, "tokenizer_2_config": mc2}
```

## 9. Pipeline Loader

[`nodes/pipeline_loader.py`](./nodes/pipeline_loader.py) 固定接收 `pipeline_type`、`torch_dtype`、
`device`、`model_configs`、`vram_limit` 和 wildcard `others`。

它先构建通用 kwargs：

```python
{
    "torch_dtype": torch.bfloat16,
    "device": device,
    "model_configs": model_configs,
    "vram_limit": vram_limit,
}
```

`others` 是 dict 时直接合并；是 list 或单个对象时，按签名中 config 参数的顺序赋值。最后动态
调用 `pipeline_class.from_pretrained(**kwargs)`。

## 10. LoRA 节点

[`nodes/lora.py`](./nodes/lora.py) 的 Clear 节点调用 `pipe.clear_lora()` 并返回同一个 pipe。

Load 节点通过注册表取得主模型属性：

```python
module = getattr(pipe, get_module_attr(pipeline_type))
pipe.load_lora(module, lora_config, alpha=alpha)
```

多个 LoRA 通过 ComfyUI 连线表达：

```text
Loader -> LoRA Clear -> LoRA Load 1 -> LoRA Load 2 -> Inference
```

## 11. 签名解析器

[`signature_parser.py`](./signature_parser.py) 是自动推理节点的核心。它读取：

```python
inspect.signature(pipeline_class.__call__)
```

然后将 Python 注解转换成 ComfyUI 类型：

| Python 注解 | ComfyUI 类型 |
|---|---|
| `str` | `STRING` |
| `int` | `INT` |
| `float` | `FLOAT` |
| `bool` | `BOOLEAN` |
| `Literal[...]` | Combo |
| `PIL.Image.Image` | `IMAGE` |
| 图片列表 | `IMAGE` |
| 其他复杂类型 | wildcard `*` |

它会跳过 `self`、进度条参数、LoRA 字典参数和可变参数。`Union[str, dict]` 只要包含 `str` 就降级为
STRING；复杂控制结构降级为 wildcard。

`_OVERRIDES` 针对 Z-Image 步数、CFG、Flux guidance、AceStep duration 等参数提供更合理默认值。

## 12. 动态推理节点

[`nodes/pipeline_inference.py`](./nodes/pipeline_inference.py) 遍历注册表，对每个 Pipeline 执行：

```text
PipelineMeta -> Pipeline class -> signature parser -> type() -> node mapping
```

所以会得到 `DiffSynthZImageInference`、`DiffSynthFluxInference`、`DiffSynthWanVideoInference`
等 20 个类。每个类都自动拥有 `pipe` 输入、解析出的参数、输出类型、`execute()` 和 `IS_CHANGED()`。

### execute 流程

1. 取出 `pipe`。
2. 删除值为 `None` 的 optional 参数。
3. 将 ComfyUI IMAGE Tensor 转为 PIL Image。
4. 执行 `pipe(**kwargs)`。
5. 根据注册表的 `output_type` 转换结果。

### 图片转换

`_to_pil()` 取 `[B,H,W,C]` Tensor 的第一张图，限制到 `0..1`，乘 255 转 uint8，再构造 PIL 图片。
`_to_image()` 将 PIL RGB 图转换回 `[B,H,W,C]` float Tensor。PIL 图片列表会堆叠成帧序列。

### 音频和多媒体转换

裸音频 Tensor 会包装为：

```python
{"waveform": tensor, "sample_rate": 44100}
```

`audio_video` 预期返回 `(video, audio)`，节点输出为 `("IMAGE", "AUDIO")`。纯视频当前也以
ComfyUI `IMAGE` 承载帧序列。

`IS_CHANGED()` 返回 seed；seed 变化会触发重新执行，没有 seed 时返回 NaN。

## 13. Z-Image 工作流调用链

```text
VRAM Config
    -> ModelConfig(transformer/text_encoder/vae)
    -> MergeModelConfigs
    -> Pipeline Loader
    -> ZImage Inference
    -> SaveImage
```

tokenizer 可以通过：

```text
ModelConfig(tokenizer) -> PipelineOthersBuilder -> Pipeline Loader.others
```

底层效果接近：

```python
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=configs,
    tokenizer_config=tokenizer_config,
    vram_limit=limit,
)
image = pipe(prompt="...", seed=42)
```

## 14. 验证结果

已执行：

```bash
python -m compileall -q plugin/ComfyUI/custom_nodes/DiffSynthStudioNodes
```

编译通过。在设置 PYTHONPATH 指向 ComfyUI custom_nodes 和 DiffSynth-Studio 后，插件成功导入并注册
29 个节点，并验证了 VRAM Config、ModelConfig、MergeModelConfigs 以及多个动态推理节点的输出类型。

当前环境 CUDA 不可用，因此没有下载真实模型或执行 GPU 推理。

## 15. 当前边界与后续工作

### Pipeline 缓存

当前 Loader 没有显式的全局 Pipeline 缓存。生产版本应根据 Pipeline 类型、模型配置、dtype、device
和显存限制缓存对象，并在切换模型时释放旧 Pipeline。

### 真实推理和依赖

真实推理需要 CUDA、模型文件、DiffSynth 依赖以及对应量化后端。LTX-2、MiniMax H3、MOVA 等音视频
Pipeline 还可能需要 `torchaudio` 等额外依赖。

### 音频采样率

当前裸音频 Tensor 统一包装为 44100 Hz。生产版本应从 Pipeline 元数据或返回值获取真实采样率。

### 复杂输入

ControlNet 输入、音频特征和复杂字典会降级为 wildcard，目前没有为每种高级数据开发独立节点。

### LoRA 模块

LoRA Load 依赖注册表的 `module_attr`。如果 Pipeline 主模型属性变化，需要同步更新注册表。

## 16. 总结

当前实现采用：

```text
静态配置节点 + Pipeline 注册表 + 反射式签名解析 + 动态推理节点
```

静态节点负责配置和生命周期；注册表描述 Pipeline 差异；签名解析器生成推理输入；动态节点负责
调用 Pipeline 和转换输出。新增 Pipeline 时，通常只需注册类路径、LoRA 属性和输出类型，特殊参数
再通过 `_OVERRIDES` 修正。

## Version 2 runtime updates

- LoRA loading auto-detects `dit`, `unet`, or `video_dit` and no longer asks for a duplicate pipeline selector.
- Pipeline Others Builder accepts scalar `from_pretrained` options and filters them against the active signature.
- Pipeline Loader uses a stable SHA-256 `IS_CHANGED` key based on model configuration fields.
- Audio output preserves pipeline-provided sample rates and emits ComfyUI's `waveform`/`sample_rate` dictionary.

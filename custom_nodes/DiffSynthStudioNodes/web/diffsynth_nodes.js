import { app } from "/scripts/app.js";

// Keep config sockets understandable when a Pipeline Others Builder is selected.
app.registerExtension({
  name: "DiffSynthStudioNodes.labels",
  nodeCreated(node) {
    if (node.comfyClass !== "DiffSynthPipelineOthersBuilder") return;
    const configNames = {
      ZImage: ["tokenizer_config"], Flux: ["tokenizer_1_config", "tokenizer_2_config", "nexus_gen_processor_config", "step1x_processor_config"],
      Flux2: ["tokenizer_config"], QwenImage: ["tokenizer_config", "processor_config"], ErnieImage: ["tokenizer_config"],
      StableDiffusion: ["tokenizer_config"], StableDiffusionXL: ["tokenizer_config", "tokenizer_2_config"],
      AnimaImage: ["tokenizer_config", "tokenizer_t5xxl_config"], BooguImage: ["processor_config"],
      JoyAIImage: ["processor_config"], Krea2: ["tokenizer_config"], Ideogram4: ["tokenizer_config"],
      HiDreamO1: ["processor_config"], WanVideo: ["tokenizer_config", "audio_processor_config"],
      LingBotVideo: ["processor_config"], LTX2AudioVideo: ["tokenizer_config", "stage2_lora_config"],
      MiniMaxH3: ["processor_config"], MiniMaxMusic3: ["tokenizer_config"], MovaAudioVideo: ["tokenizer_config"],
      AceStep: ["text_tokenizer_config", "silence_latent_config"]
    };
    const refresh = () => {
      const selected = node.widgets?.find(w => w.name === "pipeline_type")?.value;
      const names = configNames[selected] || [];
      node.inputs?.filter(i => /^config_[1-4]$/.test(i.name)).forEach((input, index) => {
        input.label = names[index] || `config_${index + 1}`;
      });
      node.setDirtyCanvas(true, true);
    };
    const pipeline = node.inputs?.find(i => i.name === "pipeline_type");
    if (pipeline?.widget) {
      const original = pipeline.widget.callback;
      pipeline.widget.callback = (...args) => { const result = original?.(...args); refresh(); return result; };
    }
    refresh();
  },
});

use crate::models::TextGeneration;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::quantized_mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use tokenizers::Tokenizer;

const MSPHI2_PATH_PREFIX: &'static str = "D:\\models\\phi-2";
const DOLPHIN_PATH_PREFIX: &'static str = "D:\\models\\dolphin_2.6";

pub fn load_model() -> (Phi, Tokenizer) {
    let config_filename = std::path::PathBuf::from(format!("{MSPHI2_PATH_PREFIX}\\config.json"));

    let config = std::fs::read_to_string(config_filename)
        .ok()
        .expect("Failed to load config.json");
    let config: PhiConfig = serde_json::from_str(&config)
        .ok()
        .expect("Failed to laod PHI Config");

    let filesnames = vec![
        std::path::PathBuf::from(format!(
            "{MSPHI2_PATH_PREFIX}\\model-00001-of-00002.safetensors"
        )),
        std::path::PathBuf::from(format!(
            "{MSPHI2_PATH_PREFIX}\\model-00002-of-00002.safetensors"
        )),
    ];

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&filesnames, DType::F32, &Device::Cpu)
            .ok()
            .expect("Failed to get files")
    };

    let tokenizer = Tokenizer::from_file(format!("{MSPHI2_PATH_PREFIX}\\tokenizer.json"))
        .expect("Failed to load tokenizer.json");

    (
        Phi::new(&config, vb).ok().expect("Failed to load model"),
        tokenizer,
    )
}

pub fn load_model_quantized() -> (QMixFormer, Tokenizer) {
    let tokerizer_filename =
        std::path::PathBuf::from(format!("{DOLPHIN_PATH_PREFIX}\\tokenizer.json"));

    let tokenizer =
        Tokenizer::from_file(tokerizer_filename).expect("Failed to load Quantized tokenizer");

    let weights_filename =
        std::path::PathBuf::from(format!("{DOLPHIN_PATH_PREFIX}\\model-q4k.gguf"));

    let config = Config::v2();

    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &weights_filename,
        &Device::Cpu,
    )
    .expect("Quantized VarBuilder failure");

    let model = QMixFormer::new_v2(&config, vb).expect("Failed to Load Quantized model");

    (model, tokenizer)
}

pub fn question(query: &str, pipeline: &mut TextGeneration) -> String {
    let response = pipeline.run(&query, 400);

    return response;
}

pub fn single_query(query: &str, pipeline: &mut TextGeneration) -> String {
    let system_message: String = format!("<|im_start|>system\nAs a friendly and helpful AI assistant named Abu. Your answer should be very concise and to the point. Do not repeat question or references.<|im_end|>\n");

    let question = format!("<|im_start|>user\nquestion: \"{query}\"\"\n<|im_end|>\n<|im_start|>assistant\n");

    let prompt = system_message + &question;

    let response = pipeline.run(&prompt,400);

    return response;
}

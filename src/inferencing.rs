use crate::models::Model as PhiModel;
use crate::models::ModelType;
use crate::models::TextGeneration;
use crate::utils;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::quantized_mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use diesel_async::RunQueryDsl;
use lazy_static::lazy_static;
use tokenizers::Tokenizer;

lazy_static! {
    pub static ref PHI_GGUF: (PhiModel, Tokenizer) =
        load_model(ModelType::GGUF).expect("Failed to load GGUF Model");
    pub static ref PHI_ST: (PhiModel, Tokenizer) =
        load_model(ModelType::SafeTensor).expect("Failed to SafeTensor Model");
}

pub fn load_model(model_type: ModelType) -> Result<(PhiModel, Tokenizer)> {
    match model_type {
        ModelType::GGUF => PHI_GGUF,
        ModelType::SafeTensor => PHI_ST,
    }
}

pub fn load_safetensor_model() -> Result<(PhiModel, Tokenizer)> {
    let safetensor_path = utils::safetensor_model_path();

    let config_file_path = safetensor_path.join("config.json");

    let config = std::fs::read_to_string(config_file_path)?;
    let config: PhiConfig = serde_json::from_str(&config)?;

    let filesnames = vec![
        safetensor_path.join("model-00001-of-00002.safetensors"),
        safetensor_path.join("model-00002-of-00002.safetensors"),
    ];

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&filesnames, DType::F32, &utils::get_device())
            .map_err(E::msg)?
    };

    let tokenizer_path = safetensor_path.join("tokenizer.json");

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    Ok((
        PhiModel::SafeTensor(Phi::new(&config, vb).ok().expect("Failed to load model")),
        tokenizer,
    ))
}

pub fn load_model_quantized() -> Result<(PhiModel, Tokenizer)> {
    let quantized_path = utils::infer_model_path();

    let tokerizer_file = quantized_path.join("tokenizer.json");

    let tokenizer =
        Tokenizer::from_file(tokerizer_file).expect("Failed to load Quantized tokenizer");

    let weights_filename = quantized_path.join("model-q4k.gguf");

    let config = Config::v2();

    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &weights_filename,
        &utils::get_device(),
    )?;

    let model = QMixFormer::new_v2(&config, vb).map_err(E::msg)?;

    Ok((PhiModel::Quantized(model), tokenizer))
}

pub fn question(query: &str, pipeline: &mut TextGeneration) -> Result<String> {
    let response = pipeline.run(&query, 400)?;

    return Ok(response);
}

pub fn single_query(query: &str, pipeline: &mut TextGeneration) -> Result<String> {
    let system_message: String = format!("<|im_start|>system\nAs a friendly and helpful AI assistant named Abu. Your answer should be very concise and to the point. Do not repeat question or references.<|im_end|>\n");

    let question =
        format!("<|im_start|>user\nquestion: \"{query}\"\"\n<|im_end|>\n<|im_start|>assistant\n");

    let prompt = system_message + &question;

    let response = pipeline.run(&prompt, 400)?;

    return Ok(response);
}

pub fn answer_with_context(
    query: &str,
    context: &str,
    pipeline: &mut TextGeneration,
) -> Result<String> {
    let system_message: String = format!("<|im_start|>System\nAs a Friendly and helpful AI Assistant, Your answer should be very concise and to the point. Do not repeat your self and always try to give references from metadata in context.<|im_end>\n");

    let context = format!("<|im_start|>Context\n{context}<|im_end|>");

    let verbose_prompt =
        system_message + &context + &format!("<|im_start|>Question\n{query}<|im_end|>\nAssistant");

    let repsonse = pipeline.run(&verbose_prompt, 2048)?;

    return Ok(response);
}

use crate::{models::EmbeddingModelType, utils};
use anyhow::{Error as E, Result};
use candle_core::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use lazy_static::lazy_static;
use tokenizers::{PaddingParams, Tokenizer};

lazy_static! {
    pub static ref EMB_TORCH: load_model(EmbeddingModelType::SafeTensor).expect("Failed to Load Embedding Model");
    pub static ref EMB_SAFETENSOR: load_model(EmbeddingModelType::TorchModel).expect("Failed to Load Torch Model");
}

pub fn load_model(model_type: EmbeddingModelType) -> Result<(BertModel, Tokenizer)> {
    match model_type {
        EmbeddingModelType::TorchModel => {
            return load_torch_embedding_model();
        }
        EmbeddingModelType::SafeTensor => {
            return load_safetensor_embedding_model();
        }
    }
}

pub fn load_torch_embedding_model() -> Result<(BertModel, Tokenizer)> {
    let embedding_model_path = utils::embedding_model_path();

    let config_path = embedding_model_path.join("config.json");
    let weights_path = embedding_model_path.join("pytorch_model.bin");
    let tokenizer_path = embedding_model_path.join("tokenizer.json");
    let config = std::fs::read_to_string(&config_path)?;

    let config: Config = serde_json::from_str(&config)?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    match tokenizer.get_padding_mut() {
        Some(padding) => padding.strategy = tokenizers::PaddingStrategy::BatchLongest,
        None => {
            let padding = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(padding));
        }
    }

    let vb = VarBuilder::from_pth(&weights_path, DTYPE, &utils::get_device())?;

    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

pub fn load_safetensor_embedding_model() -> Result<(BertModel, Tokenizer)> {
    let model_path = utils::safetensor_embedding_model_path();

    let config_filename = model_path.join("config.json");

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;

    let filesnames = vec![model_path.join("model.safetensors")];

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&filesnames, DType::F32, &utils::get_device())?
    };

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(E::msg)?;

    Ok((
        BertModel::load(vb, &config).expect("Failed to load Model"),
        tokenizer,
    ))
}

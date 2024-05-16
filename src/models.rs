use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::bert::BertModel;
use candle_transformers::models::phi::Model as Phi;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use diesel::deserialize::{Queryable, QueryableByName};
use diesel::prelude::Insertable;
use diesel::query_dsl::methods::{LimitDsl, OrderDsl};
use diesel_async::{AsyncPgConnection, RunQueryDsl};
use pgvector::Vector;
use pgvector::VectorExpressionMethods;
use tokenizers::Tokenizer;

pub enum Model {
    Quantized(QMixFormer),
    SafeTensor(Phi),
}

pub enum EmbeddingModelType {
    SafeTensor(BertModel),
    TorchModel(BertModel),
}

pub enum ModelType {
    GGUF,
    SafeTensor,
}

pub struct TextGeneration {
    pub model: Model,
    pub device: Device,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        let mut response: String = String::new();

        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        let mut tokens = tokens.get_ids().to_vec();
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => panic!("cannot find the endoftext token"),
        };
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Quantized(m) => m.forward(&input)?,
                Model::SafeTensor(m) => m.forward(&input)?,
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if next_token == eos_token || next_token == 198 {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;

            response += &token;
        }
        return Ok(response);
    }
}

#[derive(Insertable)]
#[diesel(table_name=crate::schema::items)]

struct ItemsInsertable {
    chunk_number: Option<i32>,
    title: String,
    content: Option<String>,
    embedding: Option<Vector>,
}

#[derive(Queryable, QueryableByName)]
#[diesel(table_name=crate::schema::items)]
struct Items {
    pub id: i32,
    pub chunk_number: Option<i32>,
    pub title: String,
    pub content: Option<String>,
    pub embedding: Option<Vector>,
}

impl Items {
    async fn new(
        chunk_number: Option<i32>,
        title: String,
        content: Option<String>,
        embedding: Option<Vector>,
    ) -> Self {
        Self {
            id: 0,
            chunk_number,
            title,
            content,
            embedding,
        }
    }
    async fn insert(&self, conn: &mut AsyncPgConnection) -> Result<(), diesel::result::Error> {
        ItemsInsertable {
            chunk_number: self.chunk_number,
            title: self.title.clone(),
            content: self.content.clone(),
            embedding: self.embedding.clone(),
        }
        .insert_into(crate::schema::items::table)
        .execute(conn)
        .await?;
        Ok(())
    }
    async fn get_cosine_similar(
        conn: &mut AsyncPgConnection,
        query: Option<Vector>,
        limit: i64,
    ) -> Result<Vec<Self>, diesel::result::Error> {
        use crate::schema::items;

        let res = items::table
            .order(items::embedding.cosine_distance(query))
            .limit(limit)
            .load::<Self>(conn)
            .await?;

        return Ok(res);
    }
}

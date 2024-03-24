use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi::Model as Phi;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use tokenizers::Tokenizer;

pub enum Model {
    Quantized(QMixFormer),
    SafeTensor(Phi),
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

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> String {
        let mut response: String = String::new();

        use std::io::Write;
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .ok()
            .expect("Failed to encode prompt");
        if tokens.is_empty() {
            panic!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => panic!("cannot find the endoftext token"),
        };
        std::io::stdout().flush().expect("Unable to Flush Stdout");
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)
                .expect("Unable get Input Tensor")
                .unsqueeze(0)
                .ok()
                .expect("Failed to get Input ensort final");
            let logits = match &mut self.model {
                Model::Quantized(m) => m.forward(&input).ok().expect("Failed to logits Tensor"),
                Model::SafeTensor(m) => m.forward(&input).ok().expect("Failed to logits Tensor"),
            };
            let logits = logits
                .squeeze(0)
                .expect("Failed to logits 2 Tensor")
                .to_dtype(DType::F32)
                .expect("Failed to logits 2 Tensor Final");
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                .expect("Unable to apply repeated penalty")
            };

            let next_token = self
                .logits_processor
                .sample(&logits)
                .expect("Unable to get next token");
            tokens.push(next_token);
            if next_token == eos_token || next_token == 198 {
                break;
            }
            let token = self
                .tokenizer
                .decode(&[next_token], true)
                .expect("Failed to get token");

            response += &token;
        }
        return response;
    }
}

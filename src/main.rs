use candle_core::Device;
use inferencing::single_query;

pub mod inferencing;
pub mod models;
pub mod utils;

fn main() {
    let (phi, tokenizer) = inferencing::load_model_quantized();
    let mut pipeline = models::TextGeneration::new(
        models::Model::Quantized(phi),
        tokenizer,
        398752958,
        Some(0.3),
        None,
        1.1,
        64,
        false,
        &Device::Cpu,
    );

    let query = "How the 7th USA President?";

    let resp = single_query(query, &mut pipeline);

    println!("User > {query}");
    println!("Assistant > {resp}");
}
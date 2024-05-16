use anyhow::{Error as E, Result};
use inferencing::single_query;

pub mod database;
pub mod embedding;
pub mod inferencing;
pub mod models;
pub mod utils;

#[tokio::main]
async fn main() -> Result<()> {
    let (model, tokenizer) = inferencing::load_model(models::ModelType::GGUF)?;
    let mut pipeline = models::TextGeneration::new(
        model,
        tokenizer,
        398752958,
        Some(0.3),
        None,
        1.1,
        64,
        false,
        &utils::get_device(),
    );

    let mut query = String::new();

    while query != "Exit" {

        std::io::stdin().read_line(&mut query)?;

        let resp = single_query(&query, &mut pipeline).map_err(E::msg)?;

        println!("User > {query}");
        println!("Assistant > {resp}");
    }

    Ok(())
}

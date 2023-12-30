use std::{error::Error, path::Path};

mod rustplexity;
use rustplexity::BigramPerplexityModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let uni_filename = Path::new("unigrams.txt");
    let bi_filename = Path::new("bigrams.txt");
    let model = BigramPerplexityModel::from_file(uni_filename, bi_filename).await?;

    loop {
        let mut sentence = String::new();
        println!("Enter a sentence:");
        std::io::stdin().read_line(&mut sentence)?;
        println!("perplexity: {}\n", model.compute_sentence(sentence.trim()));
    }
}

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use tokio::fs::File as AsyncFile;
use tokio::io::AsyncBufReadExt;

lazy_static! {
    static ref PUNCTUATION_PATTERN: Regex = Regex::new(r#"([,;:.!?¿¡()<>=\"'`])"#).unwrap();
}

struct BigramData {
    bigrams: HashMap<String, f64>,
    unigrams: HashMap<String, f64>,
}

pub struct BigramPerplexityModel {
    data: BigramData,
}

impl BigramPerplexityModel {
    pub fn new() -> BigramPerplexityModel {
        BigramPerplexityModel {
            data: BigramData {
                bigrams: HashMap::new(),
                unigrams: HashMap::new(),
            },
        }
    }

    async fn load_hashmap_from_file<P>(
        filename: P,
    ) -> Result<HashMap<String, f64>, Box<dyn Error + Send + Sync>>
    where
        P: AsRef<Path> + Send + Sync,
    {
        let mut hashmap = HashMap::new();
        let file = AsyncFile::open(filename).await?;
        let reader = tokio::io::BufReader::new(file);
        let mut lines = reader.lines();

        while let Some(line) = lines.next_line().await? {
            let mut split = line.rsplitn(2, ' ');
            let prob = split.next().unwrap_or("").parse::<f64>()?;
            let word = split.next().unwrap_or("").to_string();
            hashmap.insert(word, prob);
        }

        Ok(hashmap)
    }

    /// Loads a model from the unigram and bigram files
    /// These are text files with the following format:
    /// word(string) probability(float)
    /// word1(string) word2(string) probability(float)
    pub async fn from_file<P>(
        unigrams_filename: P,
        bigrams_filename: P,
    ) -> Result<BigramPerplexityModel, Box<dyn Error + Send + Sync>>
    where
        P: AsRef<Path> + Send + Sync + 'static,
    {
        let start_time = std::time::Instant::now();
        let mut model = BigramPerplexityModel::new();

        println!(
            "Loading model from files: {} and {}",
            unigrams_filename.as_ref().display(),
            bigrams_filename.as_ref().display()
        );
        let unigram_task = tokio::spawn(async move {
            BigramPerplexityModel::load_hashmap_from_file(unigrams_filename).await
        });

        let bigram_task = tokio::spawn(async move {
            BigramPerplexityModel::load_hashmap_from_file(bigrams_filename).await
        });

        let (unigrams_result, bigrams_result) = tokio::try_join!(unigram_task, bigram_task)?;
        model.data.unigrams = unigrams_result?;
        model.data.bigrams = bigrams_result?;

        println!("Loaded model in {} seconds", start_time.elapsed().as_secs());
        Ok(model)
    }

    fn tokenize_sentence(&self, sentence: &str) -> Vec<String> {
        let parsed = PUNCTUATION_PATTERN.replace_all(sentence, " $0 ");
        let mut tokens = Vec::new();
        for token in parsed.split_ascii_whitespace() {
            let s = token.to_string();
            if !s.is_empty() {
                tokens.push(s);
            }
        }
        tokens
    }

    pub fn compute_sentence(&self, sentence: &str) -> f64 {
        let words = self.tokenize_sentence(&sentence.to_lowercase());
        let num_words = words.len() as f64;
        if num_words == 0. {
            return 0.;
        }
        let mut prev_word = "#".to_string();
        let mut log_prob_sum = 0.;
        for word in words {
            let bigram = format!("{} {}", prev_word, word);
            let bi_dec = match self.data.bigrams.get(&bigram) {
                Some(x) => *x,
                None => 0.,
            };
            let uni_dec = match self.data.unigrams.get(word.as_str()) {
                Some(x) => *x,
                None => 0.,
            };
            prev_word = word;
            log_prob_sum += (bi_dec * 0.8 + uni_dec * 0.2 + 1e-6).log2();
        }
        (2.0_f64).powf(-log_prob_sum / (num_words))
    }
}

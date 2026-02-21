use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    word_to_id: BTreeMap<String, u64>,
    id_to_word: BTreeMap<u64, String>,
    next_id: u64,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            word_to_id: BTreeMap::new(),
            id_to_word: BTreeMap::new(),
            next_id: 1,
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<u64> {
        text.split_whitespace()
            .map(|word| {
                if let Some(id) = self.word_to_id.get(word) {
                    *id
                } else {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.word_to_id.insert(word.to_string(), id);
                    self.id_to_word.insert(id, word.to_string());
                    id
                }
            })
            .collect()
    }

    pub fn decode(&self, id: u64) -> String {
        self.id_to_word
            .get(&id)
            .cloned()
            .unwrap_or_else(|| "[UNK]".to_string())
    }

    pub fn normalize_text(text: &str) -> String {
        let mut normalized = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch.is_ascii_alphanumeric() {
                normalized.push(ch.to_ascii_lowercase());
            } else if ch.is_whitespace() || ch.is_ascii_punctuation() {
                normalized.push(' ');
            }
        }
        normalized
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn levenshtein(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
        let mut curr = vec![0; b_chars.len() + 1];

        for (i, ca) in a_chars.iter().enumerate() {
            curr[0] = i + 1;
            for (j, cb) in b_chars.iter().enumerate() {
                let cost = if ca == cb { 0 } else { 1 };
                curr[j + 1] = std::cmp::min(
                    std::cmp::min(curr[j] + 1, prev[j + 1] + 1),
                    prev[j] + cost,
                );
            }
            prev.clone_from_slice(&curr);
        }

        prev[b_chars.len()]
    }

    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut tokenizer: Self = serde_json::from_reader(BufReader::new(file))?;
        if tokenizer.next_id == 0 {
            tokenizer.next_id = 1;
        }
        Ok(tokenizer)
    }
}

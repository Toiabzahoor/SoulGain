use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use soulgain::eng::EnglishEngine;

fn main() -> io::Result<()> {
    let mut engine = EnglishEngine::new();
    let dialogue_lines = load_dialogue_lines("dialogue.txt")?;

    let max_epochs = 50usize;
    for _ in 0..max_epochs {
        for line in &dialogue_lines {
            engine.train_on_text(line);
        }
    }

    println!("[System] Chat mode ready. Type your message:");
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let input = line?.trim().to_string();
        if input.is_empty() {
            println!("Ready.");
            continue;
        }
        let formatted = format!("USER: {}", input);
        let response = engine.prompt(&formatted);
        println!("{}", response);
    }

    if let Err(err) = engine.save_vocab() {
        eprintln!("[System] Failed to save vocab: {}", err);
    }

    Ok(())
}

fn load_dialogue_lines(path: &str) -> io::Result<Vec<String>> {
    let path = Path::new(path);
    let mut lines = Vec::new();

    if path.exists() {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                lines.push(trimmed.to_string());
            }
        }
    }

    if lines.is_empty() {
        lines.extend(vec![
            "Hey. Greetings.".to_string(),
            "Hi. System online.".to_string(),
            "Who are you? I am SoulGain.".to_string(),
            "Ready. Ready.".to_string(),
        ]);
    }

    Ok(lines)
}

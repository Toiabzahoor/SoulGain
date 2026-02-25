mod tictactoe;
mod chess_game;

use std::io::{self, Write};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                           SOULGAIN ACTIVE                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    loop {
        println!("\nCHOOSE MODULE:");
        println!("  [1] Convert Kaggle CSV to Bincode (One-time Setup)");
        println!("  [2] Train from Bincode Dataset (Lightning Fast)");
        println!("  [3] Train from Stockfish (The Forge - Live Oracle)");
        println!("  [4] Play: SoulGain vs Stockfish (MCTS Navigator)");
        println!("  [5] Play: SoulGain vs SoulGain (MCTS Navigator)");
        println!("  [6] Exit");
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => {
                println!("Enter path to Kaggle CSV:");
                let mut path = String::new();
                io::stdin().read_line(&mut path).unwrap();
                chess_game::convert_kaggle_to_bincode(path.trim(), "chess_data.bin");
            },
            "2" => chess_game::train_from_bincode("chess_data.bin"),
            "5" => chess_game::play_self(),
            "6" => break,
            _ => println!("Invalid option."),
        }
    }
}
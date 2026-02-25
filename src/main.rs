mod tictactoe;
mod chess_game;

use std::io::{self, Write};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                           SOULGAIN ACTIVE                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    loop {
        println!("\nCHOOSE MODULE:");
        println!("  [1] Train Matrices (Stockfish vs Stockfish - THE FORGE)");
        println!("  [2] Play: SoulGain vs Stockfish (MCTS Navigator)");
        println!("  [3] Play: SoulGain vs SoulGain (MCTS Navigator)");
        println!("  [4] Exit");
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => chess_game::train_from_stockfish(),
            "2" => chess_game::play_vs_stockfish(),
            "3" => chess_game::play_self(),
            "4" => break,
            _ => println!("Invalid option."),
        }
    }
}
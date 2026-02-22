mod tictactoe;
mod chess_game;

use std::io::{self, Write};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                           SOULGAIN ACTIVE                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    loop {
        println!("\nCHOOSE MODULE TO WORK WITH:");
        println!("  [1] Tic-Tac-Toe");
        println!("  [2] Chess (Watch 1 High-Quality Match)");
        println!("  [3] Chess (Massive Self-Play Evolution)");
        println!("  [4] Exit");
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => tictactoe::run_interactive(),
            "2" => chess_game::run_self_match(),
            "3" => chess_game::run_autoplay(),
            "4" => break,
            _ => println!("⚠️ Invalid option."),
        }
    }
}
mod tictactoe;

use std::io::{self, Write};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                           SOULGAIN ACTIVE                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    loop {
        println!("\nCHOOSE MODULE TO WORK WITH:");
        println!("  [1] Tic-Tac-Toe (Human vs AGI)");
        println!("  [2] Tic-Tac-Toe (AGI Autoplay Training)");
        println!("  [3] Exit");
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            continue;
        }

        match input.trim() {
            "1" => tictactoe::run_interactive(),
            "2" => tictactoe::run_autoplay(),
            "3" | "exit" | "quit" => {
                println!("Shutting down SoulGain. Goodbye! 💖");
                break;
            }
            _ => println!("⚠️  Invalid option. Please choose 1, 2, or 3."),
        }
    }
}
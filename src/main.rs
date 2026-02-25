// main.rs

use std::io::{self, BufRead};

mod tictactoe;  // assuming your soulgain crate/module is available

// If the code you showed is in a separate file/module, import it like this:
// mod tictactoe;
use tictactoe::{run_interactive, run_autoplay};

           // adjust path if needed
// or if everything is in one big file:
// (then you don't need the above line)

fn main() {
    println!("=====================================");
    println!("      SoulGain Tic-Tac-Toe v0.1      ");
    println!("  True Negamax + Plasticity Learning  ");
    println!("=====================================\n");

    loop {
        println!("\nChoose mode:");
        println!("  [1]  Play against AI (you = O)");
        println!("  [2]  Self-play training (AI vs AI)");
        println!("  [q]  Quit");
        print!("\n→ ");

        let stdin = io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line).expect("Failed to read line");

        let choice = line.trim().to_lowercase();

        match choice.as_str() {
            "1" | "p" | "play" => {
                println!("\nStarting interactive game...\n");
                run_interactive();
            }

            "2" | "s" | "self" | "train" => {
                println!("\nStarting self-play training...\n");
                run_autoplay();
            }

            "q" | "quit" | "exit" => {
                println!("Goodbye! 🧠");
                break;
            }

            _ => {
                println!("Invalid choice. Please enter 1, 2 or q.");
            }
        }
    }
}

// If you put run_interactive() and run_autoplay() directly in main.rs,
// just keep them here (after the main function).

// Otherwise, move them to lib.rs / tictactoe.rs and import as shown above.
use std::io::{self, Write};
use soulgain::abilities::raycasting;

fn main() {
    println!("💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖");
    println!("✨        Awakening the Machine: Spatial Evolution        ✨");
    println!("💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖\n");

    println!("How would you like to interact with the Machine today?");
    println!("1. 🌸 Training Mode (Run self-play spatial evolution)");
    println!("2. 👁️  Exhibition Mode (Watch the Machine solve a puzzle)");
    print!("\nEnter your beautiful choice (1 or 2): ");
    
    // Ensure the prompt prints before waiting for input
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();

    println!(); // Just a lovely little empty line for spacing

    match choice.trim() {
        "1" => {
            raycasting::run_raycast_training();
        }
        "2" => {
            raycasting::run_raycast_exhibition();
        }
        _ => {
            println!("Oh goodness! That wasn't a 1 or a 2. Please run the program again! 💖");
        }
    }
}
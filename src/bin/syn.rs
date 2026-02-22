//! syn.rs – Human vs. Robot: The True Grandmaster Update! 🤖⭕❌
//!
//! The Robot now calculates the AVERAGE outcome of 30 different timelines 
//! for every single move, curing its optimism and making it a defensive monster!

use std::io::{self, Write};
use std::time::{SystemTime, UNIX_EPOCH};
use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. The Game Board & Rules 🎲
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct TicTacToe {
    board: [u8; 9], 
    seed: u64,      
}

impl TicTacToe {
    fn check_winner(&self) -> u8 {
        let b = self.board;
        let lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], 
            [0, 3, 6], [1, 4, 7], [2, 5, 8], 
            [0, 4, 8], [2, 4, 6]             
        ];
        for l in lines {
            if b[l[0]] != 0 && b[l[0]] == b[l[1]] && b[l[1]] == b[l[2]] {
                return b[l[0]];
            }
        }
        0
    }

    fn find_winning_move(&self, player: u8) -> Option<usize> {
        let lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ];
        for l in lines {
            let p_count = (self.board[l[0]] == player) as u8 
                        + (self.board[l[1]] == player) as u8 
                        + (self.board[l[2]] == player) as u8;
            let empty_count = (self.board[l[0]] == 0) as u8 
                            + (self.board[l[1]] == 0) as u8 
                            + (self.board[l[2]] == 0) as u8;
            
            if p_count == 2 && empty_count == 1 {
                for &idx in &l {
                    if self.board[idx] == 0 { return Some(idx); }
                }
            }
        }
        None
    }

    fn print_board(&self) {
        let ch = |i, c| match c { 
            1 => "❌".to_string(), 
            2 => "⭕".to_string(), 
            _ => format!(" {} ", i) 
        };
        println!("   │   │   ");
        println!(" {}│{}│{}", ch(0, self.board[0]), ch(1, self.board[1]), ch(2, self.board[2]));
        println!(" ──┼───┼── ");
        println!(" {}│{}│{}", ch(3, self.board[3]), ch(4, self.board[4]), ch(5, self.board[5]));
        println!(" ──┼───┼── ");
        println!(" {}│{}│{}", ch(6, self.board[6]), ch(7, self.board[7]), ch(8, self.board[8]));
        println!("   │   │   \n");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. The AI's Dream Simulator (Averaging Multiple Timelines) 🧠☁️
// ─────────────────────────────────────────────────────────────────────────────

impl UniversalWorld for TicTacToe {
    type State = [u8; 9];
    type Action = usize; 

    fn current_state(&self) -> Self::State {
        self.board
    }

    fn step(&mut self, action: Self::Action) -> Result<(), ()> {
        if action > 8 || self.board[action] != 0 || self.is_terminal() {
            return Err(()); 
        }
        
        // 🤖 AI moves
        self.board[action] = 1; 
        if self.is_terminal() { return Ok(()); }

        // 🧑‍🦱 Ghost Human responds
        if let Some(m) = self.find_winning_move(2) {
            self.board[m] = 2; // Always take a win
        } else if let Some(m) = self.find_winning_move(1) {
            self.board[m] = 2; // Always block a loss
        } else {
            // Pick a completely random spot, forcing the AI to prepare for chaos
            let empties: Vec<usize> = (0..9).filter(|&i| self.board[i] == 0).collect();
            if !empties.is_empty() {
                self.seed ^= self.seed << 13;
                self.seed ^= self.seed >> 7;
                self.seed ^= self.seed << 17;
                let random_idx = (self.seed as usize) % empties.len();
                self.board[empties[random_idx]] = 2;
            }
        }

        Ok(())
    }

    fn is_terminal(&self) -> bool {
        self.check_winner() != 0 || !self.board.contains(&0) 
    }

    /// 🌟 THE FIX: We run 30 chaotic rollouts and return the AVERAGE score!
    fn evaluate_path(&self, path: &[Self::Action]) -> (f32, u64) {
        let mut base_sim = self.clone();
        let mut cost = 0;
        
        for &m in path {
            cost += 1;
            if base_sim.step(m).is_err() { return (0.01, cost); } // 0.01 = Terrible!
        }
        
        if base_sim.is_terminal() {
            let w = base_sim.check_winner();
            let score = if w == 1 { 1.0 } else if w == 2 { 0.1 } else { 0.5 };
            return (score, cost);
        }

        let rollouts = 30; // Dream 30 alternate timelines
        let mut total_score = 0.0;

        for i in 0..rollouts {
            let mut r_sim = base_sim.clone();
            r_sim.seed = r_sim.seed.wrapping_add(i as u64).wrapping_mul(1099511628211);
            
            while !r_sim.is_terminal() {
                cost += 1;
                let empties: Vec<usize> = (0..9).filter(|&idx| r_sim.board[idx] == 0).collect();
                if empties.is_empty() { break; }
                
                // AI plays intelligently during the dream
                if let Some(w) = r_sim.find_winning_move(1) {
                    let _ = r_sim.step(w);
                } else if let Some(b) = r_sim.find_winning_move(2) {
                    let _ = r_sim.step(b);
                } else {
                    r_sim.seed ^= r_sim.seed << 13;
                    r_sim.seed ^= r_sim.seed >> 7;
                    let random_idx = (r_sim.seed as usize) % empties.len();
                    let _ = r_sim.step(empties[random_idx]); 
                }
            }
            
            let winner = r_sim.check_winner();
            // Score the timeline: Win = 1.0, Draw = 0.5, Loss = 0.1
            total_score += if winner == 1 { 1.0 } else if winner == 2 { 0.1 } else { 0.5 };
        }

        // Return the average probability of winning!
        let avg_score = total_score / (rollouts as f32);
        (avg_score, cost)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The Cognitive Policy 🔮
// ─────────────────────────────────────────────────────────────────────────────

struct TttPolicy {
    pub brain: Plasticity,
}

impl CognitivePolicy<[u8; 9], usize> for TttPolicy {
    fn evaluate(&self, _state: &[u8; 9]) -> f32 { 0.0 }

    fn priors(&self, state: &[u8; 9]) -> Vec<(usize, f32)> {
        let mut valid_moves = Vec::new();
        for i in 0..9 {
            if state[i] == 0 { valid_moves.push(i); }
        }
        if valid_moves.is_empty() { return vec![]; }
        
        // Hash the TicTacToe board into a memory context
        let mut hash = 0_u64;
        for (i, &val) in state.iter().enumerate() {
            hash ^= (val as u64) << (i * 2);
        }
        let ctx = Event::ContextWithState { data: [0; 16], state_hash: hash };

        // Ask the neural network what it thinks!
        self.brain.get_op_distribution(ctx, &valid_moves)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. The Interactive Game Loop! 🎈
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖");
    println!("✨     🤖 Robot vs Human: The Grandmaster Match! 🎮         ✨");
    println!("💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖✨💖\n");

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() as u64;
    let mut game = TicTacToe { board: [0; 9], seed: nanos };
    
    // We set max_program_len to 1, forcing it to test single moves thoroughly 
    // rather than looking for optimistic long sequences!
    let config = ReasoningConfig::<usize> {
        simulations: 200,              
        max_depth: 1,                  
        max_program_len: 1,            
        max_ops_per_candidate: 1,
        exploration_constant: 1.41,
        length_penalty: 0.0,
        loop_penalty: 0.0,
        action_space: vec![0, 1, 2, 3, 4, 5, 6, 7, 8], 
        arena_capacity: 1_000_000,
    };
    
    let policy = TttPolicy;

    game.print_board();

    loop {
        // 🧑‍🦱 HUMAN TURN
        print!("Your turn, human! Type a number (0-8): ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let m: usize = match input.trim().parse() {
            Ok(num) if num < 9 && game.board[num] == 0 => num,
            _ => {
                println!("⚠️  Oops! Invalid move. Please pick an empty spot 0-8.");
                continue;
            }
        };
        
        println!("\n🧑‍🦱 You chose spot {}!", m);
        game.board[m] = 2; // Human is ⭕
        game.print_board();

        if game.check_winner() == 2 {
            println!("🎉 WOW! You outsmarted the Robot! YOU WIN! 🎉\n");
            break;
        } else if !game.board.contains(&0) {
            println!("🤝 It's a draw! The board is full.\n");
            break;
        }

        // 🤖 AI TURN
        println!("🤖 The Robot is averaging probabilities across timelines...\n");
        
        let (best_path_opt, stats) = solve_universal_with_stats(&game, &config, &policy);

        let ai_move = if let Some(path) = best_path_opt {
            if !path.is_empty() {
                path[0] 
            } else {
                game.board.iter().position(|&c| c == 0).unwrap() 
            }
        } else {
            game.board.iter().position(|&c| c == 0).unwrap() 
        };

        println!("✨ The Robot executed {} evaluations in {:.2}ms! ✨", stats.total_op_cycles, stats.elapsed_ms);
        println!("🤖 The Robot confidently places its ❌ in spot {}!", ai_move);
        game.board[ai_move] = 1; // AI is ❌
        game.print_board();

        if game.check_winner() == 1 {
            println!("☠️  OH NO! The Robot trapped you! AI WINS! 🤖\n");
            break;
        } else if !game.board.contains(&0) {
            println!("🤝 Phew! It's a draw. You survived!\n");
            break;
        }
    }
}
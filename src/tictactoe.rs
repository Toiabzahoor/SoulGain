use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats,
};
use soulgain::plasticity::{Event, IntoOpcode, Plasticity, VMError};



// ─────────────────────────────────────────────────────────────────────────────
// True Self-Play Tic-Tac-Toe
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct TicTacToe {
    board: [u8; 9],
    current_turn: u8, // 1 for X, 2 for O
    seed: u64,
}

impl TicTacToe {
    fn check_winner(&self) -> u8 {
        let b = self.board;
        let lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
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
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
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
                    if self.board[idx] == 0 {
                        return Some(idx);
                    }
                }
            }
        }
        
        None
    }

    fn state_hash(&self) -> u64 {
        let mut hash = 0_u64;
        for (i, &val) in self.board.iter().enumerate() {
            hash |= (val as u64) << (i * 2);
        }
        hash
    }

    fn print_board(&self) {
        let ch = |i, c| match c {
            1 => "❌".to_string(),
            2 => "⭕".to_string(),
            _ => format!(" {} ", i),
        };
        
        println!("   │   │   ");
        println!(
            " {}│{}│{}",
            ch(0, self.board[0]),
            ch(1, self.board[1]),
            ch(2, self.board[2])
        );
        println!(" ──┼───┼── ");
        println!(
            " {}│{}│{}",
            ch(3, self.board[3]),
            ch(4, self.board[4]),
            ch(5, self.board[5])
        );
        println!(" ──┼───┼── ");
        println!(
            " {}│{}│{}",
            ch(6, self.board[6]),
            ch(7, self.board[7]),
            ch(8, self.board[8])
        );
        println!("   │   │   \n");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Universal World Implementation
// ─────────────────────────────────────────────────────────────────────────────

impl UniversalWorld for TicTacToe {
    type State = [u8; 9];
    type Action = usize;

    fn current_state(&self) -> Self::State {
        self.board
    }

    // 🌟 NEGAMAX FLAG
    fn current_player(&self) -> i32 {
        if self.current_turn == 1 {
            1
        } else {
            -1
        }
    }

    fn step(&mut self, action: Self::Action) -> Result<(), ()> {
        if action > 8 || self.board[action] != 0 || self.is_terminal() {
            return Err(());
        }

        self.board[action] = self.current_turn;
        self.current_turn = if self.current_turn == 1 { 2 } else { 1 };

        Ok(())
    }

    fn is_terminal(&self) -> bool {
        self.check_winner() != 0 || !self.board.contains(&0)
    }

    fn evaluate_path(&self, path: &[Self::Action]) -> (f32, u64) {
        let mut sim = self.clone();
        let mut cost = 0;

        for &m in path {
            cost += 1;
            if sim.step(m).is_err() {
                // If it fails, return worst possible absolute score for the player who tried it
                let failed_player = sim.current_player();
                return (if failed_player == 1 { -1.0 } else { 1.0 }, cost); 
            }
        }

        while !sim.is_terminal() {
            cost += 1;
            let empties: Vec<usize> = (0..9).filter(|&idx| sim.board[idx] == 0).collect();
            if empties.is_empty() {
                break;
            }

            let opponent = if sim.current_turn == 1 { 2 } else { 1 };

            if let Some(w) = sim.find_winning_move(sim.current_turn) {
                let _ = sim.step(w);
            } else if let Some(b) = sim.find_winning_move(opponent) {
                let _ = sim.step(b);
            } else {
                sim.seed ^= sim.seed << 13;
                sim.seed ^= sim.seed >> 7;
                sim.seed ^= sim.seed << 17;
                let random_idx = (sim.seed as usize) % empties.len();

                let _ = sim.step(empties[random_idx]);
            }
        }

        let winner = sim.check_winner();
        
        // 🌟 ABSOLUTE SCORING: Player 1's perspective
        let score = if winner == 1 {
            1.0
        } else if winner == 2 {
            -1.0 
        } else {
            0.0
        };

        (score, cost)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Neural Policy
// ─────────────────────────────────────────────────────────────────────────────

struct TttPolicy {
    brain: Plasticity,
}

impl CognitivePolicy<[u8; 9], usize> for TttPolicy {
    fn evaluate(&self, _state: &[u8; 9]) -> f32 {
        0.0
    }

    fn priors(&self, state: &[u8; 9]) -> Vec<(usize, f32)> {
        let valid_moves: Vec<usize> = (0..9).filter(|&i| state[i] == 0).collect();
        if valid_moves.is_empty() {
            return vec![];
        }

        let mut hash = 0_u64;
        for (i, &val) in state.iter().enumerate() {
            hash |= (val as u64) << (i * 2);
        }
        
        let ctx = Event::ContextWithState {
            data: [0; 16],
            state_hash: hash,
        };

        self.brain.get_op_distribution(ctx, &valid_moves)
    }
}

fn load_brain() -> Plasticity {
    let brain = Plasticity::new();
    if Path::new("plasticity.json").exists() {
        let _ = brain.load_from_file("plasticity.json");
        println!("🧠 [Loaded stored memories from plasticity.json]");
    } else {
        println!("🧠 [Initialized fresh neural pathways]");
    }
    brain
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode 1: Interactive Play (True Negamax Vs Human)
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_interactive() {
    println!("\nLoading module: TIC-TAC-TOE (TRUE NEGAMAX VS HUMAN)...");
    let brain = load_brain();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
        
    let mut game = TicTacToe {
        board: [0; 9],
        current_turn: 1, 
        seed: nanos,
    };

    let config = ReasoningConfig::<usize> {
        simulations: 10_000,
        max_depth: 9, 
        max_program_len: 9,
        max_ops_per_candidate: 9,
        exploration_constant: 1.41,
        length_penalty: 0.0,
        loop_penalty: 0.0,
        action_space: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        arena_capacity: 1_000_000,
    };
    
    let policy = TttPolicy {
        brain: brain.clone(),
    };

    game.print_board();

    loop {
        if game.current_turn == 2 {
            print!("Your turn, human! Type a number (0-8) or 'q' to quit: ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            
            if input.trim().eq_ignore_ascii_case("q") {
                break;
            }

            if let Ok(m) = input.trim().parse::<usize>() {
                if m < 9 && game.board[m] == 0 {
                    game.step(m).unwrap();
                    game.print_board();
                } else {
                    println!("⚠️  Invalid move. That spot is taken or out of bounds.");
                    continue;
                }
            } else {
                continue;
            }
            
        } else {
            println!("🤖 The AGI is mapping adversarial space...\n");
            
            let (best_path, stats) = solve_universal_with_stats(&game, &config, &policy);
            let ai_move = best_path
                .and_then(|p| p.first().copied())
                .unwrap_or_else(|| game.board.iter().position(|&c| c == 0).unwrap());

            let hash = game.state_hash();
            brain.observe(Event::ContextWithState {
                data: [0; 16],
                state_hash: hash,
            });
            brain.observe(Event::Opcode {
                opcode: ai_move as i64,
                stack_depth: 0,
            });

            println!(
                "✨ Explored {} deep branches in {:.2}ms!",
                stats.total_op_cycles, stats.elapsed_ms
            );
            println!("🤖 The AGI places its ❌ in spot {}!", ai_move);
            
            game.step(ai_move).unwrap();
            game.print_board();
        }

        let w = game.check_winner();
        if w == 1 {
            println!("☠️  AI WINS! 🤖\n");
            brain.observe(Event::Reward(100));
            break;
        }
        if w == 2 {
            println!("🎉 YOU WIN! 🎉\n");
            brain.observe(Event::Error(VMError::Crash));
            break;
        }
        if !game.board.contains(&0) {
            println!("🤝 It's a draw.\n");
            brain.observe(Event::Reward(50));
            break;
        }
    }
    
    brain.sync();
    let _ = brain.save_to_file("plasticity.json");
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode 2: Autoplay (True Self-Play Training)
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_autoplay() {
    println!("\nLoading module: NEGAMAX SELF-PLAY TRAINING...");
    print!("How many self-play games? ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let num_games: usize = input.trim().parse().unwrap_or(100);

    let brain = load_brain();
    
    let config = ReasoningConfig::<usize> {
        simulations: 2000,
        max_depth: 9,
        max_program_len: 9,
        max_ops_per_candidate: 9,
        exploration_constant: 1.41,
        length_penalty: 0.0,
        loop_penalty: 0.0,
        action_space: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        arena_capacity: 1_000_000,
    };
    
    let policy = TttPolicy {
        brain: brain.clone(),
    };

    let mut x_wins = 0;
    let mut o_wins = 0;
    let mut draws = 0;
    
    let start_time = SystemTime::now();

    for g in 1..=num_games {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64;
            
        let mut game = TicTacToe {
            board: [0; 9],
            current_turn: 1,
            seed: nanos,
        };

        while !game.is_terminal() {
            let (best_path, _) = solve_universal_with_stats(&game, &config, &policy);
            let ai_move = best_path
                .and_then(|p| p.first().copied())
                .unwrap_or_else(|| game.board.iter().position(|&c| c == 0).unwrap());

            if game.current_turn == 1 {
                let hash = game.state_hash();
                brain.observe(Event::ContextWithState {
                    data: [0; 16],
                    state_hash: hash,
                });
                brain.observe(Event::Opcode {
                    opcode: ai_move as i64,
                    stack_depth: 0,
                });
            }
            
            game.step(ai_move).unwrap();
        }

        let w = game.check_winner();
        if w == 1 {
            x_wins += 1;
            brain.observe(Event::Reward(100));
        } else if w == 2 {
            o_wins += 1;
            brain.observe(Event::Error(VMError::Crash)); 
        } else {
            draws += 1;
            brain.observe(Event::Reward(50));
        }

        if g % 10 == 0 || g == num_games {
            print!(
                "\rSelf-Play: {}/{} ... (X Wins: {}, O Wins: {}, Draws: {})",
                g, num_games, x_wins, o_wins, draws
            );
            io::stdout().flush().unwrap();
        }
    }
    
    brain.sync();
    let _ = brain.save_to_file("plasticity.json");
    
    println!(
        "\n\n✅ Self-Play complete in {:.2}s!",
        start_time.elapsed().unwrap().as_secs_f32()
    );
}
use std::io::{self, Write, BufRead, BufReader};
use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio, Child};
use std::time::Instant;
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats,
};
use soulgain::plasticity::{Event, IntoOpcode, Plasticity};

// ─────────────────────────────────────────────────────────────────────────────
// The Oracle Wrapper
// ─────────────────────────────────────────────────────────────────────────────

struct Oracle {
    process: Child,
}

impl Oracle {
    fn new() -> Self {
        let child = Command::new("./stockfish_oracle")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to start stockfish_oracle");
        
        let mut oracle = Self { process: child };
        oracle.send("uci");
        oracle.send("isready");
        oracle.send("setoption name Skill Level value 20"); // Max Power
        oracle
    }

    fn send(&mut self, cmd: &str) {
        let stdin = self.process.stdin.as_mut().expect("Failed to open stdin");
        writeln!(stdin, "{}", cmd).unwrap();
    }

    fn get_best_move(&mut self, fen: &str) -> String {
        self.send(&format!("position fen {}", fen));
        self.send("go movetime 50"); 
        let stdout = self.process.stdout.as_mut().expect("Failed to open stdout");
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let l = line.unwrap();
            if l.starts_with("bestmove") {
                return l.split_whitespace().nth(1).unwrap_or("").to_string();
            }
        }
        "".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chess World & Action
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WrappedMove(pub ChessMove);

impl IntoOpcode for WrappedMove {
    fn into_opcode(self) -> i64 {
        let from = self.0.get_source().to_index() as i64;
        let to = self.0.get_dest().to_index() as i64;
        (from << 6) | to
    }
}

#[derive(Clone, Debug)]
struct ChessWorld { board: Board }

impl UniversalWorld for ChessWorld {
    type State = Board;
    type Action = WrappedMove;
    fn current_state(&self) -> Self::State { self.board.clone() }
    fn current_player(&self) -> i32 { if self.board.side_to_move() == Color::White { 1 } else { -1 } }
    fn step(&mut self, action: Self::Action) -> Result<(), ()> {
        self.board = self.board.make_move_new(action.0);
        Ok(())
    }
    fn is_terminal(&self) -> bool { self.board.status() != BoardStatus::Ongoing }
    fn evaluate_path(&self, path: &[Self::Action]) -> (f32, u64) {
        let mut sim = self.clone();
        for m in path { if sim.step(*m).is_err() { return (if sim.current_player() == 1 { -1.0 } else { 1.0 }, 1); } }
        let score: f32 = match sim.board.status() {
            BoardStatus::Checkmate => if sim.board.side_to_move() == Color::White { -1.0 } else { 1.0 },
            _ => {
                let mut sum = 0.0f32;
                for i in 0..64 {
                    let sq = unsafe { Square::new(i) };
                    if let Some(p) = sim.board.piece_on(sq) {
                        let val = match p { Piece::Pawn => 1.0, Piece::Knight | Piece::Bishop => 3.0, Piece::Rook => 5.0, Piece::Queen => 9.0, _ => 0.0 };
                        if sim.board.color_on(sq) == Some(Color::White) { sum += val; } else { sum -= val; }
                    }
                }
                (sum / 40.0).clamp(-1.0, 1.0)
            }
        };
        (score, path.len() as u64)
    }
}

struct ChessPolicy { brain: Plasticity }
impl CognitivePolicy<Board, WrappedMove> for ChessPolicy {
    fn evaluate(&self, _state: &Board) -> f32 { 0.0 }
    fn priors(&self, state: &Board) -> Vec<(WrappedMove, f32)> {
        let moves = MoveGen::new_legal(state).map(WrappedMove).collect::<Vec<_>>();
        if moves.is_empty() { return vec![]; }
        // 🌟 GET BRAIN FEEDBACK
        let mut dist = self.brain.get_op_distribution(Event::ContextWithState { data: [0; 16], state_hash: state.get_hash() }, &moves);
        
        // 🌟 FORCE CONVERGENCE: Amplify the Brain's favored moves
        for (_, weight) in dist.iter_mut() {
            *weight = (*weight * 10.0).exp(); // Softmax-ish temperature scaling
        }
        dist
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE MASTER GOD-FEAST
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_autoplay() {
    println!("\n🚀 STARTING GOD-FEAST (Forced Prior Convergence)...");
    let brain = Plasticity::new();
    if Path::new("chess_brain.json").exists() { let _ = brain.load_from_file("chess_brain.json"); }
    let policy = ChessPolicy { brain: brain.clone() };
    let mut oracle = Oracle::new(); 
    
    let config = ReasoningConfig::<WrappedMove> {
        simulations: 600, // 🌟 Increased depth of thought
        max_depth: 6, 
        max_program_len: 4, 
        max_ops_per_candidate: 4,
        exploration_constant: 1.0, // 🌟 Lowered to stick to "Divine" paths more
        length_penalty: 0.0, 
        loop_penalty: 0.0,
        action_space: vec![], 
        arena_capacity: 1_000_000,
    };

    let start_time = Instant::now();
    let mut total_moves = 0;
    let mut total_divine_matches = 0;

    print!("Games: "); io::stdout().flush().unwrap();
    let mut input = String::new(); io::stdin().read_line(&mut input).unwrap();
    let num_games: usize = input.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut game = ChessWorld { board: Board::default() };
        while !game.is_terminal() {
            let mut local_config = config.clone();
            local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
            let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
            let chosen_move = best_path.and_then(|p| p.first().copied()).unwrap();

            let oracle_move_str = oracle.get_best_move(&format!("{}", game.board));

            brain.observe(Event::ContextWithState { data: [0; 16], state_hash: game.board.get_hash() });
            brain.observe(Event::Opcode { opcode: chosen_move.into_opcode(), stack_depth: 0 });

            if chosen_move.0.to_string() == oracle_move_str {
                brain.observe(Event::Reward(250i16)); // 🌟 HUGE reward spike
                total_divine_matches += 1;
            } else {
                brain.observe(Event::Reward(-50i16)); // 🌟 HARSHER penalty for being "Lame"
            }

            game.step(chosen_move).unwrap();
            total_moves += 1;

            let alignment = (total_divine_matches as f32 / total_moves as f32) * 100.0;
            print!(
                "\r[FEAST] Game {} | Alignment: {:.1}% | MPS: {:.2}",
                g, alignment, total_moves as f64 / start_time.elapsed().as_secs_f64()
            );
            io::stdout().flush().unwrap();
        }
        
        // 🌩️ SYNC EVERY GAME
        brain.sync(); 
        if g % 2 == 0 { let _ = brain.save_to_file("chess_brain.json"); }
    }
    brain.sync();
    let _ = brain.save_to_file("chess_brain.json");
    println!("\n✅ Feast complete. SoulGain is ascending.");
}

pub fn run_self_match() {
    println!("\n🎬 SPECTATOR MATCH...");
    let brain = Plasticity::new();
    if Path::new("chess_brain.json").exists() { let _ = brain.load_from_file("chess_brain.json"); }
    let mut game = ChessWorld { board: Board::default() };
    let policy = ChessPolicy { brain: brain.clone() };

    let config = ReasoningConfig::<WrappedMove> {
        simulations: 3000, max_depth: 10, max_program_len: 8, max_ops_per_candidate: 8,
        exploration_constant: 1.0, length_penalty: 0.0, loop_penalty: 0.0,
        action_space: vec![], arena_capacity: 1_000_000,
    };

    while !game.is_terminal() {
        let mut local_config = config.clone();
        local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
        let (best_path, stats) = solve_universal_with_stats(&game, &local_config, &policy);
        if let Some(m) = best_path.and_then(|p| p.first().copied()) {
            println!("🤖 Move: {} | Sim: {} | Side: {}", m.0.to_string(), stats.simulations_run, if game.current_player() == 1 {"White"} else {"Black"});
            game.step(m).unwrap();
            println!("{}", game.board);
        } else { break; }
    }
}
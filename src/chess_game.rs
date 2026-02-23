use std::io::{self, Write, BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio, Child};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats,
};
use soulgain::intuition::{UniversalIntuition, ActionOutcome};

// ─────────────────────────────────────────────────────────────────────────────
// The Oracle Wrapper (Stockfish as the "Teacher")
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
        oracle.send("setoption name Skill Level value 20"); 
        oracle
    }

    fn send(&mut self, cmd: &str) {
        let stdin = self.process.stdin.as_mut().expect("Failed to open stdin");
        let _ = writeln!(stdin, "{}", cmd);
    }

    fn get_best_move(&mut self, fen: &str) -> String {
        self.send(&format!("position fen {}", fen));
        self.send("go depth 5"); 
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

    fn get_evaluation(&mut self, fen: &str) -> f64 {
        self.send(&format!("position fen {}", fen));
        self.send("go depth 5"); 
        let stdout = self.process.stdout.as_mut().expect("Failed to open stdout");
        let reader = BufReader::new(stdout);
        let mut final_score = 0.0;
        for line in reader.lines() {
            let l = line.unwrap();
            if l.contains("score cp") {
                let parts: Vec<&str> = l.split_whitespace().collect();
                if let Some(pos) = parts.iter().position(|&r| r == "cp") {
                    if let Some(val) = parts.get(pos + 1).and_then(|v| v.parse::<f64>().ok()) {
                        final_score = val / 100.0;
                    }
                }
            } else if l.contains("score mate") {
                final_score = if l.contains("score mate -") { -50.0 } else { 50.0 };
            }
            if l.starts_with("bestmove") { break; }
        }
        final_score
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chess World
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WrappedMove(pub ChessMove);

impl WrappedMove {
    fn into_action_id(self) -> i64 {
        let from = self.0.get_source().to_index() as i64;
        let to = self.0.get_dest().to_index() as i64;
        (from << 6) | to
    }
}

#[derive(Clone, Debug)]
struct ChessWorld { 
    board: Board,
    history: Vec<u64>, 
}

impl UniversalWorld for ChessWorld {
    type State = Board;
    type Action = WrappedMove;
    fn current_state(&self) -> Self::State { self.board.clone() }
    fn current_player(&self) -> i32 { if self.board.side_to_move() == Color::White { 1 } else { -1 } }
    
    fn step(&mut self, action: Self::Action) -> Result<(), ()> {
        self.board = self.board.make_move_new(action.0);
        self.history.push(self.board.get_hash()); 
        Ok(())
    }
    
    fn is_terminal(&self) -> bool { 
        self.board.status() != BoardStatus::Ongoing || 
        self.history.iter().filter(|&&h| h == *self.history.last().unwrap_or(&0)).count() >= 3 
    }
    
    fn evaluate_path(&self, path: &[Self::Action]) -> (f32, u64) {
        let mut sim = self.clone();
        for m in path { if sim.step(*m).is_err() { break; } }
        let mut sum = 0.0f32;
        for i in 0..64 {
            let sq = unsafe { Square::new(i) };
            if let Some(p) = sim.board.piece_on(sq) {
                let val = match p { Piece::Pawn => 1.0, Piece::Knight | Piece::Bishop => 3.0, Piece::Rook => 5.0, Piece::Queen => 9.0, _ => 0.0 };
                if sim.board.color_on(sq) == Some(Color::White) { sum += val; } else { sum -= val; }
            }
        }
        ((sum / 40.0).clamp(-1.0, 1.0), path.len() as u64)
    }
}

fn board_to_features(board: &Board) -> [u64; 16] {
    let mut data = [0u64; 16];
    let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];
    let mut idx = 0;
    for &color in &[Color::White, Color::Black] {
        let color_bb = board.color_combined(color);
        for &piece in &pieces {
            let piece_bb = board.pieces(piece);
            data[idx] = (*color_bb & *piece_bb).0; 
            idx += 1;
        }
    }
    data
}

struct ChessPolicy { brain: Arc<Mutex<UniversalIntuition>> }
impl CognitivePolicy<Board, WrappedMove> for ChessPolicy {
    fn evaluate(&self, _state: &Board) -> f32 { 0.0 }
    fn priors(&self, state: &Board) -> Vec<(WrappedMove, f32)> {
        let moves = MoveGen::new_legal(state).map(WrappedMove).collect::<Vec<_>>();
        if moves.is_empty() { return vec![]; }
        let mut brain_lock = self.brain.lock().unwrap();
        let ctx = brain_lock.build_context(board_to_features(state), None);
        let mut dist = brain_lock.get_action_distribution(&ctx, &moves, |m| m.into_action_id());
        for (_, weight) in dist.iter_mut() { *weight = (*weight * 10.0).exp(); }
        dist
    }
}

pub fn run_autoplay() {
    println!("\n🚀 STARTING GOD-FEAST (Teacher-Led Alignment)...");
    let mut brain = UniversalIntuition::new();
    if Path::new("chess_brain.bin").exists() { let _ = brain.load_from_file("chess_brain.bin"); }
    let brain_arc = Arc::new(Mutex::new(brain));
    let mut oracle = Oracle::new(); 
    
    let config = ReasoningConfig::<WrappedMove> {
        simulations: 500, max_depth: 6, max_program_len: 4, max_ops_per_candidate: 4,
        exploration_constant: 0.2, length_penalty: 0.0, loop_penalty: 1.0,
        action_space: vec![], arena_capacity: 1_000_000,
    };

    let start_time = Instant::now();
    let (mut total_moves, mut total_divine_matches, mut total_reward) = (0, 0, 0.0f64);

    print!("Games: "); io::stdout().flush().unwrap();
    let mut input = String::new(); io::stdin().read_line(&mut input).unwrap();
    let num_games: usize = input.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut game = ChessWorld { board: Board::default(), history: vec![Board::default().get_hash()] };
        while !game.is_terminal() {
            let mut local_config = config.clone();
            local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
            let (best_path, _) = solve_universal_with_stats(&game, &local_config, &ChessPolicy { brain: brain_arc.clone() });
            let chosen_move = best_path.and_then(|p| p.first().copied()).unwrap();
            
            // 🌟 THE TEACHER IS BACK
            let oracle_best_move = oracle.get_best_move(&format!("{}", game.board));
            let is_divine = chosen_move.0.to_string() == oracle_best_move;
            
            let mut sim_game = game.clone(); sim_game.step(chosen_move).unwrap();
            let eval = oracle.get_evaluation(&format!("{}", sim_game.board));
            let player_multiplier = if game.current_player() == 1 { 1.0 } else { -1.0 };
            
            // Reward = Position Quality + Alignment Bonus
            let quality_reward = eval * player_multiplier * 5.0;
            let final_reward = if is_divine { quality_reward + 20.0 } else { quality_reward - 10.0 };

            let mut b_lock = brain_arc.lock().unwrap();
            let ctx = b_lock.build_context(board_to_features(&game.board), None);
            b_lock.update_after_execution(chosen_move.into_action_id(), ActionOutcome {
                success: is_divine, // 🌟 Neural net now knows if it matched Stockfish
                reward_delta: final_reward, 
                used_tick: total_moves as u64,
                domain_tag: None, features: ctx.features, context_hash: ctx.context_hash,
            });
            drop(b_lock);

            if is_divine { total_divine_matches += 1; }
            game.step(chosen_move).unwrap();
            total_moves += 1; total_reward += final_reward;

            print!("\r[FEAST] Game {} | Align: {:.1}% | Avg Rew: {:.2} | MPS: {:.1}",
                g, (total_divine_matches as f32 / total_moves as f32) * 100.0, total_reward / total_moves as f64, total_moves as f64 / start_time.elapsed().as_secs_f64());
            io::stdout().flush().unwrap();
        }
        if g % 5 == 0 { let _ = brain_arc.lock().unwrap().save_to_file("chess_brain.bin"); }
    }
}

pub fn run_self_match() {
    let mut brain = UniversalIntuition::new();
    if Path::new("chess_brain.bin").exists() { let _ = brain.load_from_file("chess_brain.bin"); }
    let mut game = ChessWorld { board: Board::default(), history: vec![Board::default().get_hash()] };
    let policy = ChessPolicy { brain: Arc::new(Mutex::new(brain)) };

    while !game.is_terminal() {
        let config = ReasoningConfig::<WrappedMove> {
            simulations: 2500, max_depth: 12, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 0.2, length_penalty: 0.1, loop_penalty: 2.0,
            action_space: MoveGen::new_legal(&game.board).map(WrappedMove).collect(), arena_capacity: 1_000_000,
        };
        let (best_path, _) = solve_universal_with_stats(&game, &config, &policy);
        if let Some(m) = best_path.and_then(|p| p.first().copied()) {
            println!("🤖 Move: {} | Side: {:?}", m.0.to_string(), game.board.side_to_move());
            game.step(m).unwrap(); println!("{}", game.board);
        } else { break; }
    }
}
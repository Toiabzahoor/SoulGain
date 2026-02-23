use std::io::{self, Write, BufRead, BufReader, Read};
use std::fs::File;
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats,
    TaskSpec, TaskExample, VmWorld, UniversalPolicy
};
use soulgain::vm::{CoreMind, Op, StepStatus};
use soulgain::types::UVal;

struct Oracle {
    process: Child,
    cache: HashMap<u64, (String, f64)>, 
}

impl Oracle {
    fn new() -> Self {
        let child = Command::new("./stockfish_oracle")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to start stockfish_oracle");
        
        let mut oracle = Self { process: child, cache: HashMap::with_capacity(100_000) };
        oracle.send("uci");
        oracle.send("isready");
        oracle.send("setoption name Skill Level value 20"); 
        oracle
    }

    fn send(&mut self, cmd: &str) {
        if let Some(stdin) = self.process.stdin.as_mut() {
            let _ = writeln!(stdin, "{}", cmd);
        }
    }

    fn consult(&mut self, board: &Board) -> (String, f64) {
        let hash = board.get_hash();
        if let Some(res) = self.cache.get(&hash) {
            return res.clone();
        }

        let fen = format!("{}", board);
        self.send(&format!("position fen {}", fen));
        self.send("go depth 4"); 

        let mut final_score = 0.0;
        let mut best_move = String::new();

        if let Some(stdout) = self.process.stdout.as_mut() {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if line.contains("score cp") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if let Some(pos) = parts.iter().position(|&r| r == "cp") {
                            if let Some(val) = parts.get(pos + 1).and_then(|v| v.parse::<f64>().ok()) {
                                final_score = val / 100.0;
                            }
                        }
                    } else if line.contains("score mate") {
                        final_score = if line.contains("score mate -") { -50.0 } else { 50.0 };
                    }
                    
                    if line.starts_with("bestmove") {
                        best_move = line.split_whitespace().nth(1).unwrap_or("").to_string();
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        
        let res = (best_move, final_score);
        self.cache.insert(hash, res.clone());
        res
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WrappedMove(pub ChessMove);

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
                let val = match p { 
                    Piece::Pawn => 1.0, 
                    Piece::Knight | Piece::Bishop => 3.0,
                    Piece::Rook => 5.0, 
                    Piece::Queen => 9.0, 
                    _ => 0.0 
                };
                if sim.board.color_on(sq) == Some(Color::White) { sum += val; } else { sum -= val; }
            }
        }
        ((sum / 40.0).clamp(-1.0, 1.0), path.len() as u64)
    }
}

// RAW BOARD ENCODING: 64 squares * 7 piece types = 448 values
// Square encoding: empty=0, white_pawn=1, white_knight=2, ..., black_queen=13, black_king=14
// NO feature preprocessing. Let the program learn.
fn extract_raw_board(board: &Board) -> Vec<UVal> {
    let mut result = Vec::new();
    
    for i in 0..64 {
        let sq = unsafe { Square::new(i) };
        let piece_val = if let Some(p) = board.piece_on(sq) {
            let base = match p {
                Piece::Pawn => 1,
                Piece::Knight => 2,
                Piece::Bishop => 3,
                Piece::Rook => 4,
                Piece::Queen => 5,
                Piece::King => 6,
            };
            if board.color_on(sq) == Some(Color::White) {
                base as f64
            } else {
                -(base as f64)
            }
        } else {
            0.0
        };
        result.push(UVal::Number(piece_val));
    }
    
    // Add whose turn it is
    result.push(UVal::Number(if board.side_to_move() == Color::White { 1.0 } else { -1.0 }));
    
    result
}

struct ProgrammaticChessPolicy {
    evaluator_program: Arc<Mutex<Vec<Op>>>,
}

impl CognitivePolicy<Board, WrappedMove> for ProgrammaticChessPolicy {
    fn evaluate(&self, state: &Board) -> f32 {
        let program = self.evaluator_program.lock().unwrap().clone();
        let mut mind = CoreMind::new();
        let features = extract_raw_board(state);
        
        mind.reset(&features);
        
        let mut step_count = 0;
        while mind.ip() < program.len() && step_count < 256 {
            if mind.step(program[mind.ip()]) == StepStatus::Halt { break; }
            step_count += 1;
        }
        
        mind.extract_output().last()
            .and_then(|v| if let UVal::Number(n) = v { Some((*n as f32).clamp(-100.0, 100.0)) } else { None })
            .unwrap_or(0.0) 
    }

    fn priors(&self, state: &Board) -> Vec<(WrappedMove, f32)> {
        let moves = MoveGen::new_legal(state).map(WrappedMove).collect::<Vec<_>>();
        if moves.is_empty() { return vec![]; }
        let uniform = 1.0 / moves.len() as f32;
        moves.into_iter().enumerate().map(|(i, m)| {
            let noise = ((i * 13) % 11) as f32 * 0.001;
            (m, uniform + noise)
        }).collect()
    }
}

fn save_program(program: &[Op], path: &str) {
    if let Ok(mut file) = File::create(path) {
        let bytes: Vec<u8> = program.iter().map(|&op| op.as_i64() as u8).collect();
        let _ = file.write_all(&bytes);
    }
}

fn load_program(path: &str) -> Option<Vec<Op>> {
    if let Ok(mut file) = File::open(path) {
        let mut bytes = Vec::new();
        if file.read_to_end(&mut bytes).is_ok() {
            return Some(bytes.into_iter().filter_map(|b| Op::from_i64(b as i64)).collect());
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// REASONING-BASED LEARNING: Understand WHY, not memorize WHAT
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_autoplay() {
    println!("\n🧠 REASONING-BASED LEARNING (RAW BOARD STATE)");
    println!("   Program gets raw 64-square board as input");
    println!("   Must discover patterns: undefended pieces, king safety, etc.\n");
    
    let mut oracle = Oracle::new(); 
    let best_program_init = load_program("chess_brain.bin")
        .unwrap_or_else(|| vec![Op::Dup, Op::Gt, Op::Halt]);
    let program_arc = Arc::new(Mutex::new(best_program_init));
    let policy = ProgrammaticChessPolicy { evaluator_program: program_arc.clone() };

    let config = ReasoningConfig::<WrappedMove> {
        simulations: 100, 
        max_depth: 6, max_program_len: 4, max_ops_per_candidate: 4,
        exploration_constant: 1.5, 
        length_penalty: 0.0, loop_penalty: 1.0,
        action_space: vec![], arena_capacity: 1_000_000,
    };

    let mut total_evolutions = 0;
    let mut oracle_overrides = 0;

    print!("Games to play: "); 
    io::stdout().flush().unwrap();
    let mut input_text = String::new(); 
    io::stdin().read_line(&mut input_text).unwrap();
    let num_games: usize = input_text.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut game = ChessWorld { board: Board::default(), history: vec![Board::default().get_hash()] };
        let soul_color = if g % 2 == 1 { Color::White } else { Color::Black };
        
        let mut training_data = Vec::new();

        while !game.is_terminal() && game.history.len() < 150 { 
            if game.board.side_to_move() == soul_color {
                let mut local_config = config.clone();
                local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
                
                let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
                let chosen_move = best_path.and_then(|p| p.first().copied()).unwrap();
                
                // Check if this move is a catastrophic blunder
                let (sf_best_str, best_eval) = oracle.consult(&game.board);
                let after_move = game.board.make_move_new(chosen_move.0);
                let (_, opp_eval) = oracle.consult(&after_move);
                let actual_eval = -opp_eval;
                let eval_delta = best_eval - actual_eval;

                if eval_delta > 3.0 {
                    // CATASTROPHIC BLUNDER! Force Stockfish's move instead
                    println!("[G{}] ❌ CATASTROPHIC ({:.1}p)", g, eval_delta);
                    oracle_overrides += 1;
                    
                    if let Some(sf_move) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_best_str) {
                        game.step(WrappedMove(sf_move)).unwrap();
                    }
                } else {
                    // Valid move, record why it's good
                    let board_features = extract_raw_board(&game.board);
                    let move_quality = ((100.0 - (eval_delta * 10.0).abs()) / 100.0).clamp(0.0, 1.0);
                    training_data.push(TaskExample {
                        input: board_features,
                        expected_output: vec![UVal::Number(move_quality as f64)],
                    });
                    game.step(chosen_move).unwrap();
                }
            } else {
                let (sf_move_str, _) = oracle.consult(&game.board);
                if let Some(m) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_move_str) {
                    game.step(WrappedMove(m)).unwrap();
                } else {
                    let fallback = MoveGen::new_legal(&game.board).next().unwrap();
                    game.step(WrappedMove(fallback)).unwrap();
                }
            }
        }

        // After game: synthesize understanding from collected valid moves
        if !training_data.is_empty() && g % 2 == 0 {
            println!("\n[G{}] Synthesizing from {} valid moves (rest were oversights)", 
                g, training_data.len());
            
            let synthesis_task = TaskSpec { train_cases: training_data.clone() };
            let pristine_core = CoreMind::new();
            
            let vm_config = ReasoningConfig::<Op> {
                simulations: 2000,
                max_depth: 12, 
                max_program_len: 32, 
                max_ops_per_candidate: 128, 
                exploration_constant: 2.0,
                length_penalty: 0.0,
                loop_penalty: 0.3,
                action_space: vec![
                    Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Dup, Op::Drop, Op::Swap, Op::Over,
                    Op::Eq, Op::Gt, Op::Not, Op::Jmp, Op::JmpIf, Op::Literal, Op::Halt,
                    Op::And, Op::Or, Op::Xor, Op::Inc, Op::Dec, Op::Mod
                ],
                arena_capacity: 100_000,
            };
            
            let vm_world = VmWorld::new(pristine_core, &synthesis_task, &vm_config);
            let vm_policy = UniversalPolicy::from_task(&synthesis_task);
            
            let (new_program, stats) = solve_universal_with_stats(&vm_world, &vm_config, &vm_policy);
            
            if let Some(evolved) = new_program {
                if stats.best_score > 0.6 {
                    total_evolutions += 1;
                    *program_arc.lock().unwrap() = evolved.clone();
                    save_program(&evolved, "chess_brain.bin");
                    println!("   ✅ LEARNED (fit: {:.3}, len: {}, ops used reason)", 
                        stats.best_score, evolved.len());
                } else {
                    println!("   ⚠️  Synthesis weak (fit: {:.3}), keeping previous", stats.best_score);
                }
            }
        }

        if g % 5 == 0 {
            println!("\n📊 Progress: {} evolutions, {} blunders overridden", 
                total_evolutions, oracle_overrides);
        }
    }

    println!("\n🎓 COMPLETE");
    println!("   Evolutions: {}", total_evolutions);
    println!("   Blunders Caught: {}", oracle_overrides);
}

pub fn run_self_match() {
    println!("\n⚔️ SOULGAIN vs STOCKFISH\n");
    let best_program = load_program("chess_brain.bin").unwrap_or_else(|| vec![Op::Literal, Op::Halt]);
    let policy = ProgrammaticChessPolicy { evaluator_program: Arc::new(Mutex::new(best_program)) };
    let mut oracle = Oracle::new();
    
    let mut game = ChessWorld { board: Board::default(), history: vec![Board::default().get_hash()] };
    let soul_color = Color::White;

    while !game.is_terminal() {
        if game.board.side_to_move() == soul_color {
            let config = ReasoningConfig::<WrappedMove> {
                simulations: 2500, max_depth: 12, max_program_len: 8, max_ops_per_candidate: 8,
                exploration_constant: 0.2, length_penalty: 0.1, loop_penalty: 2.0,
                action_space: MoveGen::new_legal(&game.board).map(WrappedMove).collect(), arena_capacity: 1_000_000,
            };
            let (best_path, _) = solve_universal_with_stats(&game, &config, &policy);
            if let Some(m) = best_path.and_then(|p| p.first().copied()) {
                println!("🤖 SoulGain: {}", m.0.to_string());
                game.step(m).unwrap(); 
            } else { break; }
        } else {
            let (sf_move_str, _) = oracle.consult(&game.board);
            if let Some(m) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_move_str) {
                println!("🐟 Stockfish: {}", sf_move_str);
                game.step(WrappedMove(m)).unwrap();
            } else { break; }
        }
        println!("{}\n", game.board);
    }
    println!("Done: {:?}", game.board.status());
}
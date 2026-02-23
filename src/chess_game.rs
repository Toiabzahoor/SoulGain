use std::io::{self, Write, BufRead, BufReader, Read};
use std::fs::File;
use std::process::{Command, Stdio, Child};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats,
    TaskSpec, TaskExample, VmWorld, UniversalPolicy
};
use soulgain::vm::{CoreMind, Op, StepStatus};
use soulgain::types::UVal;

// ─────────────────────────────────────────────────────────────────────────────
// BLUNDER ANALYSIS: Episodic Memory of Disasters
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct DisasterRecord {
    position_before: Board,
    bad_move: ChessMove,
    oracle_best_move: ChessMove,
    eval_delta: f64,
    position_after: Board,
    position_should_have_been: Board,
    game_number: usize,
}

#[derive(Clone, Debug)]
struct BlunderPriorityBuffer {
    disasters: Vec<DisasterRecord>,
    max_size: usize,
}

impl BlunderPriorityBuffer {
    fn new(max_size: usize) -> Self {
        Self { disasters: Vec::new(), max_size }
    }

    fn record_blunder(&mut self, disaster: DisasterRecord) {
        self.disasters.push(disaster);
        
        if self.disasters.len() > self.max_size {
            self.disasters.sort_by(|a, b| {
                b.eval_delta.abs().partial_cmp(&a.eval_delta.abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
            self.disasters.truncate(self.max_size);
        }
    }

    fn get_critical_blunder(&self) -> Option<DisasterRecord> {
        self.disasters
            .iter()
            .max_by(|a, b| {
                a.eval_delta.abs().partial_cmp(&b.eval_delta.abs()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The Oracle Wrapper
// ─────────────────────────────────────────────────────────────────────────────

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
        if let Some(mut stdin) = self.process.stdin.as_mut() {
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

        if let Some(mut stdout) = self.process.stdout.as_mut() {
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

// ─────────────────────────────────────────────────────────────────────────────
// Chess World & Features
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

fn extract_topology_features(board: &Board) -> Vec<UVal> {
    let mut w_king_sq = None;
    let mut b_king_sq = None;
    
    for i in 0..64 {
        let sq = unsafe { Square::new(i) };
        if let Some(p) = board.piece_on(sq) {
            if p == Piece::King {
                if board.color_on(sq) == Some(Color::White) { w_king_sq = Some(sq); }
                else { b_king_sq = Some(sq); }
            }
        }
    }

    let w_king = w_king_sq.unwrap_or(unsafe { Square::new(4) }); 
    let b_king = b_king_sq.unwrap_or(unsafe { Square::new(60) }); 

    let w_k_r = w_king.get_rank().to_index() as i32;
    let w_k_f = w_king.get_file().to_index() as i32;
    let b_k_r = b_king.get_rank().to_index() as i32;
    let b_k_f = b_king.get_file().to_index() as i32;

    let mut mass_diff = 0.0f64;
    let mut center_diff = 0.0f64;
    let mut dev_diff = 0.0f64;
    let mut space_diff = 0.0f64;
    let mut w_danger = 0.0f64;
    let mut b_danger = 0.0f64;
    let mut w_shield = 0.0f64;
    let mut b_shield = 0.0f64;

    for i in 0..64 {
        let sq = unsafe { Square::new(i) };
        if let Some(p) = board.piece_on(sq) {
            let is_white = board.color_on(sq) == Some(Color::White);
            let r = sq.get_rank().to_index() as i32;
            let f = sq.get_file().to_index() as i32;
            
            let power = match p {
                Piece::Pawn => 1.0, 
                Piece::Knight | Piece::Bishop => 3.0,
                Piece::Rook => 5.0, 
                Piece::Queen => 9.0, 
                Piece::King => 0.0,
            };
            
            if is_white { mass_diff += power; } else { mass_diff -= power; }

            if (r == 3 || r == 4) && (f == 3 || f == 4) {
                if is_white { center_diff += 1.0; } else { center_diff -= 1.0; }
            }

            if p != Piece::Pawn && p != Piece::King {
                if is_white && r > 0 { dev_diff += 1.0; }
                if !is_white && r < 7 { dev_diff -= 1.0; }
            }

            if is_white && r >= 4 { space_diff += 1.0; }
            if !is_white && r <= 3 { space_diff -= 1.0; }

            if is_white {
                let dist = ((r - b_k_r).abs()).max((f - b_k_f).abs()) as f64;
                if dist <= 2.0 { b_danger += power / dist.max(1.0); }
                
                if p == Piece::Pawn {
                    let s_dist = ((r - w_k_r).abs()).max((f - w_k_f).abs()) as f64;
                    if s_dist <= 1.0 { w_shield += 1.0; }
                }
            } else {
                let dist = ((r - w_k_r).abs()).max((f - w_k_f).abs()) as f64;
                if dist <= 2.0 { w_danger += power / dist.max(1.0); }
                
                if p == Piece::Pawn {
                    let s_dist = ((r - b_k_r).abs()).max((f - b_k_f).abs()) as f64;
                    if s_dist <= 1.0 { b_shield += 1.0; }
                }
            }
        }
    }
    
    vec![
        UVal::Number(mass_diff),              
        UVal::Number(center_diff),          
        UVal::Number(dev_diff),                
        UVal::Number(space_diff), 
        UVal::Number(b_danger - w_danger),  
        UVal::Number(w_shield - b_shield),  
        UVal::Number(if board.side_to_move() == Color::White { 1.0 } else { -1.0 }),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// The Programmatic Chess Policy
// ─────────────────────────────────────────────────────────────────────────────

struct ProgrammaticChessPolicy {
    evaluator_program: Arc<Mutex<Vec<Op>>>,
}

impl CognitivePolicy<Board, WrappedMove> for ProgrammaticChessPolicy {
    fn evaluate(&self, state: &Board) -> f32 {
        let program = self.evaluator_program.lock().unwrap().clone();
        let mut mind = CoreMind::new();
        let features = extract_topology_features(state);
        
        mind.reset(&features);
        
        let mut step_count = 0;
        while mind.ip() < program.len() && step_count < 128 {
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

// ─────────────────────────────────────────────────────────────────────────────
// Memory Management
// ─────────────────────────────────────────────────────────────────────────────

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
// BLUNDER-PRIORITY TRAINING
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_autoplay() {
    println!("\n🔥 BLUNDER-PRIORITY LEARNING START");
    
    let mut oracle = Oracle::new(); 
    let best_program_init = load_program("chess_brain.bin").unwrap_or_else(|| vec![Op::Literal, Op::Halt]);
    let program_arc = Arc::new(Mutex::new(best_program_init));
    let policy = ProgrammaticChessPolicy { evaluator_program: program_arc.clone() };

    let mut blunder_buffer = BlunderPriorityBuffer::new(100);
    
    let config = ReasoningConfig::<WrappedMove> {
        simulations: 100, 
        max_depth: 6, max_program_len: 4, max_ops_per_candidate: 4,
        exploration_constant: 1.5, 
        length_penalty: 0.0, loop_penalty: 1.0,
        action_space: vec![], arena_capacity: 1_000_000,
    };

    let _start_time = Instant::now();
    let mut total_evolutions = 0;

    print!("Games to play: "); 
    io::stdout().flush().unwrap();
    let mut input_text = String::new(); 
    io::stdin().read_line(&mut input_text).unwrap();
    let num_games: usize = input_text.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut game = ChessWorld { board: Board::default(), history: vec![Board::default().get_hash()] };
        let soul_color = if g % 2 == 1 { Color::White } else { Color::Black };
        
        let mut soul_history: Vec<(Board, WrappedMove)> = Vec::new();

        while !game.is_terminal() && game.history.len() < 150 { 
            if game.board.side_to_move() == soul_color {
                let mut local_config = config.clone();
                local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
                
                let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
                let chosen_move = best_path.and_then(|p| p.first().copied()).unwrap();
                
                soul_history.push((game.board.clone(), chosen_move));
                game.step(chosen_move).unwrap();
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

        print!("\r[Game {}] Analyzing {} moves...      ", g, soul_history.len());
        io::stdout().flush().unwrap();
        
        let mut game_blunders = Vec::new();

        for (board_before, chosen_move) in &soul_history {
            let (sf_best_move_str, best_eval) = oracle.consult(&board_before);
            let board_after = board_before.make_move_new(chosen_move.0);
            let (_, opp_eval) = oracle.consult(&board_after);
            let actual_eval = -opp_eval;
            let eval_delta = best_eval - actual_eval;

            if eval_delta > 0.3 {
                if let Some(best_m) = MoveGen::new_legal(&board_before).find(|m| m.to_string() == sf_best_move_str) {
                    let board_should = board_before.make_move_new(best_m);
                    
                    game_blunders.push(DisasterRecord {
                        position_before: board_before.clone(),
                        bad_move: chosen_move.0,
                        oracle_best_move: best_m,
                        eval_delta,
                        position_after: board_after.clone(),
                        position_should_have_been: board_should,
                        game_number: g,
                    });
                }
            }
        }

        if !game_blunders.is_empty() {
            let worst = game_blunders.iter()
                .max_by(|a, b| a.eval_delta.partial_cmp(&b.eval_delta).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .clone();

            println!("\n💥 BLUNDER: {:.2} pawns | {} → {}", 
                worst.eval_delta, worst.bad_move.to_string(), worst.oracle_best_move.to_string());

            blunder_buffer.record_blunder(worst.clone());

            let mut training_examples = Vec::new();

            training_examples.push(TaskExample {
                input: extract_topology_features(&worst.position_before),
                expected_output: vec![UVal::Number(-0.95)],
            });

            let (_, best_opp_eval) = oracle.consult(&worst.position_should_have_been);
            let squashed_best = (-best_opp_eval / 10.0).clamp(-1.0, 1.0);
            
            training_examples.push(TaskExample {
                input: extract_topology_features(&worst.position_should_have_been),
                expected_output: vec![UVal::Number(squashed_best.max(0.5))],
            });

            if soul_history.len() >= 2 {
                if let Some(board_two_moves_back) = soul_history.get(soul_history.len().saturating_sub(2)).map(|(b, _)| b) {
                    training_examples.push(TaskExample {
                        input: extract_topology_features(&board_two_moves_back),
                        expected_output: vec![UVal::Number(-0.8)],
                    });
                }
            }

            let synthesis_task = TaskSpec { train_cases: training_examples };
            let pristine_core = CoreMind::new();
            
            let vm_config = ReasoningConfig::<Op> {
                simulations: 2000,
                max_depth: 10, 
                max_program_len: 32, 
                max_ops_per_candidate: 64, 
                exploration_constant: 1.41, 
                length_penalty: 0.0,
                loop_penalty: 0.5,
                action_space: vec![
                    Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Dup, Op::Drop, Op::Swap, Op::Over,
                    Op::Eq, Op::Gt, Op::Not, Op::Jmp, Op::JmpIf, Op::Literal, Op::Halt
                ],
                arena_capacity: 50_000,
            };
            
            let vm_world = VmWorld::new(pristine_core, &synthesis_task, &vm_config);
            let vm_policy = UniversalPolicy::from_task(&synthesis_task);
            
            let (new_program, stats) = solve_universal_with_stats(&vm_world, &vm_config, &vm_policy);
            
            if let Some(evolved) = new_program {
                total_evolutions += 1;
                *program_arc.lock().unwrap() = evolved.clone();
                save_program(&evolved, "chess_brain.bin");
                
                println!("✅ EVOLVED (fit: {:.2})", stats.best_score);
            } else {
                println!("⚠️  No improvement");
            }
        } else {
            println!("\n✨ Clean game");
        }

        if g % 5 == 0 {
            println!("📊 Stats: {} evolutions, {} disasters", total_evolutions, blunder_buffer.disasters.len());
            if let Some(worst) = blunder_buffer.get_critical_blunder() {
                println!("   Worst ever: {:.2} pawns", worst.eval_delta);
            }
        }
    }

    println!("\n🎓 DONE: {} games, {} evolutions, {} disasters learned", 
        num_games, total_evolutions, blunder_buffer.disasters.len());
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
use std::io::{self, Write, BufRead, BufReader};
use std::fs::{File, OpenOptions}; 
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher}; 

use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use serde::{Deserialize, Serialize};
use rand::Rng; 

use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats
};

// ─────────────────────────────────────────────────────────────────────────────
// THE ORACLE WRAPPER 
// ─────────────────────────────────────────────────────────────────────────────

struct Oracle {
    process: Child,
    cache: FxHashMap<u64, (String, f64)>, 
}

impl Oracle {
    fn new(skill_level: i32) -> Self {
        let child = Command::new("./stockfish_oracle") 
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to start stockfish_oracle");
        
        let mut oracle = Self { 
            process: child, 
            cache: FxHashMap::with_capacity_and_hasher(100_000, Default::default()) 
        };
        oracle.send("uci");
        oracle.send("isready");
        oracle.send(&format!("setoption name Skill Level value {}", skill_level)); 
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
        self.send("go depth 5"); 

        let mut final_score = 0.0;
        let mut best_move = String::new();

        if let Some(stdout) = self.process.stdout.as_mut() {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
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
            }
        }
        
        let res = (best_move, final_score);
        self.cache.insert(hash, res.clone());
        res
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CHESS WORLD & PURE RAW TOPOLOGY 
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WrappedMove(pub ChessMove);

#[derive(Clone)]
struct ChessWorld { 
    board: Board,
    history: Vec<u64>, 
    brain: Arc<Mutex<SoulBrain>>, 
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
        let mut sim_board = self.board.clone();
        for m in path { sim_board = sim_board.make_move_new(m.0); }
        
        let mut brain = self.brain.lock().unwrap();
        let hash = sim_board.get_hash();

        let (signature, feelings, _) = if let Some(cached) = brain.sensory_cache.get(&hash) {
            cached.clone()
        } else {
            let base_inputs = extract_topology_features(&sim_board);
            let result = brain.senses.get_sensory_signature(&base_inputs, None);
            brain.sensory_cache.insert(hash, result.clone());
            result
        };
        
        let score = brain.memory.evaluate_context(signature, &feelings);
        (score, path.len() as u64)
    }
}

fn extract_topology_features(board: &Board) -> [f32; 65] {
    let mut features = [0.0; 65];
    for i in 0..64 {
        let sq = unsafe { Square::new(i as u8) }; 
        if let Some(p) = board.piece_on(sq) {
            let type_val = match p {
                Piece::Pawn => 1.0, Piece::Knight => 2.0, Piece::Bishop => 3.0,
                Piece::Rook => 4.0, Piece::Queen => 5.0, Piece::King => 6.0,
            };
            features[i] = if board.color_on(sq) == Some(Color::White) { type_val } else { -type_val };
        }
    }
    features[64] = if board.side_to_move() == Color::White { 1.0 } else { -1.0 };
    features
}

// ─────────────────────────────────────────────────────────────────────────────
// THE COUNCIL OF MINISTERS (Biological Visual Cortex) 
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EyeOp {
    Add, Sub, Mul, Div, Gt,
    Abs, Min, Max, Sign,
    Pick, Drop,
    PushConst(f32),
    Read, 
    Halt,
}

#[derive(PartialEq)]
pub enum EyeStatus { Ok, Halt }

pub struct EyeMind {
    pub stack: [f32; 256], 
    pub stack_ptr: usize,
    pub ip: usize,
}

impl EyeMind {
    pub fn new() -> Self { Self { stack: [0.0; 256], stack_ptr: 0, ip: 0 } }
    
    #[inline(always)]
    pub fn push(&mut self, val: f32) {
        if self.stack_ptr < 256 {
            self.stack[self.stack_ptr] = val;
            self.stack_ptr += 1;
        }
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Option<f32> {
        if self.stack_ptr > 0 {
            self.stack_ptr -= 1;
            Some(self.stack[self.stack_ptr])
        } else {
            None
        }
    }

    pub fn step(&mut self, op: &EyeOp, board_box: &[f32; 65]) -> EyeStatus {
        self.ip += 1;
        match op {
            EyeOp::Add => { if let (Some(b), Some(a)) = (self.pop(), self.pop()) { self.push(a + b); } }
            EyeOp::Sub => { if let (Some(b), Some(a)) = (self.pop(), self.pop()) { self.push(a - b); } }
            EyeOp::Mul => { if let (Some(b), Some(a)) = (self.pop(), self.pop()) { self.push(a * b); } }
            EyeOp::Div => { if let (Some(b), Some(a)) = (self.pop(), self.pop()) { self.push(if b.abs() > 0.001 { a / b } else { 0.0 }); } }
            EyeOp::Gt => { if let (Some(b), Some(a)) = (self.pop(), self.pop()) { self.push(if a > b { 1.0 } else { -1.0 }); } }
            EyeOp::Abs => { if let Some(a) = self.pop() { self.push(a.abs()); } }
            EyeOp::Min => { if let (Some(b), Some(a)) = (self.pop(), self.pop()) { self.push(a.min(b)); } }
            EyeOp::Max => { if let (Some(b), Some(a)) = (self.pop(), self.pop()) { self.push(a.max(b)); } }
            EyeOp::Sign => { if let Some(a) = self.pop() { self.push(a.signum()); } }
            EyeOp::Drop => { self.pop(); }
            EyeOp::PushConst(val) => { self.push(*val); }
            EyeOp::Pick => {
                if let Some(idx_f) = self.pop() {
                    let len = self.stack_ptr;
                    if len > 0 {
                        let actual_idx = (idx_f.abs() as usize) % len;
                        let val = self.stack[actual_idx];
                        self.push(val);
                    }
                }
            }
            EyeOp::Read => { 
                if let Some(idx_f) = self.pop() {
                    let actual_idx = (idx_f.abs() as usize) % 65;
                    self.push(board_box[actual_idx]);
                }
            }
            EyeOp::Halt => return EyeStatus::Halt,
        }
        EyeStatus::Ok
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveredFeature {
    pub snippet: Vec<EyeOp>,
    pub cumulative_error: f32, // 🌸 Tracks how wrong the opinion is over 25 games
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeaturePool {
    pub features: Vec<DiscoveredFeature>,
    pub max_features: usize,
}

impl FeaturePool {
    pub fn new(max_features: usize) -> Self {
        let mut pool = Self { features: Vec::new(), max_features };
        for _ in 0..max_features { pool.add_random_hypothesis(); }
        pool
    }

    pub fn add_random_hypothesis(&mut self) {
        let mut rng = rand::thread_rng();
        let mut snippet = Vec::new();
        
        let len = rng.gen_range(10..=30); 
        for _ in 0..len {
            let op = match rng.gen_range(0..13) { 
                0 => EyeOp::Add, 1 => EyeOp::Sub, 2 => EyeOp::Mul, 3 => EyeOp::Div,
                4 => EyeOp::Gt, 5 => EyeOp::Abs, 6 => EyeOp::Min, 7 => EyeOp::Max,
                8 => EyeOp::Sign, 9 => EyeOp::Pick, 10 => EyeOp::Drop,
                11 => EyeOp::PushConst(rng.gen_range(-6.0_f32..=6.0_f32).round()), 
                12 => EyeOp::Read, 
                _ => unreachable!(),
            };
            snippet.push(op);
        }
        snippet.push(EyeOp::Halt);

        if self.features.len() < self.max_features {
            self.features.push(DiscoveredFeature { snippet, cumulative_error: 0.0 });
        }
    }

    pub fn get_sensory_signature(&self, board_box: &[f32; 65], ignore_index: Option<usize>) -> (u64, Vec<i64>, Vec<f32>) {
        let mut feelings = Vec::with_capacity(128);
        let mut opinions = Vec::with_capacity(128);
        
        for (i, feature) in self.features.iter().take(128).enumerate() {
            if Some(i) == ignore_index { continue; }

            let mut mind = EyeMind::new();
            
            let mut step_count = 0;
            while mind.ip < feature.snippet.len() && step_count < 64 {
                if mind.step(&feature.snippet[mind.ip], board_box) == EyeStatus::Halt { break; }
                step_count += 1;
            }

            let opinion = if mind.stack_ptr > 0 { mind.stack[mind.stack_ptr - 1] } else { 0.0 };
            opinions.push(opinion);
            
            let discrete_feeling = (opinion * 2.0).round() as i64;
            feelings.push(discrete_feeling); 
        }

        let mut hasher = FxHasher::default();
        feelings.hash(&mut hasher);
        let signature = hasher.finish();

        (signature, feelings, opinions)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TRUE EPISODIC MEMORY GRAPH 
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub experiences: FxHashMap<u64, f32>, 
    pub visit_counts: FxHashMap<u64, u32>,
    pub bucket_values: FxHashMap<u64, f32>,
    pub bucket_counts: FxHashMap<u64, u32>,
}

impl EpisodicMemory {
    pub fn new() -> Self {
        Self {
            experiences: FxHashMap::default(),
            visit_counts: FxHashMap::default(),
            bucket_values: FxHashMap::default(),
            bucket_counts: FxHashMap::default(),
        }
    }

    pub fn evaluate_context(&self, signature: u64, feelings: &[i64]) -> f32 {
        if let Some(&exact_val) = self.experiences.get(&signature) {
            return exact_val;
        }

        let mut total_familiarity = 0.0;
        let mut known_feelings = 0;

        for &feeling in feelings.iter() {
            let bucket_key = feeling as u64 & 0xFFFFFFFF;
            if let Some(&val) = self.bucket_values.get(&bucket_key) {
                total_familiarity += val;
                known_feelings += 1;
            }
        }

        if known_feelings > 0 {
            total_familiarity / (known_feelings as f32) 
        } else {
            0.0 
        }
    }

    pub fn get_feeling_belief(&self, feeling: i64) -> f32 {
        let bucket_key = feeling as u64 & 0xFFFFFFFF;
        *self.bucket_values.get(&bucket_key).unwrap_or(&0.0)
    }

    pub fn learn_context(&mut self, signature: u64, feelings: &[i64], target_eval: f32) {
        let count = self.visit_counts.entry(signature).or_insert(0);
        *count += 1;
        let current_val = self.experiences.entry(signature).or_insert(0.0);
        let learning_rate = 1.0 / (*count as f32).max(1.0);
        *current_val += learning_rate * (target_eval - *current_val);

        for &feeling in feelings.iter() {
            let bucket_key = feeling as u64 & 0xFFFFFFFF;
            let b_count = self.bucket_counts.entry(bucket_key).or_insert(0);
            *b_count += 1;
            
            let b_val = self.bucket_values.entry(bucket_key).or_insert(0.0);
            let b_lr = 1.0 / (*b_count as f32).max(1.0);
            *b_val += b_lr * (target_eval - *b_val);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE UNIFIED BRAIN 
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct SoulBrain {
    pub senses: FeaturePool,
    pub memory: EpisodicMemory,
    #[serde(skip)]
    pub sensory_cache: FxHashMap<u64, (u64, Vec<i64>, Vec<f32>)>, 
}

impl SoulBrain {
    pub fn save_and_merge(&self, path: &str) {
        let mut disk_brain = if let Ok(file) = File::open(path) {
            serde_json::from_reader(file).unwrap_or_else(|_| {
                SoulBrain {
                    senses: self.senses.clone(),
                    memory: EpisodicMemory::new(),
                    sensory_cache: FxHashMap::default(),
                }
            })
        } else {
            SoulBrain {
                senses: self.senses.clone(),
                memory: EpisodicMemory::new(),
                sensory_cache: FxHashMap::default(),
            }
        };

        for (&sig, &val) in &self.memory.experiences {
            disk_brain.memory.experiences.insert(sig, val);
        }
        for (&sig, &count) in &self.memory.visit_counts {
            let existing = disk_brain.memory.visit_counts.entry(sig).or_insert(0);
            *existing = (*existing).max(count); 
        }
        for (&b_key, &val) in &self.memory.bucket_values {
            disk_brain.memory.bucket_values.insert(b_key, val);
        }
        for (&b_key, &count) in &self.memory.bucket_counts {
            let existing = disk_brain.memory.bucket_counts.entry(b_key).or_insert(0);
            *existing = (*existing).max(count); 
        }

        disk_brain.senses = self.senses.clone();

        let temp_path = format!("{}.tmp", path);
        if let Ok(file) = File::create(&temp_path) {
            if serde_json::to_writer(file, &disk_brain).is_ok() {
                let _ = std::fs::rename(&temp_path, path);
            }
        }
    }

    pub fn load_binary(path: &str, max_features: usize) -> Self {
        if let Ok(file) = File::open(path) {
            match serde_json::from_reader::<_, SoulBrain>(file) {
                Ok(brain) => return brain,
                Err(_) => {
                    SoulBrain {
                        senses: FeaturePool::new(max_features),
                        memory: EpisodicMemory::new(),
                        sensory_cache: FxHashMap::default(),
                    }
                }
            }
        } else {
            SoulBrain {
                senses: FeaturePool::new(max_features),
                memory: EpisodicMemory::new(),
                sensory_cache: FxHashMap::default(),
            }
        }
    }
}

struct ProgrammaticChessPolicy {
    brain: Arc<Mutex<SoulBrain>>,
}

impl CognitivePolicy<Board, WrappedMove> for ProgrammaticChessPolicy {
    fn evaluate(&self, state: &Board) -> f32 {
        let mut b = self.brain.lock().unwrap(); 
        let hash = state.get_hash();

        let (sig, feelings, _) = if let Some(cached) = b.sensory_cache.get(&hash) {
            cached.clone()
        } else {
            let base_inputs = extract_topology_features(state);
            let result = b.senses.get_sensory_signature(&base_inputs, None);
            b.sensory_cache.insert(hash, result.clone());
            result
        };
        
        b.memory.evaluate_context(sig, &feelings)
    }

    fn priors(&self, state: &Board) -> Vec<(WrappedMove, f32)> {
        let moves = MoveGen::new_legal(state).map(WrappedMove).collect::<Vec<_>>();
        if moves.is_empty() { return vec![]; }
        let uniform = 1.0 / moves.len() as f32;
        
        let mut rng = rand::thread_rng();
        moves.into_iter().map(|m| {
            let noise = rng.gen_range(0.0..0.001);
            (m, uniform + noise)
        }).collect()
    }
}

struct GameMoment {
    fen: String,
    signature: u64,
    feelings: Vec<i64>,
    raw_opinions: Vec<f32>,
    oracle_eval: f32,
    current_eval: f32,
    surprise: f32,
}

fn log_trauma_to_file(mode: &str, game_num: usize, trauma_moments: &[GameMoment]) {
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open("soul_trauma_logs.txt") {
        let _ = writeln!(file, "\n==================================================");
        let _ = writeln!(file, "🌸 {} | Game {}", mode, game_num);
        let _ = writeln!(file, "==================================================");
        for (i, m) in trauma_moments.iter().enumerate() {
            let _ = writeln!(file, "Trauma #{} ── Surprise Level: {:.4}", i + 1, m.surprise);
            let _ = writeln!(file, "Truth (Oracle): {:>6.2} | Guess (SoulGain): {:>6.2}", m.oracle_eval, m.current_eval);
            let _ = writeln!(file, "FEN: {}", m.fen);
            let _ = writeln!(file, "--------------------------------------------------");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 🌸 MODE 1: SOULGAIN VS STOCKFISH TRAINING 
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_autoplay() {
    println!("\n🌸 EVOLUTIONARY SURVIVAL LOOP START (Council of Ministers)");
    
    // 🌸 Lowered Stockfish skill so SoulGain gets a chance to breathe
    let mut oracle = Oracle::new(5); 
    let brain = SoulBrain::load_binary("soul_memory.bin", 128); 
    let brain_arc = Arc::new(Mutex::new(brain));
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };

    print!("Games to play: "); 
    io::stdout().flush().unwrap();
    let mut input_text = String::new(); 
    io::stdin().read_line(&mut input_text).unwrap();
    let num_games: usize = input_text.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut game = ChessWorld { 
            board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc.clone() 
        };
        let soul_color = if g % 2 == 1 { Color::White } else { Color::Black };
        let mut peak_eval: f32 = -100.0; 
        
        let mut game_history_moments: Vec<GameMoment> = Vec::new();

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 400, max_depth: 10, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 1.5, length_penalty: 0.0, loop_penalty: 1.0,
            action_space: vec![], arena_capacity: 2_000_000,
        };

        while !game.is_terminal() && game.history.len() < 200 { 
            if game.board.side_to_move() == soul_color {
                let mut local_config = config.clone();
                local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
                
                let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
                let chosen_move = best_path.and_then(|p| p.first().copied()).expect("No legal moves found");
                game.step(chosen_move).unwrap();
            } else {
                let (sf_move_str, oracle_eval_raw) = oracle.consult(&game.board);
                
                let mut b = brain_arc.lock().unwrap();
                let base_inputs = extract_topology_features(&game.board);
                
                // 🌸 We grab the raw opinions to evaluate the ministers!
                let (signature, feelings, raw_opinions) = b.senses.get_sensory_signature(&base_inputs, None);
                
                let current_eval = b.memory.evaluate_context(signature, &feelings);
                
                let side_to_move = game.board.side_to_move();
                let absolute_eval = if side_to_move == Color::White { 
                    (oracle_eval_raw as f32).clamp(-100.0, 100.0) 
                } else { 
                    (-oracle_eval_raw as f32).clamp(-100.0, 100.0) 
                };

                let sg_perspective_eval = if soul_color == Color::White { absolute_eval } else { -absolute_eval };
                if sg_perspective_eval > peak_eval { peak_eval = sg_perspective_eval; }
                
                let surprise = (absolute_eval - current_eval).abs();

                let fen = format!("{}", game.board);
                game_history_moments.push(GameMoment {
                    fen, signature, feelings, raw_opinions,
                    oracle_eval: absolute_eval, current_eval, surprise
                });

                if let Some(m) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_move_str) {
                    game.step(WrappedMove(m)).unwrap();
                } else {
                    let fallback = MoveGen::new_legal(&game.board).next().unwrap();
                    game.step(WrappedMove(fallback)).unwrap();
                }
            }
        }

        game_history_moments.sort_by(|a, b| b.surprise.partial_cmp(&a.surprise).unwrap_or(std::cmp::Ordering::Equal));
        let trauma_moments = game_history_moments.into_iter().take(5).collect::<Vec<_>>();
        
        log_trauma_to_file("SG vs Stockfish", g, &trauma_moments);

        let mut b = brain_arc.lock().unwrap();
        let mut pruned_features = 0;

        for moment in &trauma_moments {
            b.memory.learn_context(moment.signature, &moment.feelings, moment.oracle_eval);
            
            // 🌸 CORRECT PRUNING LOGIC: Track minister inaccuracy over time based on their raw opinion
            for (i, &opinion) in moment.raw_opinions.iter().enumerate() {
                let error = (opinion - moment.oracle_eval).abs();
                b.senses.features[i].cumulative_error += error; 
            }
        }

        // 🌸 We only prune the worst minister every 25 games, giving them plenty of time to be evaluated!
        if g % 25 == 0 {
            let mut worst_idx = 0;
            let mut max_err = -1.0;
            for (i, feature) in b.senses.features.iter().enumerate() {
                if feature.cumulative_error > max_err {
                    max_err = feature.cumulative_error;
                    worst_idx = i;
                }
            }

            b.senses.features.remove(worst_idx);
            b.senses.add_random_hypothesis();
            pruned_features += 1;

            // Reset everyone's error for the next 25-game horizon!
            for f in &mut b.senses.features { f.cumulative_error = 0.0; }
            b.sensory_cache.clear(); 
        }

        b.save_and_merge("soul_memory.bin");
        let actual_length = game.history.len();
        println!("Game {} | Len: {} | Peak: {:.2} | Pruned: {}", g, actual_length, peak_eval, pruned_features);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 🌸 MODE 2: SOULGAIN VS STOCKFISH EXHIBITION 
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_self_match() {
    println!("\n⚔️ SOULGAIN vs STOCKFISH\n");
    let brain = SoulBrain::load_binary("soul_memory.bin", 128); 
    let brain_arc = Arc::new(Mutex::new(brain));
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };
    let mut oracle = Oracle::new(10); 
    
    let mut game = ChessWorld { 
        board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc.clone() 
    };
    let soul_color = Color::White;

    let mut game_history_moments: Vec<GameMoment> = Vec::new();

    while !game.is_terminal() {
        if game.board.side_to_move() == soul_color {
            let config = ReasoningConfig::<WrappedMove> {
                simulations: 80000, max_depth: 20, max_program_len: 8, max_ops_per_candidate: 8,
                exploration_constant: 0.5, length_penalty: 0.1, loop_penalty: 2.0,
                action_space: MoveGen::new_legal(&game.board).map(WrappedMove).collect(), arena_capacity: 1_000_000,
            };
            let (best_path, _) = solve_universal_with_stats(&game, &config, &policy);
            if let Some(m) = best_path.and_then(|p| p.first().copied()) {
                println!("🤖 SoulGain: {}", m.0.to_string());
                game.step(m).unwrap(); 
            } else { break; }
        } else {
            let (sf_move_str, oracle_eval_raw) = oracle.consult(&game.board);
            
            let mut b = brain_arc.lock().unwrap();
            let base_inputs = extract_topology_features(&game.board);
            let (signature, feelings, raw_opinions) = b.senses.get_sensory_signature(&base_inputs, None);
            let current_eval = b.memory.evaluate_context(signature, &feelings);
            
            let side_to_move = game.board.side_to_move();
            let absolute_eval = if side_to_move == Color::White { (oracle_eval_raw as f32).clamp(-100.0, 100.0) } 
                                else { (-oracle_eval_raw as f32).clamp(-100.0, 100.0) };
            
            let surprise = (absolute_eval - current_eval).abs();
            game_history_moments.push(GameMoment {
                fen: format!("{}", game.board), signature, feelings, raw_opinions,
                oracle_eval: absolute_eval, current_eval, surprise
            });

            if let Some(m) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_move_str) {
                println!("🐟 Stockfish: {}", sf_move_str);
                game.step(WrappedMove(m)).unwrap();
            } else { break; }
        }
        println!("{}\n", game.board);
    }
    game_history_moments.sort_by(|a, b| b.surprise.partial_cmp(&a.surprise).unwrap_or(std::cmp::Ordering::Equal));
    let trauma_moments = game_history_moments.into_iter().take(5).collect::<Vec<_>>();
    log_trauma_to_file("Exhibition vs Stockfish", 0, &trauma_moments);
    
    println!("Match finished! Status: {:?}", game.board.status());
}

// ─────────────────────────────────────────────────────────────────────────────
// 🌸 MODE 3: SOULGAIN VS SOULGAIN (Self-Play Sandbox Training!)
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_sg_vs_sg_training() {
    println!("\n🌸 SOULGAIN VS SOULGAIN: SELF-PLAY SANDBOX START");
    
    let mut oracle = Oracle::new(5); 
    let brain = SoulBrain::load_binary("soul_memory.bin", 128);
    let brain_arc = Arc::new(Mutex::new(brain));
    
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };

    print!("Games to play: "); 
    io::stdout().flush().unwrap();
    let mut input_text = String::new(); 
    io::stdin().read_line(&mut input_text).unwrap();
    let num_games: usize = input_text.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut game = ChessWorld { 
            board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc.clone() 
        };
        
        let mut game_history_moments: Vec<GameMoment> = Vec::new();

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 4000, max_depth: 12, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 1.5, length_penalty: 0.0, loop_penalty: 1.0,
            action_space: vec![], arena_capacity: 2_000_000,
        };

        while !game.is_terminal() && game.history.len() < 200 { 
            let (_, oracle_eval_raw) = oracle.consult(&game.board);
            let side_to_move = game.board.side_to_move();
            
            let absolute_eval = if side_to_move == Color::White {
                (oracle_eval_raw as f32).clamp(-100.0, 100.0)
            } else {
                (-oracle_eval_raw as f32).clamp(-100.0, 100.0)
            };

            {
                let mut b = brain_arc.lock().unwrap();
                let base_inputs = extract_topology_features(&game.board);
                let (signature, feelings, raw_opinions) = b.senses.get_sensory_signature(&base_inputs, None);
                
                let current_eval = b.memory.evaluate_context(signature, &feelings);
                let surprise = (absolute_eval - current_eval).abs();

                let fen = format!("{}", game.board);
                game_history_moments.push(GameMoment {
                    fen, signature, feelings, raw_opinions,
                    oracle_eval: absolute_eval, current_eval, surprise
                });
            }

            let mut local_config = config.clone();
            local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
            
            let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
            let chosen_move = best_path.and_then(|p| p.first().copied()).expect("No legal moves found");
            
            game.step(chosen_move).unwrap();
        }

        game_history_moments.sort_by(|a, b| b.surprise.partial_cmp(&a.surprise).unwrap_or(std::cmp::Ordering::Equal));
        let trauma_moments = game_history_moments.into_iter().take(5).collect::<Vec<_>>();
        
        log_trauma_to_file("SG vs SG Sandbox", g, &trauma_moments);

        let mut b = brain_arc.lock().unwrap();
        let mut pruned_features = 0;

        for moment in &trauma_moments {
            b.memory.learn_context(moment.signature, &moment.feelings, moment.oracle_eval);
            
            // 🌸 CORRECT PRUNING LOGIC
            for (i, &opinion) in moment.raw_opinions.iter().enumerate() {
                let error = (opinion - moment.oracle_eval).abs();
                b.senses.features[i].cumulative_error += error; 
            }
        }

        if g % 25 == 0 {
            let mut worst_idx = 0;
            let mut max_err = -1.0;
            for (i, feature) in b.senses.features.iter().enumerate() {
                if feature.cumulative_error > max_err {
                    max_err = feature.cumulative_error;
                    worst_idx = i;
                }
            }

            b.senses.features.remove(worst_idx);
            b.senses.add_random_hypothesis();
            pruned_features += 1;

            for f in &mut b.senses.features { f.cumulative_error = 0.0; }
            b.sensory_cache.clear(); 
        }

        b.save_and_merge("soul_memory.bin");
        println!("Sandbox Game {} | Len: {} | Pruned: {}", g, game.history.len(), pruned_features);
    }
}

// 🌸 THE AUTOMATED HEADLESS VERSION
pub fn run_sg_vs_sg_training_automated(num_games: usize) {
    println!("\n🌸 HEADLESS SANDBOX START: {} Games", num_games);
    
    let mut oracle = Oracle::new(5); 
    let brain = SoulBrain::load_binary("soul_memory.bin", 128);
    let brain_arc = Arc::new(Mutex::new(brain));
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };

    for g in 1..=num_games {
        let mut game = ChessWorld { 
            board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc.clone() 
        };
        
        let mut game_history_moments: Vec<GameMoment> = Vec::new();

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 400, max_depth: 10, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 1.5, length_penalty: 0.0, loop_penalty: 1.0,
            action_space: vec![], arena_capacity: 2_000_000,
        };

        while !game.is_terminal() && game.history.len() < 200 { 
            let (_, oracle_eval_raw) = oracle.consult(&game.board);
            let side_to_move = game.board.side_to_move();
            
            let absolute_eval = if side_to_move == Color::White { 
                (oracle_eval_raw as f32).clamp(-100.0, 100.0) 
            } else { 
                (-oracle_eval_raw as f32).clamp(-100.0, 100.0) 
            };

            {
                let mut b = brain_arc.lock().unwrap();
                let base_inputs = extract_topology_features(&game.board);
                let (signature, feelings, raw_opinions) = b.senses.get_sensory_signature(&base_inputs, None);
                
                let current_eval = b.memory.evaluate_context(signature, &feelings);
                let surprise = (absolute_eval - current_eval).abs();

                let fen = format!("{}", game.board);
                game_history_moments.push(GameMoment {
                    fen, signature, feelings, raw_opinions, 
                    oracle_eval: absolute_eval, current_eval, surprise
                });
            }

            let mut local_config = config.clone();
            local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
            let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
            let chosen_move = best_path.and_then(|p| p.first().copied()).expect("No moves");
            game.step(chosen_move).unwrap();
        }

        game_history_moments.sort_by(|a, b| b.surprise.partial_cmp(&a.surprise).unwrap_or(std::cmp::Ordering::Equal));
        let trauma_moments = game_history_moments.into_iter().take(5).collect::<Vec<_>>();
        
        log_trauma_to_file("HEADLESS Sandbox", g, &trauma_moments);

        let mut b = brain_arc.lock().unwrap();
        for moment in &trauma_moments {
            b.memory.learn_context(moment.signature, &moment.feelings, moment.oracle_eval);
            for (i, &opinion) in moment.raw_opinions.iter().enumerate() {
                let error = (opinion - moment.oracle_eval).abs();
                b.senses.features[i].cumulative_error += error; 
            }
        }

        let mut pruned = 0; 
        if g % 25 == 0 {
            let mut worst_idx = 0;
            let mut max_err = -1.0;
            for (i, feature) in b.senses.features.iter().enumerate() {
                if feature.cumulative_error > max_err {
                    max_err = feature.cumulative_error;
                    worst_idx = i;
                }
            }
            b.senses.features.remove(worst_idx);
            b.senses.add_random_hypothesis();
            pruned = 1;

            for f in &mut b.senses.features { f.cumulative_error = 0.0; }
            b.sensory_cache.clear(); 
        }

        b.save_and_merge("soul_memory.bin");
        println!("🤖 Game {}/{} | Len: {} | Pruned: {}", g, num_games, game.history.len(), pruned);
    }
    println!("✅ Automated Training Complete.");
}

// ─────────────────────────────────────────────────────────────────────────────
// 🌸 MODE 4: SOULGAIN VS SOULGAIN EXHIBITION
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_sg_vs_sg_exhibition() {
    println!("\n⚔️ SOULGAIN vs SOULGAIN (Exhibition Match)\n");
    let brain = SoulBrain::load_binary("soul_memory.bin", 128); 
    let brain_arc = Arc::new(Mutex::new(brain));
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };
    
    let mut game = ChessWorld { 
        board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc.clone() 
    };

    while !game.is_terminal() {
        let color_name = if game.board.side_to_move() == Color::White { "White" } else { "Black" };

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 800000, max_depth: 20, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 0.5, length_penalty: 0.1, loop_penalty: 2.0,
            action_space: MoveGen::new_legal(&game.board).map(WrappedMove).collect(), arena_capacity: 1_000_000,
        };
        
        let (best_path, _) = solve_universal_with_stats(&game, &config, &policy);
        
        if let Some(m) = best_path.and_then(|p| p.first().copied()) {
            println!("🤖 SoulGain ({}): {}", color_name, m.0.to_string());
            game.step(m).unwrap(); 
        } else { 
            break; 
        }
        println!("{}\n", game.board);
    }
    println!("Match finished! Status: {:?}", game.board.status());
}
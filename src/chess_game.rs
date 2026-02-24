use std::io::{self, Write, BufRead, BufReader};
use std::fs::File;
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use serde::{Deserialize, Serialize};
use rand::Rng; // 🌸 True randomness for evolution!

use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats
};

// ─────────────────────────────────────────────────────────────────────────────
// THE ORACLE WRAPPER (Depth 10 Stress Test 🌸)
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
        
        self.send("go depth 12"); 

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
        let mut sim = self.clone();
        for m in path { if sim.step(*m).is_err() { break; } }
        
        let brain = self.brain.lock().unwrap();
        let base_inputs = extract_topology_features(&sim.board);
        
        let (signature, feelings) = brain.senses.get_sensory_signature(&base_inputs, None);
        let score = brain.memory.evaluate_context(signature, &feelings);
        
        (score, path.len() as u64)
    }
}

// 🌸 REWRITTEN: 100% Raw Board Geometry mapped strictly to floats
fn extract_topology_features(board: &Board) -> Vec<f32> {
    let mut features = Vec::with_capacity(65);
    
    for i in 0..64 {
        let sq = unsafe { Square::new(i) };
        let val = match board.piece_on(sq) {
            Some(p) => {
                let type_val = match p {
                    Piece::Pawn => 1.0, 
                    Piece::Knight => 2.0,
                    Piece::Bishop => 3.0,
                    Piece::Rook => 4.0, 
                    Piece::Queen => 5.0, 
                    Piece::King => 6.0,
                };
                if board.color_on(sq) == Some(Color::White) { type_val } else { -type_val }
            },
            None => 0.0,
        };
        features.push(val);
    }
    
    features.push(if board.side_to_move() == Color::White { 1.0 } else { -1.0 });
    features
}

// ─────────────────────────────────────────────────────────────────────────────
// THE BIOLOGICAL VISUAL CORTEX (Custom PushGP Eyes) 🌸
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EyeOp {
    Add, Sub, Mul, Div, Gt,
    Abs, Min, Max, Sign,
    Pick, Drop,
    PushConst(f32),
    Halt,
}

#[derive(PartialEq)]
pub enum EyeStatus { Ok, Halt }

pub struct EyeMind {
    pub stack: Vec<f32>,
    pub ip: usize,
}

impl EyeMind {
    pub fn new() -> Self { Self { stack: Vec::with_capacity(256), ip: 0 } }
    
    pub fn reset(&mut self, inputs: &[f32]) {
        self.stack.clear();
        self.stack.extend_from_slice(inputs);
        self.ip = 0;
    }

    pub fn step(&mut self, op: &EyeOp) -> EyeStatus {
        self.ip += 1;
        match op {
            EyeOp::Add => { if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) { self.stack.push(a + b); } }
            EyeOp::Sub => { if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) { self.stack.push(a - b); } }
            EyeOp::Mul => { if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) { self.stack.push(a * b); } }
            EyeOp::Div => { if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) { self.stack.push(if b.abs() > 0.001 { a / b } else { 0.0 }); } }
            EyeOp::Gt => { if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) { self.stack.push(if a > b { 1.0 } else { -1.0 }); } }
            EyeOp::Abs => { if let Some(a) = self.stack.pop() { self.stack.push(a.abs()); } }
            EyeOp::Min => { if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) { self.stack.push(a.min(b)); } }
            EyeOp::Max => { if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) { self.stack.push(a.max(b)); } }
            EyeOp::Sign => { if let Some(a) = self.stack.pop() { self.stack.push(a.signum()); } }
            EyeOp::Drop => { self.stack.pop(); }
            EyeOp::PushConst(val) => { self.stack.push(*val); }
            EyeOp::Pick => {
                if let Some(idx_f) = self.stack.pop() {
                    let len = self.stack.len();
                    if len > 0 {
                        let actual_idx = (idx_f.abs() as usize) % len;
                        let val = self.stack[actual_idx];
                        self.stack.push(val);
                    }
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
    pub reliability: f32,
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
        
        let len = rng.gen_range(10..=23); 
        for _ in 0..len {
            let op = match rng.gen_range(0..12) {
                0 => EyeOp::Add,
                1 => EyeOp::Sub,
                2 => EyeOp::Mul,
                3 => EyeOp::Div,
                4 => EyeOp::Gt,
                5 => EyeOp::Abs,
                6 => EyeOp::Min,
                7 => EyeOp::Max,
                8 => EyeOp::Sign,
                9 => EyeOp::Pick,
                10 => EyeOp::Drop,
                11 => EyeOp::PushConst(rng.gen_range(-6.0_f32..=6.0_f32).round()), 
                _ => unreachable!(),
            };
            snippet.push(op);
        }
        snippet.push(EyeOp::Halt);

        if self.features.len() < self.max_features {
            self.features.push(DiscoveredFeature { snippet, reliability: 1.0 });
        }
    }

    // 🌸 128 EYES HASH SIGNATURE: Safely condenses the massive vision into a single u64
    pub fn get_sensory_signature(&self, base_inputs: &[f32], ignore_index: Option<usize>) -> (u64, Vec<i64>) {
        let mut feelings = Vec::with_capacity(128);
        
        for (i, feature) in self.features.iter().take(128).enumerate() {
            if Some(i) == ignore_index { continue; }

            let mut mind = EyeMind::new();
            mind.reset(base_inputs);
            
            let mut step_count = 0;
            while mind.ip < feature.snippet.len() && step_count < 64 {
                if mind.step(&feature.snippet[mind.ip]) == EyeStatus::Halt { break; }
                step_count += 1;
            }

            let output_val = mind.stack.last().copied().unwrap_or(0.0);
            let discrete_feeling = (output_val * 2.0).round() as i64;
            feelings.push(discrete_feeling); 
        }

        let mut hasher = DefaultHasher::new();
        feelings.hash(&mut hasher);
        let signature = hasher.finish();

        (signature, feelings)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TRUE EPISODIC MEMORY GRAPH (With the Universal Amygdala! 🌸)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub experiences: HashMap<u64, f32>, // 🌸 Changed to u64 to support the 128-eye Hash!
    pub visit_counts: HashMap<u64, u32>,
    pub bucket_values: HashMap<u64, f32>,
    pub bucket_counts: HashMap<u64, u32>,
}

impl EpisodicMemory {
    pub fn new() -> Self {
        Self {
            experiences: HashMap::new(),
            visit_counts: HashMap::new(),
            bucket_values: HashMap::new(),
            bucket_counts: HashMap::new(),
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
}

impl SoulBrain {
    pub fn save_and_merge(&self, path: &str) {
        let mut disk_brain = if let Ok(file) = File::open(path) {
            serde_json::from_reader(file).unwrap_or_else(|_| {
                SoulBrain {
                    senses: self.senses.clone(),
                    memory: EpisodicMemory::new(),
                }
            })
        } else {
            SoulBrain {
                senses: self.senses.clone(),
                memory: EpisodicMemory::new(),
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
            match serde_json::from_reader(file) {
                Ok(brain) => return brain,
                Err(_) => {
                    println!("🌱 Reincarnating Soul into 128-Eye Architecture. Starting fresh memory!");
                    SoulBrain {
                        senses: FeaturePool::new(max_features),
                        memory: EpisodicMemory::new(),
                    }
                }
            }
        } else {
            println!("🌱 No previous memory found. Birthing new Soul...");
            SoulBrain {
                senses: FeaturePool::new(max_features),
                memory: EpisodicMemory::new(),
            }
        }
    }
}

struct ProgrammaticChessPolicy {
    brain: Arc<Mutex<SoulBrain>>,
}

impl CognitivePolicy<Board, WrappedMove> for ProgrammaticChessPolicy {
    fn evaluate(&self, state: &Board) -> f32 {
        let b = self.brain.lock().unwrap(); 
        let base_inputs = extract_topology_features(state);
        let (sig, feelings) = b.senses.get_sensory_signature(&base_inputs, None);
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

// ─────────────────────────────────────────────────────────────────────────────
// 🌸 MODE 1: SOULGAIN VS STOCKFISH TRAINING (128 EYES / HYBRID PRUNING)
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_autoplay() {
    println!("\n🌸 EVOLUTIONARY SURVIVAL LOOP START (128 Eyes | Hybrid Reservoir)");
    
    let mut oracle = Oracle::new(); 
    let brain = SoulBrain::load_binary("soul_memory.bin", 128); // 🌸 Birthing 128 eyes!
    let brain_arc = Arc::new(Mutex::new(brain));
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };

    const TARGET_SURVIVAL: f32 = 40.0;
    const TRAUMA_WEIGHT: f32 = 0.5;

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
        let mut total_surprisal = 0.0;
        let mut peak_eval: f32 = -1.0; 
        let mut feature_histories: Vec<Vec<i64>> = vec![Vec::new(); 128]; // 🌸 Tracking all 128!

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 8000, max_depth: 16, max_program_len: 8, max_ops_per_candidate: 8,
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
                let (signature, feelings) = b.senses.get_sensory_signature(&base_inputs, None);
                
                for (i, &feeling) in feelings.iter().enumerate() {
                    if i < feature_histories.len() { feature_histories[i].push(feeling); }
                }

                let current_eval = b.memory.evaluate_context(signature, &feelings);
                let oracle_eval = if soul_color == Color::White {
                    (oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0)
                } else {
                    (-oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0)
                };

                if oracle_eval > peak_eval { peak_eval = oracle_eval; }
                total_surprisal += (oracle_eval - current_eval).abs();
                b.memory.learn_context(signature, &feelings, oracle_eval);

                if let Some(m) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_move_str) {
                    game.step(WrappedMove(m)).unwrap();
                } else {
                    let fallback = MoveGen::new_legal(&game.board).next().unwrap();
                    game.step(WrappedMove(fallback)).unwrap();
                }
            }
        }

        let actual_length = game.history.len() as f32;
        let prediction_surprise = total_surprisal / actual_length.max(1.0);
        let survival_penalty = if actual_length < TARGET_SURVIVAL {
            ((TARGET_SURVIVAL - actual_length) / TARGET_SURVIVAL) * TRAUMA_WEIGHT
        } else { 0.0 };

        let final_avg_surprise = prediction_surprise + survival_penalty;

        let mut b = brain_arc.lock().unwrap();
        let mut pruned_features = 0;

        // 🌸 HYBRID RESERVOIR PRUNING: Only prune every 10th game, and ONLY 1 eye max!
        if g % 10 == 0 {
            let mut kill_list = Vec::new();
            let mut unique_counts = Vec::new();

            for i in 0..128 {
                if i >= b.senses.features.len() { break; }
                let history = &feature_histories[i];
                if history.is_empty() { continue; }
                let mut unique_vals = history.clone();
                unique_vals.sort(); unique_vals.dedup();
                unique_counts.push((i, unique_vals.len()));
                if unique_vals.len() <= 1 { kill_list.push(i); }
            }

            if kill_list.is_empty() && final_avg_surprise > 0.20 {
                unique_counts.sort_by_key(|&(_, count)| count);
                if let Some(&(idx, _)) = unique_counts.first() { kill_list.push(idx); }
            }

            kill_list.sort(); kill_list.dedup(); 
            kill_list.truncate(1); // 🌸 Extreme stability limit!
            
            for &idx in kill_list.iter().rev() {
                b.senses.features.remove(idx);
                b.senses.add_random_hypothesis();
                pruned_features += 1;
            }
        }

        b.save_and_merge("soul_memory.bin");
        println!("Game {} | Len: {} | Peak: {:.2} | Surprise: {:.4} | Pruned: {}", 
            g, actual_length, peak_eval, final_avg_surprise, pruned_features);
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
    let mut oracle = Oracle::new();
    
    let mut game = ChessWorld { 
        board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc 
    };
    let soul_color = Color::White;

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
            let (sf_move_str, _) = oracle.consult(&game.board);
            if let Some(m) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_move_str) {
                println!("🐟 Stockfish: {}", sf_move_str);
                game.step(WrappedMove(m)).unwrap();
            } else { break; }
        }
        println!("{}\n", game.board);
    }
    println!("Match finished! Status: {:?}", game.board.status());
}

// ─────────────────────────────────────────────────────────────────────────────
// 🌸 MODE 3: SOULGAIN VS SOULGAIN (Self-Play Sandbox Training!)
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_sg_vs_sg_training() {
    println!("\n🌸 SOULGAIN VS SOULGAIN: SELF-PLAY SANDBOX START (128 Eyes)");
    
    let mut oracle = Oracle::new(); 
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
        
        let mut total_surprisal = 0.0;
        let mut feature_histories: Vec<Vec<i64>> = vec![Vec::new(); 128]; // 🌸 Track all 128!

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 8000, max_depth: 16, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 1.5, length_penalty: 0.0, loop_penalty: 1.0,
            action_space: vec![], arena_capacity: 2_000_000,
        };

        while !game.is_terminal() && game.history.len() < 200 { 
            let (_, oracle_eval_raw) = oracle.consult(&game.board);
            let side_to_move = game.board.side_to_move();
            
            let oracle_eval = if side_to_move == Color::White {
                (oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0)
            } else {
                (-oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0)
            };

            {
                let mut b = brain_arc.lock().unwrap();
                let base_inputs = extract_topology_features(&game.board);
                let (signature, feelings) = b.senses.get_sensory_signature(&base_inputs, None);
                
                for (i, &feeling) in feelings.iter().enumerate() {
                    if i < feature_histories.len() { feature_histories[i].push(feeling); }
                }

                let current_eval = b.memory.evaluate_context(signature, &feelings);
                total_surprisal += (oracle_eval - current_eval).abs();
                b.memory.learn_context(signature, &feelings, oracle_eval);
            }

            let mut local_config = config.clone();
            local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
            
            let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
            let chosen_move = best_path.and_then(|p| p.first().copied()).expect("No legal moves found");
            
            game.step(chosen_move).unwrap();
        }

        let actual_length = game.history.len() as f32;
        let final_avg_surprise = total_surprisal / actual_length.max(1.0);

        let mut b = brain_arc.lock().unwrap();
        let mut pruned_features = 0;

        // 🌸 HYBRID RESERVOIR: Only check and prune every 10th game!
        if g % 10 == 0 {
            let mut kill_list = Vec::new();
            let mut unique_counts = Vec::new();

            for i in 0..128 {
                if i >= b.senses.features.len() { break; }
                let history = &feature_histories[i];
                if history.is_empty() { continue; }
                let mut unique_vals = history.clone();
                unique_vals.sort(); unique_vals.dedup();
                unique_counts.push((i, unique_vals.len()));
                if unique_vals.len() <= 1 { kill_list.push(i); }
            }

            if kill_list.is_empty() && final_avg_surprise > 0.20 {
                unique_counts.sort_by_key(|&(_, count)| count);
                if let Some(&(idx, _)) = unique_counts.first() { kill_list.push(idx); }
            }

            kill_list.sort(); kill_list.dedup(); 
            kill_list.truncate(1); // 🌸 Max 1 dead pixel replaced per sweep!
            
            for &idx in kill_list.iter().rev() {
                b.senses.features.remove(idx);
                b.senses.add_random_hypothesis();
                pruned_features += 1;
            }
        }

        b.save_and_merge("soul_memory.bin");
        println!("Sandbox Game {} | Len: {} | Avg Surprise: {:.4} | Pruned: {}", 
            g, actual_length, final_avg_surprise, pruned_features);
    }
}
// 🌸 THE AUTOMATED HEADLESS VERSION (No Stdin!)
pub fn run_sg_vs_sg_training_automated(num_games: usize) {
    println!("\n🌸 HEADLESS SANDBOX START: {} Games | 128 Eyes", num_games);
    
    let mut oracle = Oracle::new(); 
    let brain = SoulBrain::load_binary("soul_memory.bin", 128);
    let brain_arc = Arc::new(Mutex::new(brain));
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };

    // We track this outside the loop so we can display it every game
    let mut last_pruned_count = 0;

    for g in 1..=num_games {
        let mut game = ChessWorld { 
            board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc.clone() 
        };
        
        let mut total_surprisal = 0.0;
        let mut feature_histories: Vec<Vec<i64>> = vec![Vec::new(); 128];

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 8000, max_depth: 16, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 1.5, length_penalty: 0.0, loop_penalty: 1.0,
            action_space: vec![], arena_capacity: 2_000_000,
        };

        while !game.is_terminal() && game.history.len() < 200 { 
            let (_, oracle_eval_raw) = oracle.consult(&game.board);
            let side_to_move = game.board.side_to_move();
            let oracle_eval = if side_to_move == Color::White { (oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0) } 
                               else { (-oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0) };

            {
                let mut b = brain_arc.lock().unwrap();
                let base_inputs = extract_topology_features(&game.board);
                let (signature, feelings) = b.senses.get_sensory_signature(&base_inputs, None);
                for (i, &feeling) in feelings.iter().enumerate() { feature_histories[i].push(feeling); }
                let current_eval = b.memory.evaluate_context(signature, &feelings);
                total_surprisal += (oracle_eval - current_eval).abs();
                b.memory.learn_context(signature, &feelings, oracle_eval);
            }

            let mut local_config = config.clone();
            local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
            let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
            let chosen_move = best_path.and_then(|p| p.first().copied()).expect("No moves");
            game.step(chosen_move).unwrap();
        }

        // 🌸 Patient Hybrid Pruning every 20 games
        if g % 20 == 0 {
            let mut b = brain_arc.lock().unwrap();
            let mut kill_list = Vec::new();
            let mut unique_counts = Vec::new();
            for i in 0..128 {
                let history = &feature_histories[i];
                let mut unique_vals = history.clone();
                unique_vals.sort(); unique_vals.dedup();
                unique_counts.push((i, unique_vals.len()));
                if unique_vals.len() <= 1 { kill_list.push(i); }
            }
            kill_list.sort(); kill_list.dedup(); 
            
            // We only take 1 to keep it stable
            kill_list.truncate(1);
            last_pruned_count = kill_list.len(); 

            for &idx in kill_list.iter().rev() {
                b.senses.features.remove(idx);
                b.senses.add_random_hypothesis();
            }
            b.save_and_merge("soul_memory.bin");
        } else {
            // Reset the count for games where we didn't prune
            last_pruned_count = 0;
        }

        let actual_length = game.history.len() as f32;
        let avg_surprise = total_surprisal / actual_length.max(1.0);
        
        // 🌸 NOW it shows the pruned count in the console!
        println!("🤖 Game {}/{} | Len: {} | Surprise: {:.4} | Pruned: {}", 
            g, num_games, actual_length, avg_surprise, last_pruned_count);
    }

    let b = brain_arc.lock().unwrap();
    b.save_and_merge("soul_memory.bin");
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
        board: Board::default(), history: vec![Board::default().get_hash()], brain: brain_arc 
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
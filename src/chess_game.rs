use std::io::{self, Write, BufRead, BufReader};
use std::fs::File;
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use serde::{Deserialize, Serialize};
use rand::Rng; // 🌸 True randomness for evolution!

use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats
};
use soulgain::vm::{CoreMind, Op, StepStatus};
use soulgain::types::UVal;

// ─────────────────────────────────────────────────────────────────────────────
// THE ORACLE WRAPPER (Depth 10 Stress Test 🌸)
// ─────────────────────────────────────────────────────────────────────────────

struct Oracle {
    process: Child,
    cache: HashMap<u64, (String, f64)>, 
}

impl Oracle {
    fn new() -> Self {
        let child = Command::new("./stockfish_oracle") // Ensure your stockfish executable is here
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
        
        // 🌸 The Oracle is fully awake!
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
// CHESS WORLD & TOPOLOGY FEATURES
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
// TRUE EPISODIC MEMORY GRAPH WITH GENERALIZATION (u128) 🌸
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub experiences: HashMap<u128, f32>,
    pub visit_counts: HashMap<u128, u32>,
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

    pub fn evaluate_context(&self, signature: u128, feelings: &[i64]) -> f32 {
        if let Some(&exact_val) = self.experiences.get(&signature) {
            return exact_val;
        }

        let mut total_familiarity = 0.0;
        let mut known_feelings = 0;

        for (i, &feeling) in feelings.iter().enumerate() {
            let bucket_key = (i as u64) << 32 | (feeling as u64 & 0xFFFFFFFF);
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

    pub fn learn_context(&mut self, signature: u128, feelings: &[i64], target_eval: f32) {
        let count = self.visit_counts.entry(signature).or_insert(0);
        *count += 1;
        let current_val = self.experiences.entry(signature).or_insert(0.0);
        let learning_rate = 1.0 / (*count as f32).max(1.0);
        *current_val += learning_rate * (target_eval - *current_val);

        for (i, &feeling) in feelings.iter().enumerate() {
            let bucket_key = (i as u64) << 32 | (feeling as u64 & 0xFFFFFFFF);
            let b_count = self.bucket_counts.entry(bucket_key).or_insert(0);
            *b_count += 1;
            
            let b_val = self.bucket_values.entry(bucket_key).or_insert(0.0);
            let b_lr = 1.0 / (*b_count as f32).max(1.0);
            *b_val += b_lr * (target_eval - *b_val);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SENSORY ORGANS (The "Eyes" - 32 Slots) 🌸
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveredFeature {
    pub snippet: Vec<Op>,
    pub reliability: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeaturePool {
    pub features: Vec<DiscoveredFeature>,
    pub max_features: usize,
}

impl FeaturePool {
    pub fn new(max_features: usize) -> Self {
        let mut pool = Self {
            features: Vec::new(),
            max_features,
        };
        for _ in 0..max_features {
            pool.add_random_hypothesis();
        }
        pool
    }

    pub fn add_random_hypothesis(&mut self) {
        let mut rng = rand::thread_rng();
        let ops = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Dup, Op::Swap, Op::Over, Op::Gt];
        let mut snippet = Vec::new();
        
        let len = rng.gen_range(3..=7); 
        for _ in 0..len {
            snippet.push(ops[rng.gen_range(0..ops.len())]);
        }
        snippet.push(Op::Halt);

        if self.features.len() < self.max_features {
            self.features.push(DiscoveredFeature { snippet, reliability: 1.0 });
        }
    }

    pub fn get_sensory_signature(&self, base_inputs: &[UVal], ignore_index: Option<usize>) -> (u128, Vec<i64>) {
        let mut signature: u128 = 0; 
        let mut feelings = Vec::new();
        
        for (i, feature) in self.features.iter().take(32).enumerate() {
            if Some(i) == ignore_index { continue; }

            let mut mind = CoreMind::new();
            mind.reset(base_inputs);
            
            let mut step_count = 0;
            while mind.ip() < feature.snippet.len() && step_count < 32 {
                if mind.step(feature.snippet[mind.ip()]) == StepStatus::Halt { break; }
                step_count += 1;
            }

            let output_val = mind.extract_output().last()
                .and_then(|v| if let UVal::Number(n) = v { Some(*n as f32) } else { None })
                .unwrap_or(0.0);
            
            let discrete_feeling = (output_val * 2.0).round() as i64;
            feelings.push(discrete_feeling); 
            
            // Pack 4 bits into the u128 register (32 * 4 = 128 perfect bits)
            let packed_feeling = (discrete_feeling.wrapping_add(8) as u128) & 0x0F;
            signature |= packed_feeling << (i * 4);
        }

        (signature, feelings)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE UNIFIED BRAIN (With Safe Merge Memory fix 🌸)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct SoulBrain {
    pub senses: FeaturePool,
    pub memory: EpisodicMemory,
}

impl SoulBrain {
    pub fn save_and_merge(&self, path: &str) {
        // 1. Try to load existing brain to merge into
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

        // 2. Merge all memory pools safely
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

        // 3. Keep the latest, pruned "Eyes"
        disk_brain.senses = self.senses.clone();

        // 4. Save to disk securely!
        if let Ok(file) = File::create(path) {
            let _ = serde_json::to_writer(file, &disk_brain);
        }
    }

    pub fn load_binary(path: &str, max_features: usize) -> Self {
        if let Ok(file) = File::open(path) {
            match serde_json::from_reader(file) {
                Ok(brain) => return brain,
                Err(e) => println!("⚠️ Failed to parse soul_memory.bin. Starting fresh! Error: {}", e),
            }
        } else {
            println!("🌱 No previous memory found. Birthing new Soul...");
        }
        Self {
            senses: FeaturePool::new(max_features),
            memory: EpisodicMemory::new(),
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
// PURE TEMPORAL EPISODIC LEARNING LOOP
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_autoplay() {
    println!("\n🌸 EVOLUTIONARY SURVIVAL LOOP START (32 Eyes | 8k Sims)");
    
    let mut oracle = Oracle::new(); 
    let brain = SoulBrain::load_binary("soul_memory.bin", 32);
    let brain_arc = Arc::new(Mutex::new(brain));
    
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };
    
    // Hardwired Evolution Constants
    const TARGET_SURVIVAL: f32 = 80.0; // 40 moves
    const TRAUMA_WEIGHT: f32 = 0.3;    // Penalty for dying early
    const STAGNATION_PENALTY: f32 = 0.15; // Penalty for not achieving an advantage

    print!("Games to play: "); 
    io::stdout().flush().unwrap();
    let mut input_text = String::new(); 
    io::stdin().read_line(&mut input_text).unwrap();
    let num_games: usize = input_text.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut game = ChessWorld { 
            board: Board::default(), 
            history: vec![Board::default().get_hash()],
            brain: brain_arc.clone() 
        };
        let soul_color = if g % 2 == 1 { Color::White } else { Color::Black };
        
        let mut total_surprisal = 0.0;
        let mut peak_eval: f32 = -1.0; // Track the best 'vibe' SoulGain ever reached
        let mut feature_histories: Vec<Vec<i64>> = vec![Vec::new(); 32];

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 8000, 
            max_depth: 16, 
            max_program_len: 8, 
            max_ops_per_candidate: 8,
            exploration_constant: 1.5, 
            length_penalty: 0.0, 
            loop_penalty: 1.0,
            action_space: vec![], 
            arena_capacity: 2_000_000,
        };

        while !game.is_terminal() && game.history.len() < 200 { 
            if game.board.side_to_move() == soul_color {
                // SOULGAIN PLAYS
                let mut local_config = config.clone();
                local_config.action_space = MoveGen::new_legal(&game.board).map(WrappedMove).collect();
                
                let (best_path, _) = solve_universal_with_stats(&game, &local_config, &policy);
                let chosen_move = best_path.and_then(|p| p.first().copied()).expect("No legal moves found");
                
                game.step(chosen_move).unwrap();
            } else {
                // STOCKFISH PLAYS & SOULGAIN LEARNS
                let (sf_move_str, oracle_eval_raw) = oracle.consult(&game.board);
                
                let mut b = brain_arc.lock().unwrap();
                let base_inputs = extract_topology_features(&game.board);
                let (signature, feelings) = b.senses.get_sensory_signature(&base_inputs, None);
                
                // Recording feelings to check for "Dead Pixels"
                for (i, &feeling) in feelings.iter().enumerate() {
                    if i < feature_histories.len() { feature_histories[i].push(feeling); }
                }

                let current_eval = b.memory.evaluate_context(signature, &feelings);
                
                // Flip eval if Soul is Black, and clamp it
                let oracle_eval = if soul_color == Color::White {
                    (oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0)
                } else {
                    (-oracle_eval_raw as f32 / 100.0).clamp(-1.0, 1.0)
                };

                // Track Peak Vibe for Winning Bias
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

        // ─────────────────────────────────────────────────────────────────────
        // POST-MORTEM BIASING 🌸
        // ─────────────────────────────────────────────────────────────────────
        let actual_length = game.history.len() as f32;
        let prediction_surprise = total_surprisal / actual_length.max(1.0);
        
        // 1. Survival Penalty (Dying early is traumatic)
        let survival_penalty = if actual_length < TARGET_SURVIVAL {
            ((TARGET_SURVIVAL - actual_length) / TARGET_SURVIVAL) * TRAUMA_WEIGHT
        } else {
            0.0
        };

        // 2. Stagnation Penalty (Living long without attacking is also failure)
        // If peak_eval < 0.1, it means the bot never actually threatened Stockfish.
        let win_bias_penalty = if peak_eval < 0.1 { STAGNATION_PENALTY } else { 0.0 };

        // The final surprise determines if the bot prunes its eyes
        let final_avg_surprise = prediction_surprise + survival_penalty + win_bias_penalty;

        let mut b = brain_arc.lock().unwrap();
        let mut kill_list = Vec::new();
        let mut unique_counts = Vec::new();
        let mut pruned_features = 0;

        for i in 0..32 {
            if i >= b.senses.features.len() { break; }
            let history = &feature_histories[i];
            if history.is_empty() { continue; }

            let mut unique_vals = history.clone();
            unique_vals.sort();
            unique_vals.dedup();
            unique_counts.push((i, unique_vals.len()));

            // PRUNE DEAD PIXELS (Eyes that don't change)
            if unique_vals.len() <= 1 { kill_list.push(i); }
        }

        // SURVIVAL CHURN: If the game was short OR passive, mutate the weakest eye
        if kill_list.is_empty() && final_avg_surprise > 0.15 {
            unique_counts.sort_by_key(|&(_, count)| count);
            if let Some(&(idx, _)) = unique_counts.first() {
                kill_list.push(idx);
            }
        }

        kill_list.sort();
        kill_list.dedup();
        for &idx in kill_list.iter().rev() {
            b.senses.features.remove(idx);
            b.senses.add_random_hypothesis();
            pruned_features += 1;
        }

        // 🌸 Using the new safe merge to protect our precious memories!
        b.save_and_merge("soul_memory.bin");

        println!("Game {} | Len: {} | Peak: {:.2} | Surprise: {:.4} | Pruned: {}", 
            g, actual_length, peak_eval, final_avg_surprise, pruned_features);
    }
}

pub fn run_self_match() {
    println!("\n⚔️ SOULGAIN vs STOCKFISH\n");
    let brain = SoulBrain::load_binary("soul_memory.bin", 32); 
    let brain_arc = Arc::new(Mutex::new(brain));
    let policy = ProgrammaticChessPolicy { brain: brain_arc.clone() };
    let mut oracle = Oracle::new();
    
    let mut game = ChessWorld { 
        board: Board::default(), 
        history: vec![Board::default().get_hash()],
        brain: brain_arc 
    };
    let soul_color = Color::White;

    while !game.is_terminal() {
        if game.board.side_to_move() == soul_color {
            let config = ReasoningConfig::<WrappedMove> {
                simulations: 80000, max_depth: 18, max_program_len: 8, max_ops_per_candidate: 8,
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
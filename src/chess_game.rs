use std::io::{self, Write, BufRead, BufReader};
use std::fs::File; 
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};

use rustc_hash::FxHashMap; 
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use serde::{Deserialize, Serialize};
use rand::Rng; 

use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats
};

// ─────────────────────────────────────────────────────────────────────────────
// STOCKFISH PROCESS WRAPPER (Fixed Persistent Reader)
// ─────────────────────────────────────────────────────────────────────────────

struct StockfishProcess {
    process: Child,
    stdout_reader: BufReader<std::process::ChildStdout>,
    cache: FxHashMap<(u64, u8), (String, f32)>, 
}

impl StockfishProcess {
    fn new(skill_level: i32) -> Self {
        let mut child = Command::new("./stockfish_oracle") 
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to start stockfish");
        
        // 🌸 Taking ownership of stdout ONCE to prevent buffer dropping
        let stdout = child.stdout.take().expect("Failed to capture stockfish stdout");
        let reader = BufReader::new(stdout);
        
        let mut sf = Self { 
            process: child,
            stdout_reader: reader,
            cache: FxHashMap::with_capacity_and_hasher(100_000, Default::default()) 
        };
        
        sf.send("uci");
        sf.send("isready");
        sf.send(&format!("setoption name Skill Level value {}", skill_level)); 
        sf
    }

    fn send(&mut self, cmd: &str) {
        if let Some(stdin) = self.process.stdin.as_mut() {
            let _ = writeln!(stdin, "{}", cmd);
        }
    }

    fn evaluate(&mut self, board: &Board, depth: u8) -> (String, f32) {
        let hash = board.get_hash();
        let cache_key = (hash, depth);
        
        if let Some(res) = self.cache.get(&cache_key) {
            return res.clone();
        }

        let fen = format!("{}", board);
        self.send(&format!("position fen {}", fen));
        self.send(&format!("go depth {}", depth)); 

        let mut final_score = 0.0;
        let mut best_move = String::new();
        let mut line = String::new();

        // 🌸 Persistent read loop that won't choke the OS pipe
        loop {
            line.clear();
            if self.stdout_reader.read_line(&mut line).unwrap_or(0) == 0 {
                break; // EOF
            }
            
            if line.contains("score cp") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(pos) = parts.iter().position(|&r| r == "cp") {
                    if let Some(val) = parts.get(pos + 1).and_then(|v| v.parse::<f32>().ok()) {
                        final_score = val; 
                    }
                }
            } else if line.contains("score mate") {
                final_score = if line.contains("score mate -") { -5000.0 } else { 5000.0 };
            }
            
            if line.starts_with("bestmove") {
                best_move = line.split_whitespace().nth(1).unwrap_or("").to_string();
                break;
            }
        }
        
        let res = (best_move, final_score);
        self.cache.insert(cache_key, res.clone());
        res
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CHESS ENVIRONMENT
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct WrappedMove(pub ChessMove);

#[derive(Clone)]
struct ChessWorld { 
    board: Board,
    history: Vec<u64>, 
    model: Arc<Mutex<ChessModel>>, 
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
        
        let model = self.model.lock().unwrap();
        let base_inputs = extract_board_features(&sim_board);
        
        let (score, _) = model.ensemble.evaluate_and_get_outputs(&base_inputs);
        (score, path.len() as u64)
    }
}

fn extract_board_features(board: &Board) -> [f32; 65] {
    let mut features = [0.0; 65];
    for i in 0..64 {
        let sq = unsafe { Square::new(i as u8) }; 
        if let Some(p) = board.piece_on(sq) {
            let type_val = match p {
                Piece::Pawn => 1.0, Piece::Knight => 3.0, Piece::Bishop => 3.5,
                Piece::Rook => 5.0, Piece::Queen => 9.0, Piece::King => 20.0,
            };
            features[i] = if board.color_on(sq) == Some(Color::White) { type_val } else { -type_val };
        }
    }
    features[64] = if board.side_to_move() == Color::White { 1.0 } else { -1.0 };
    features
}

// ─────────────────────────────────────────────────────────────────────────────
// MATRIX ENSEMBLE MODEL
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatrixFilter {
    pub weights: [f32; 9],
    pub x: usize, 
    pub y: usize, 
}

impl MatrixFilter {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = [0.0; 9];
        for w in &mut weights {
            *w = rng.gen_range(-5.0..=5.0);
        }
        Self {
            weights,
            x: rng.gen_range(0..=5),
            y: rng.gen_range(0..=5),
        }
    }

    pub fn dot_product(&self, board_box: &[f32; 65]) -> f32 {
        let mut output = 0.0;
        for dy in 0..3 {
            for dx in 0..3 {
                let board_idx = (self.y + dy) * 8 + (self.x + dx);
                output += self.weights[dy * 3 + dx] * board_box[board_idx];
            }
        }
        output
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ensemble {
    pub filters: Vec<MatrixFilter>,
}

impl Ensemble {
    pub fn new(size: usize) -> Self {
        let mut filters = Vec::with_capacity(size);
        for _ in 0..size { filters.push(MatrixFilter::new_random()); }
        Self { filters }
    }

    pub fn evaluate_and_get_outputs(&self, board_box: &[f32; 65]) -> (f32, Vec<f32>) {
        let mut outputs = Vec::with_capacity(self.filters.len());
        let mut sum = 0.0;
        for filter in &self.filters {
            let out = filter.dot_product(board_box);
            outputs.push(out);
            sum += out;
        }
        let avg = if outputs.is_empty() { 0.0 } else { sum / outputs.len() as f32 };
        (avg, outputs)
    }

    pub fn update_highest_error_filter(&mut self, board_box: &[f32; 65], outputs: &[f32], target_eval: f32) {
        let mut worst_idx = 0;
        let mut max_err = -1.0;
        let mut rng = rand::thread_rng();

        for (i, &output) in outputs.iter().enumerate() {
            let err = (target_eval - output).abs();
            
            if (err - max_err).abs() < 0.1 {
                if rng.gen_bool(0.5) { worst_idx = i; }
            } else if err > max_err {
                max_err = err;
                worst_idx = i;
            }
        }

        let lr = 0.5; 
        let worst_filter = &mut self.filters[worst_idx];
        let worst_output = outputs[worst_idx];
        let delta = target_eval - worst_output;

        for dy in 0..3 {
            for dx in 0..3 {
                let board_idx = (worst_filter.y + dy) * 8 + (worst_filter.x + dx);
                let input_val = board_box[board_idx];
                
                let mut new_w = worst_filter.weights[dy * 3 + dx] + (delta * input_val * lr);

                if new_w > 100.0 { new_w = 100.0; }
                else if new_w < -100.0 { new_w = -100.0; }

                if new_w.abs() < 1.0 { new_w = 0.0; } 
                else { new_w = new_w.round(); }

                worst_filter.weights[dy * 3 + dx] = new_w;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA MANAGEMENT (Restored to beautifully readable JSON)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ChessModel {
    pub ensemble: Ensemble,
}

impl ChessModel {
    pub fn load_or_create(path: &str, num_matrices: usize) -> Self {
        if let Ok(file) = File::open(path) {
            serde_json::from_reader(file).unwrap_or_else(|_| ChessModel { ensemble: Ensemble::new(num_matrices) })
        } else {
            ChessModel { ensemble: Ensemble::new(num_matrices) }
        }
    }

    pub fn save(&self, path: &str) {
        let temp_path = format!("{}.tmp", path);
        if let Ok(file) = File::create(&temp_path) {
            if serde_json::to_writer_pretty(file, &self).is_ok() {
                let _ = std::fs::rename(&temp_path, path);
            }
        }
    }
}

struct ProgrammaticChessPolicy {
    model: Arc<Mutex<ChessModel>>,
}

impl CognitivePolicy<Board, WrappedMove> for ProgrammaticChessPolicy {
    fn evaluate(&self, state: &Board) -> f32 {
        let m = self.model.lock().unwrap(); 
        let base_inputs = extract_board_features(state);
        let (score, _) = m.ensemble.evaluate_and_get_outputs(&base_inputs);
        score
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
// THE MODES
// ─────────────────────────────────────────────────────────────────────────────

pub fn train_from_stockfish() {
    println!("\nSTARTING THE FORGE: STOCKFISH VS STOCKFISH");
    
    let mut sf = StockfishProcess::new(20); 
    let model = ChessModel::load_or_create("soul_matrices.json", 2048); 
    let model_arc = Arc::new(Mutex::new(model));

    print!("Games to process: "); 
    io::stdout().flush().unwrap();
    let mut input_text = String::new(); 
    io::stdin().read_line(&mut input_text).unwrap();
    let num_games: usize = input_text.trim().parse().unwrap_or(10);

    for g in 1..=num_games {
        let mut board = Board::default();
        let mut move_count = 0;
        let mut total_error = 0.0;
        let mut updates = 0;

        let mut rng = rand::thread_rng();
        let depth_white = rng.gen_range(1..=10);
        let depth_black = rng.gen_range(1..=10);

        while board.status() == BoardStatus::Ongoing && move_count < 150 {
            let side_to_move = board.side_to_move();
            let current_depth = if side_to_move == Color::White { depth_white } else { depth_black };

            let (sf_move_str, sf_eval) = sf.evaluate(&board, current_depth);
            let absolute_eval = if side_to_move == Color::White { sf_eval } else { -sf_eval };
            
            {
                let mut m = model_arc.lock().unwrap();
                let base_inputs = extract_board_features(&board);
                
                let (current_eval, outputs) = m.ensemble.evaluate_and_get_outputs(&base_inputs);
                let error = (absolute_eval - current_eval).abs();
                total_error += error;

                if error > 10.0 {
                    m.ensemble.update_highest_error_filter(&base_inputs, &outputs, absolute_eval);
                    updates += 1;
                }
            }

            if let Some(m) = MoveGen::new_legal(&board).find(|m| m.to_string() == sf_move_str) {
                board = board.make_move_new(m);
                move_count += 1;
            } else {
                break;
            }
        }

        let avg_error = if move_count > 0 { total_error / move_count as f32 } else { 0.0 };
        println!("Game {} (W: D{}, B: D{}) | Moves: {} | Hammer Strikes: {} | Avg Error: {:.2} cp", 
                 g, depth_white, depth_black, move_count, updates, avg_error);

        model_arc.lock().unwrap().save("soul_matrices.json");
    }
    
    println!("Training complete.");
}

pub fn play_vs_stockfish() {
    println!("\nARENA: SOULGAIN vs STOCKFISH\n");
    let model = ChessModel::load_or_create("soul_matrices.json", 2048); 
    let model_arc = Arc::new(Mutex::new(model));
    let policy = ProgrammaticChessPolicy { model: model_arc.clone() };
    let mut sf = StockfishProcess::new(10); 
    
    let mut game = ChessWorld { 
        board: Board::default(), history: vec![Board::default().get_hash()], model: model_arc.clone() 
    };
    let soul_color = Color::White;

    while !game.is_terminal() {
        if game.board.side_to_move() == soul_color {
            let config = ReasoningConfig::<WrappedMove> {
                simulations: 8000, max_depth: 20, max_program_len: 8, max_ops_per_candidate: 8,
                exploration_constant: 0.5, length_penalty: 0.1, loop_penalty: 2.0,
                action_space: MoveGen::new_legal(&game.board).map(WrappedMove).collect(), arena_capacity: 1_000_000,
            };
            let (best_path, _) = solve_universal_with_stats(&game, &config, &policy);
            if let Some(m) = best_path.and_then(|p| p.first().copied()) {
                println!("SoulGain: {}", m.0.to_string());
                game.step(m).unwrap(); 
            } else { break; }
        } else {
            let (sf_move_str, _) = sf.evaluate(&game.board, 10);
            if let Some(m) = MoveGen::new_legal(&game.board).find(|m| m.to_string() == sf_move_str) {
                println!("Stockfish: {}", sf_move_str);
                game.step(WrappedMove(m)).unwrap();
            } else { break; }
        }
        println!("{}\n", game.board);
    }
    println!("Match finished! Status: {:?}", game.board.status());
}

pub fn play_self() {
    println!("\nARENA: SOULGAIN vs SOULGAIN\n");
    let model = ChessModel::load_or_create("soul_matrices.json", 2048); 
    let model_arc = Arc::new(Mutex::new(model));
    let policy = ProgrammaticChessPolicy { model: model_arc.clone() };
    
    let mut game = ChessWorld { 
        board: Board::default(), history: vec![Board::default().get_hash()], model: model_arc.clone() 
    };

    while !game.is_terminal() {
        let color_name = if game.board.side_to_move() == Color::White { "White" } else { "Black" };

        let config = ReasoningConfig::<WrappedMove> {
            simulations: 8000, max_depth: 20, max_program_len: 8, max_ops_per_candidate: 8,
            exploration_constant: 0.5, length_penalty: 0.1, loop_penalty: 2.0,
            action_space: MoveGen::new_legal(&game.board).map(WrappedMove).collect(), arena_capacity: 1_000_000,
        };
        
        let (best_path, _) = solve_universal_with_stats(&game, &config, &policy);
        
        if let Some(m) = best_path.and_then(|p| p.first().copied()) {
            println!("SoulGain ({}): {}", color_name, m.0.to_string());
            game.step(m).unwrap(); 
        } else { 
            break; 
        }
        println!("{}\n", game.board);
    }
    println!("Match finished! Status: {:?}", game.board.status());
}
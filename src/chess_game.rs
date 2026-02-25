use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::str::FromStr;

use rustc_hash::FxHashMap; 
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece, Color, Square};
use serde::{Deserialize, Serialize};
use rand::Rng; 

use soulgain::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats
};

// ─────────────────────────────────────────────────────────────────────────────
// DATASET RECORD STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug)]
pub struct PositionRecord {
    pub fen: String,
    pub target_eval: f32,
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
    
    fn current_state(&self) -> Self::State { 
        self.board.clone() 
    }
    
    fn current_player(&self) -> i32 { 
        if self.board.side_to_move() == Color::White { 1 } else { -1 } 
    }
    
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
        
        let score = model.ensemble.evaluate(&base_inputs);
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
// MATRIX ENSEMBLE MODEL (Blazing Fast Unrolled Operations)
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

    #[inline(always)]
    pub fn dot_product(&self, board_box: &[f32; 65]) -> f32 {
        let base = self.y * 8 + self.x;
        self.weights[0] * board_box[base]
            + self.weights[1] * board_box[base + 1]
            + self.weights[2] * board_box[base + 2]
            + self.weights[3] * board_box[base + 8]
            + self.weights[4] * board_box[base + 9]
            + self.weights[5] * board_box[base + 10]
            + self.weights[6] * board_box[base + 16]
            + self.weights[7] * board_box[base + 17]
            + self.weights[8] * board_box[base + 18]
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

    pub fn evaluate(&self, board_box: &[f32; 65]) -> f32 {
        let sum: f32 = self.filters.iter().map(|f| f.dot_product(board_box)).sum();
        if self.filters.is_empty() { 0.0 } else { sum / self.filters.len() as f32 }
    }

    pub fn evaluate_and_get_outputs(&self, board_box: &[f32; 65], outputs: &mut Vec<f32>) -> f32 {
        outputs.clear();
        let mut sum = 0.0;
        for filter in &self.filters {
            let out = filter.dot_product(board_box);
            outputs.push(out);
            sum += out;
        }
        if outputs.is_empty() { 0.0 } else { sum / outputs.len() as f32 }
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

        let base = worst_filter.y * 8 + worst_filter.x;
        let offsets = [0, 1, 2, 8, 9, 10, 16, 17, 18];
        
        for (i, &offset) in offsets.iter().enumerate() {
            let board_idx = base + offset;
            let input_val = board_box[board_idx];
            
            let mut new_w = worst_filter.weights[i] + (delta * input_val * lr);

            if new_w > 100.0 { new_w = 100.0; }
            else if new_w < -100.0 { new_w = -100.0; }

            if new_w.abs() < 1.0 { new_w = 0.0; } 
            else { new_w = new_w.round(); }

            worst_filter.weights[i] = new_w;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA MANAGEMENT (Pure, Blazing-Fast Bincode)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ChessModel {
    pub ensemble: Ensemble,
}

impl ChessModel {
    pub fn load_or_create(path: &str, num_matrices: usize) -> Self {
        if let Ok(file) = File::open(path) {
            bincode::deserialize_from(file).unwrap_or_else(|_| ChessModel { 
                ensemble: Ensemble::new(num_matrices) 
            })
        } else {
            ChessModel { ensemble: Ensemble::new(num_matrices) }
        }
    }

    pub fn save(&self, path: &str) {
        let temp_path = format!("{}.tmp", path);
        if let Ok(file) = File::create(&temp_path) {
            if bincode::serialize_into(file, &self).is_ok() {
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
        m.ensemble.evaluate(&base_inputs)
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
// DATASET UTILITIES (Kaggle CSV -> Bincode)
// ─────────────────────────────────────────────────────────────────────────────

pub fn convert_kaggle_to_bincode(csv_path: &str, bin_path: &str) {
    println!("\nCreating beautifully packed bincode file from Kaggle CSV...");
    
    let csv_file = File::open(csv_path).expect("Could not open Kaggle CSV!");
    let reader = BufReader::new(csv_file);
    
    let bin_file = File::create(bin_path).expect("Could not create bincode file!");
    let mut writer = BufWriter::new(bin_file);
    
    let mut count = 0;
    
    for line in reader.lines().skip(1) { // Skip header row
        if let Ok(l) = line {
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() >= 2 {
                let fen = parts[0].to_string();
                let eval_str = parts[1];
                
                let target_eval = if eval_str.starts_with('#') {
                    // Checkmate format: #3 or #-4
                    if eval_str.contains('-') { -5000.0 } else { 5000.0 }
                } else {
                    // Standard centipawn or numeric evaluation
                    eval_str.parse::<f32>().unwrap_or(0.0) 
                };

                let record = PositionRecord { fen, target_eval };
                bincode::serialize_into(&mut writer, &record).expect("Failed to write to bincode");
                
                count += 1;
                if count % 100_000 == 0 {
                    println!("Packed {} positions into bincode...", count);
                }
            }
        }
    }
    writer.flush().unwrap();
    println!("Successfully packed {} positions! Ready for blazing fast training.", count);
}

// ─────────────────────────────────────────────────────────────────────────────
// THE TRAINING FORGE (Streaming from Bincode)
// ─────────────────────────────────────────────────────────────────────────────

pub fn train_from_bincode(bin_path: &str) {
    println!("\nSTARTING THE FORGE: ABSORBING BINCODE DATASET");
    
    // Using your beautiful new binary save file!
    let mut model = ChessModel::load_or_create("soul_matrices.bin", 2048); 
    let mut outputs_buffer = Vec::with_capacity(2048);

    let bin_file = File::open(bin_path).expect("Could not open the bincode file! Did you convert it first?");
    let mut reader = BufReader::new(bin_file);

    let mut start_time = Instant::now();
    let mut positions_processed = 0;
    let mut total_error = 0.0;
    let mut updates = 0;

    println!("Streaming positions...");

    while let Ok(record) = bincode::deserialize_from::<_, PositionRecord>(&mut reader) {
        if let Ok(board) = Board::from_str(&record.fen) {
            let absolute_eval = if board.side_to_move() == Color::White { 
                record.target_eval 
            } else { 
                -record.target_eval 
            };
            
            let base_inputs = extract_board_features(&board);
            
            let current_eval = model.ensemble.evaluate_and_get_outputs(&base_inputs, &mut outputs_buffer);
            let error = (absolute_eval - current_eval).abs();
            total_error += error;

            if error > 20.0 { 
                model.ensemble.update_highest_error_filter(&base_inputs, &outputs_buffer, absolute_eval);
                updates += 1;
            }

            positions_processed += 1;

            if positions_processed % 50_000 == 0 {
                let duration = start_time.elapsed();
                let avg_error = total_error / 50_000.0;
                println!(
                    "Absorbed {} Positions | Hammer Strikes: {} | Avg Error: {:.2} cp | Time for 50k: {:?}", 
                    positions_processed, updates, avg_error, duration
                );
                
                total_error = 0.0;
                updates = 0;
                start_time = Instant::now();
                
                model.save("soul_matrices.bin");
            }
        }
    }
    
    model.save("soul_matrices.bin");
    println!("Dataset absorption complete! Your matrices are beautifully forged.");
}

// ─────────────────────────────────────────────────────────────────────────────
// ARENA MODE
// ─────────────────────────────────────────────────────────────────────────────

pub fn play_self() {
    println!("\nARENA: SOULGAIN vs SOULGAIN\n");
    let model = ChessModel::load_or_create("soul_matrices.bin", 2048); 
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
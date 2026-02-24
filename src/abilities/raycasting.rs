use std::fmt;
use std::io::{self, Write};
use std::fs::File;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::alphazero::{
    CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. PHYSICS
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pixel { Empty = 0, Wall = 1, UpGravity = 2, DownGravity = 3 }

const W: usize = 10;
const H: usize = 10;
const GRID_SIZE: usize = W * H;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    pub data: [Pixel; GRID_SIZE],
}

impl Grid {
    pub fn new() -> Self { Self { data: [Pixel::Empty; GRID_SIZE] } }

    #[inline] pub fn get(&self, x: usize, y: usize) -> Pixel { self.data[y * W + x] }
    #[inline] pub fn set(&mut self, x: usize, y: usize, p: Pixel) { self.data[y * W + x] = p; }

    pub fn difference(&self, other: &Grid) -> usize {
        self.data.iter().zip(other.data.iter()).filter(|(a, b)| a != b).count()
    }
}

impl fmt::Display for Grid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..H {
            for x in 0..W {
                write!(f, "{} ", match self.get(x, y) {
                    Pixel::Empty => '.', Pixel::Wall => '#',
                    Pixel::UpGravity => '^', Pixel::DownGravity => 'v',
                })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn calculate_perfect_grid(initial: &Grid) -> Grid {
    let mut g = initial.clone();
    for y in 0..H {
        for x in 0..W {
            match initial.get(x, y) {
                Pixel::UpGravity => {
                    let mut cy = y;
                    while cy > 0 {
                        cy -= 1;
                        if g.get(x, cy) == Pixel::Wall { break; }
                        g.set(x, cy, Pixel::UpGravity);
                    }
                }
                Pixel::DownGravity => {
                    let mut cy = y + 1;
                    while cy < H {
                        if g.get(x, cy) == Pixel::Wall { break; }
                        g.set(x, cy, Pixel::DownGravity);
                        cy += 1;
                    }
                }
                _ => {}
            }
        }
    }
    g
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. ORACLE
// ─────────────────────────────────────────────────────────────────────────────

pub struct RaycastOracle { pub target: Grid }

impl RaycastOracle {
    pub fn new(initial: &Grid) -> Self { Self { target: calculate_perfect_grid(initial) } }

    #[inline]
    pub fn consult(&self, grid: &Grid) -> f32 {
        let d = grid.difference(&self.target);
        if d == 0 { 1.0 } else { 1.0 / (1.0 + d as f32) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. CORRECTNESS-BASED PERCEPTION
//
// No random EyeMinds. Instead: features that actually mean something.
//
// Signature: exact hash of the per-cell correctness pattern (which cells match
// the target). Two grids that are wrong in exactly the same way share a signature.
//
// Buckets: per-column and per-row correctness counts. Two grids that have the
// same column/row correctness profile share bucket values, even if the exact
// wrong cells differ. This is meaningful generalization — "column 3 is fully
// correct" transfers across puzzles where column 3 behaves similarly.
// ─────────────────────────────────────────────────────────────────────────────

pub struct CorrectnessFeatures {
    // Which cells are correct — 100 bits, packed as u64 pair + u32
    pub cell_mask: [u64; 2],        // bits 0-127 (we use 0-99)
    // Per-column correct count (0-10 each)
    pub col_counts: [u8; W],
    // Per-row correct count (0-10 each)
    pub row_counts: [u8; H],
    // Total correct
    pub total_correct: usize,
}

impl CorrectnessFeatures {
    pub fn compute(grid: &Grid, target: &Grid) -> Self {
        let mut cell_mask = [0u64; 2];
        let mut col_counts = [0u8; W];
        let mut row_counts = [0u8; H];
        let mut total_correct = 0;

        for y in 0..H {
            for x in 0..W {
                let idx = y * W + x;
                if grid.get(x, y) == target.get(x, y) {
                    // Set bit idx in the mask
                    cell_mask[idx / 64] |= 1u64 << (idx % 64);
                    col_counts[x] += 1;
                    row_counts[y] += 1;
                    total_correct += 1;
                }
            }
        }

        Self { cell_mask, col_counts, row_counts, total_correct }
    }

    // Exact signature — two grids wrong in exactly the same cells share this
    pub fn exact_signature(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.cell_mask.hash(&mut hasher);
        hasher.finish()
    }

    // Structural buckets — column and row correctness patterns
    // "Column 3 has 8/10 correct" transfers across puzzles
    pub fn bucket_keys(&self) -> Vec<u64> {
        let mut keys = Vec::with_capacity(W + H + 1);

        // Per-column keys: encode (column_index, correct_count)
        for (x, &count) in self.col_counts.iter().enumerate() {
            let mut h = DefaultHasher::new();
            (0u8, x as u8, count).hash(&mut h);
            keys.push(h.finish());
        }

        // Per-row keys: encode (row_index, correct_count)
        for (y, &count) in self.row_counts.iter().enumerate() {
            let mut h = DefaultHasher::new();
            (1u8, y as u8, count).hash(&mut h);
            keys.push(h.finish());
        }

        // Total correct count bucket — coarse but useful early in training
        let mut h = DefaultHasher::new();
        (2u8, self.total_correct as u8).hash(&mut h);
        keys.push(h.finish());

        keys
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. EPISODIC MEMORY
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodicMemory {
    // Exact state → value (two grids wrong in the same cells share this)
    pub experiences: HashMap<u64, f32>,
    pub visit_counts: HashMap<u64, u32>,
    // Structural bucket → value (column/row correctness generalization)
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

    pub fn evaluate(&self, sig: u64, bucket_keys: &[u64]) -> f32 {
        // Exact match first
        if let Some(&v) = self.experiences.get(&sig) { return v; }

        // Structural generalization: average over known bucket values
        let mut total = 0.0;
        let mut known = 0;
        for key in bucket_keys {
            if let Some(&v) = self.bucket_values.get(key) {
                total += v;
                known += 1;
            }
        }
        if known > 0 { total / known as f32 } else { 0.0 }
    }

    pub fn learn(&mut self, sig: u64, bucket_keys: &[u64], target: f32) {
        // Exact state update
        let count = self.visit_counts.entry(sig).or_insert(0);
        *count += 1;
        let lr = 1.0 / (*count as f32).max(1.0);
        let val = self.experiences.entry(sig).or_insert(0.0);
        *val += lr * (target - *val);

        // Structural bucket updates
        for &key in bucket_keys {
            let bc = self.bucket_counts.entry(key).or_insert(0);
            *bc += 1;
            let blr = 1.0 / (*bc as f32).max(1.0);
            let bv = self.bucket_values.entry(key).or_insert(0.0);
            *bv += blr * (target - *bv);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. SOUL BRAIN — no FeaturePool, just memory
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct SoulBrain { pub memory: EpisodicMemory }

impl SoulBrain {
    pub fn new() -> Self { Self { memory: EpisodicMemory::new() } }

    pub fn save_and_merge(&self, path: &str) {
        let mut disk = if let Ok(f) = File::open(path) {
            serde_json::from_reader(f).unwrap_or_else(|_| SoulBrain::new())
        } else { SoulBrain::new() };

        for (&s, &v) in &self.memory.experiences { disk.memory.experiences.insert(s, v); }
        for (&s, &c) in &self.memory.visit_counts {
            let e = disk.memory.visit_counts.entry(s).or_insert(0);
            *e = (*e).max(c);
        }
        for (&k, &v) in &self.memory.bucket_values { disk.memory.bucket_values.insert(k, v); }
        for (&k, &c) in &self.memory.bucket_counts {
            let e = disk.memory.bucket_counts.entry(k).or_insert(0);
            *e = (*e).max(c);
        }

        let tmp = format!("{}.tmp", path);
        if let Ok(f) = File::create(&tmp) {
            if serde_json::to_writer(f, &disk).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
    }

    pub fn load(path: &str) -> Self {
        if let Ok(f) = File::open(path) {
            if let Ok(brain) = serde_json::from_reader(f) { return brain; }
            println!("🌱 Reincarnating — fresh memory.");
        } else {
            println!("🌱 No memory found. Starting fresh.");
        }
        SoulBrain::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. WORLD
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RaycastWorld {
    pub grid: Grid,
    pub target: Grid,
    pub steps: usize,
    pub max_steps: usize,
    pub brain: Arc<Mutex<SoulBrain>>,
    pub oracle: Arc<RaycastOracle>,
}

impl RaycastWorld {
    #[inline]
    pub fn decode_action(&self, action: usize) -> (usize, usize, Pixel) {
        let pv = action % 4;
        let x = (action / 4) % W;
        let y = (action / 4) / W;
        (x, y, match pv { 0 => Pixel::Empty, 1 => Pixel::Wall, 2 => Pixel::UpGravity, _ => Pixel::DownGravity })
    }
}

impl UniversalWorld for RaycastWorld {
    type State = Grid;
    type Action = usize;

    fn current_state(&self) -> Grid { self.grid.clone() }
    fn current_player(&self) -> i32 { 1 }

    fn step(&mut self, action: usize) -> Result<(), ()> {
        if self.steps >= self.max_steps { return Err(()); }
        let (x, y, pixel) = self.decode_action(action);
        self.grid.set(x, y, pixel);
        self.steps += 1;
        Ok(())
    }

    fn is_terminal(&self) -> bool {
        self.steps >= self.max_steps || self.grid.difference(&self.target) == 0
    }

    fn evaluate_path(&self, path: &[usize]) -> (f32, u64) {
        // Apply actions with backtracking — no world clone needed
        let mut sim = self.grid.clone();
        let mut undo: Vec<(usize, usize, Pixel)> = Vec::with_capacity(path.len());
        let mut sim_steps = self.steps;

        for &action in path {
            if sim_steps >= self.max_steps { break; }
            let (x, y, pixel) = self.decode_action(action);
            undo.push((x, y, sim.get(x, y)));
            sim.set(x, y, pixel);
            sim_steps += 1;
        }

        let oracle_score = self.oracle.consult(&sim);

        // Write this simulated future state into memory
        {
            let feats = CorrectnessFeatures::compute(&sim, &self.target);
            let sig = feats.exact_signature();
            let buckets = feats.bucket_keys();
            if let Ok(mut brain) = self.brain.try_lock() {
                brain.memory.learn(sig, &buckets, oracle_score);
            }
        }

        // Undo
        for (x, y, original) in undo.into_iter().rev() {
            sim.set(x, y, original);
        }

        (oracle_score, path.len() as u64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. POLICY — brain-informed priors via backtracking
// ─────────────────────────────────────────────────────────────────────────────

struct RaycastPolicy {
    brain: Arc<Mutex<SoulBrain>>,
    target: Grid,
}

impl CognitivePolicy<Grid, usize> for RaycastPolicy {
    fn evaluate(&self, state: &Grid) -> f32 {
        let b = self.brain.lock().unwrap();
        let feats = CorrectnessFeatures::compute(state, &self.target);
        b.memory.evaluate(feats.exact_signature(), &feats.bucket_keys())
    }

    fn priors(&self, state: &Grid) -> Vec<(usize, f32)> {
        let b = self.brain.lock().unwrap();
        let mut rng = rand::thread_rng();
        let total = W * H * 4;
        let mut working = state.clone();
        let mut priors = Vec::with_capacity(total);

        for action in 0..total {
            let pv = action % 4;
            let x = (action / 4) % W;
            let y = (action / 4) / W;
            let pixel = match pv { 0 => Pixel::Empty, 1 => Pixel::Wall, 2 => Pixel::UpGravity, _ => Pixel::DownGravity };

            // No-ops get minimum prior weight
            if working.get(x, y) == pixel { priors.push((action, 0.001)); continue; }

            let original = working.get(x, y);
            working.set(x, y, pixel);

            let feats = CorrectnessFeatures::compute(&working, &self.target);
            let score = b.memory.evaluate(feats.exact_signature(), &feats.bucket_keys());
            let noise = rng.gen_range(0.0_f32..0.001);
            priors.push((action, (score + noise).max(0.001)));

            working.set(x, y, original);
        }

        let sum: f32 = priors.iter().map(|(_, s)| s).sum();
        if sum > 0.0 { for (_, s) in &mut priors { *s /= sum; } }
        priors
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. TRAINING
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_raycast_training() {
    println!("\n🌸 SPATIAL EVOLUTION START: {}x{} Grid (Correctness Features)", W, H);

    let brain = SoulBrain::load("src/memories/raycast_memory.bin");
    let brain_arc = Arc::new(Mutex::new(brain));

    print!("Puzzles to solve: ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let num_puzzles: usize = input.trim().parse().unwrap_or(10);

    for g in 1..=num_puzzles {
        let mut rng = rand::thread_rng();
        let mut initial = Grid::new();
        initial.set(rng.gen_range(0..W), H - 1, Pixel::UpGravity);
        initial.set(rng.gen_range(0..W), 0, Pixel::DownGravity);
        initial.set(rng.gen_range(2..W-2), rng.gen_range(2..H-2), Pixel::Wall);

        let oracle = Arc::new(RaycastOracle::new(&initial));

        let policy = RaycastPolicy {
            brain: brain_arc.clone(),
            target: oracle.target.clone(),
        };

        let mut game = RaycastWorld {
            grid: initial.clone(), target: oracle.target.clone(),
            steps: 0, max_steps: 15,
            brain: brain_arc.clone(), oracle: oracle.clone(),
        };

        let mut total_surprise = 0.0;

        let config = ReasoningConfig::<usize> {
            simulations: 2000, max_depth: 6, max_program_len: 6, max_ops_per_candidate: 6,
            exploration_constant: 1.5, length_penalty: 0.0, loop_penalty: 0.5,
            action_space: vec![], // filled per step
            arena_capacity: 300_000,
        };

        while !game.is_terminal() {
            let oracle_eval = oracle.consult(&game.grid);

            // Measure surprise and learn current state
            {
                let mut b = brain_arc.lock().unwrap();
                let feats = CorrectnessFeatures::compute(&game.grid, &oracle.target);
                let sig = feats.exact_signature();
                let buckets = feats.bucket_keys();

                // Surprise = distance from solved
                total_surprise += 1.0 - oracle_eval;

                // Learn current state — brain needs the full value landscape
                b.memory.learn(sig, &buckets, oracle_eval);
            }

            // Only actions that change the grid — prunes no-ops from search
            let live_actions: Vec<usize> = (0..W * H * 4).filter(|&action| {
                let (x, y, pixel) = game.decode_action(action);
                game.grid.get(x, y) != pixel
            }).collect();

            let mut step_config = config.clone();
            step_config.action_space = if live_actions.is_empty() {
                (0..W * H * 4).collect()
            } else {
                live_actions
            };

            let (best_path, _) = solve_universal_with_stats(&game, &step_config, &policy);
            let chosen = best_path.and_then(|p| p.first().copied()).unwrap_or(0);
            game.step(chosen).unwrap();
        }

        let avg_surprise = total_surprise / game.steps.max(1) as f32;
        let solved = game.grid.difference(&game.target) == 0;

        // Save after every puzzle
        {
            let b = brain_arc.lock().unwrap();
            b.save_and_merge("src/memories/raycast_memory.bin");
        }

        println!(
            "Puzzle {} | Steps: {} | Surprise: {:.4} | Solved: {}",
            g, game.steps, avg_surprise, solved
        );
    }
    println!("✅ Training Complete.");
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. EXHIBITION
// ─────────────────────────────────────────────────────────────────────────────

pub fn run_raycast_exhibition() {
    println!("\n👁️✨ SPATIAL EXHIBITION ✨👁️\n");

    let brain = SoulBrain::load("src/memories/raycast_memory.bin");
    let brain_arc = Arc::new(Mutex::new(brain));

    let mut initial = Grid::new();
    initial.set(3, H - 1, Pixel::UpGravity);
    initial.set(7, 0, Pixel::DownGravity);
    initial.set(3, 4, Pixel::Wall);

    let oracle = Arc::new(RaycastOracle::new(&initial));

    let policy = RaycastPolicy {
        brain: brain_arc.clone(),
        target: oracle.target.clone(),
    };

    let mut game = RaycastWorld {
        grid: initial.clone(), target: oracle.target.clone(),
        steps: 0, max_steps: 15,
        brain: brain_arc.clone(), oracle: oracle.clone(),
    };

    println!("--- Initial ---\n{}", initial);
    println!("--- Target ---\n{}", oracle.target);

    let config = ReasoningConfig::<usize> {
        simulations: 10_000, max_depth: 6, max_program_len: 6, max_ops_per_candidate: 6,
        exploration_constant: 1.2, length_penalty: 0.0, loop_penalty: 1.0,
        action_space: vec![],
        arena_capacity: 1_000_000,
    };

    while !game.is_terminal() {
        let live_actions: Vec<usize> = (0..W * H * 4).filter(|&action| {
            let (x, y, pixel) = game.decode_action(action);
            game.grid.get(x, y) != pixel
        }).collect();

        let mut step_config = config.clone();
        step_config.action_space = if live_actions.is_empty() {
            (0..W * H * 4).collect()
        } else {
            live_actions
        };

        let (best_path, stats) = solve_universal_with_stats(&game, &step_config, &policy);
        let chosen = best_path.and_then(|p| p.first().copied()).unwrap_or(0);
        let (x, y, pixel) = game.decode_action(chosen);
        println!(
            "✨ [{:.0}ms] {:?} → ({},{}) | Oracle: {:.3}",
            stats.elapsed_ms, pixel, x, y, oracle.consult(&game.grid)
        );
        game.step(chosen).unwrap();
        println!("{}", game.grid);
    }

    let diff = game.grid.difference(&game.target);
    if diff == 0 {
        println!("✅ PERFECT RECONSTRUCTION");
    } else {
        println!("⚠️ Halted. Error: {} pixels", diff);
        println!("--- Achieved ---\n{}", game.grid);
        println!("--- Target ---\n{}", game.target);
    }
}
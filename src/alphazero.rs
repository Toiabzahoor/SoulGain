use crate::plasticity::{Event, IntoOpcode, Plasticity};
use crate::types::UVal;
use crate::vm::{CoreMind, Op, StepStatus};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// WorldModel trait – implement this to plug in any environment
// ─────────────────────────────────────────────────────────────────────────────

pub trait WorldModel {
    type Action: Clone + Copy;

    fn num_states(&self) -> usize;
    fn current_state(&self) -> usize;
    fn sense(&mut self) -> f64;
    fn step(&mut self, action: Self::Action) -> (usize, f64);
    fn flip_physics(&mut self) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// The New Universal World Search Traits 🌍
// ─────────────────────────────────────────────────────────────────────────────

pub trait UniversalWorld: Clone {
    type State;
    type Action: Clone + Copy + PartialEq;

    fn current_state(&self) -> Self::State;
    
    // 🌟 NEGAMAX CORE: Tells the tree whose turn it is. Defaults to 1 for single-player.
    fn current_player(&self) -> i32 {
        1
    } 
    
    fn step(&mut self, action: Self::Action) -> Result<(), ()>;
    
    fn is_terminal(&self) -> bool;
    
    // 🌟 Must return the ABSOLUTE SCORE from Player 1's perspective.
    fn evaluate_path(&self, path: &[Self::Action]) -> (f32, u64);
}

#[derive(Clone)]
pub struct VmWorld<'a> {
    pub pristine_core: CoreMind,
    pub current_core: CoreMind,
    pub task: &'a TaskSpec,
    pub config: &'a ReasoningConfig<Op>,
}

impl<'a> VmWorld<'a> {
    pub fn new(pristine_core: CoreMind, task: &'a TaskSpec, config: &'a ReasoningConfig<Op>) -> Self {
        let mut current_core = pristine_core.clone();
        if let Some(first) = task.train_cases.first() {
            current_core.reset(&first.input);
        }
        Self { pristine_core, current_core, task, config }
    }
}

impl<'a> UniversalWorld for VmWorld<'a> {
    type State = CoreMind;
    type Action = Op;

    fn current_state(&self) -> Self::State {
        self.current_core.clone()
    }

    fn step(&mut self, action: Self::Action) -> Result<(), ()> {
        match self.current_core.step(action) {
            StepStatus::Crash => Err(()),
            _ => Ok(()),
        }
    }

    fn is_terminal(&self) -> bool {
        self.current_core.is_halted()
    }

    fn evaluate_path(&self, path: &[Self::Action]) -> (f32, u64) {
        let score = aggregate_value(&self.pristine_core, self.task, path, self.config);
        let cost = (self.task.train_cases.len() * path.len().min(self.config.max_ops_per_candidate)) as u64;
        (score, cost)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReasoningConfig & MCTS solve
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ReasoningConfig<A> {
    pub simulations: u32,
    pub max_depth: usize,
    pub max_program_len: usize,
    pub max_ops_per_candidate: usize,
    pub exploration_constant: f32,
    pub length_penalty: f32,
    pub loop_penalty: f32,
    pub action_space: Vec<A>,
    pub arena_capacity: usize,
}

impl Default for ReasoningConfig<Op> {
    fn default() -> Self {
        Self {
            simulations: 4_000,
            max_depth: 8,
            max_program_len: 8,
            max_ops_per_candidate: 16,
            exploration_constant: 1.41,
            length_penalty: 0.05,
            loop_penalty: 0.5,
            action_space: vec![
                Op::Dup, Op::Add, Op::Sub, Op::Gt, Op::Not,
                Op::Jmp, Op::JmpIf, Op::Inc, Op::Halt,
            ],
            arena_capacity: 1_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TaskExample {
    pub input: Vec<UVal>,
    pub expected_output: Vec<UVal>,
}

#[derive(Clone, Debug)]
pub struct TaskSpec {
    pub train_cases: Vec<TaskExample>,
}

pub trait CognitivePolicy<S, A> {
    fn evaluate(&self, state: &S) -> f32;
    fn priors(&self, state: &S) -> Vec<(A, f32)>;
}

pub struct UniversalPolicy {
    target_tops: Vec<UVal>,
}

impl UniversalPolicy {
    pub fn from_task(task: &TaskSpec) -> Self {
        let mut target_tops = Vec::new();
        for case in &task.train_cases {
            if let Some(value) = case.expected_output.first() {
                if !target_tops.contains(value) {
                    target_tops.push(value.clone());
                }
            }
        }
        Self { target_tops }
    }

    fn all_ops() -> Vec<Op> {
        (0_i64..=32).filter_map(Op::from_i64).collect()
    }
}

impl CognitivePolicy<CoreMind, Op> for UniversalPolicy {
    fn evaluate(&self, state: &CoreMind) -> f32 {
        let output = state.extract_output();
        if output.is_empty() {
            return 0.0;
        }
        if let Some(top) = output.last() {
            if self.target_tops.iter().any(|target| target == top) {
                return 1.0;
            }
        }
        0.05
    }

    fn priors(&self, _state: &CoreMind) -> Vec<(Op, f32)> {
        let ops = Self::all_ops();
        let uniform = 1.0 / ops.len() as f32;
        ops.into_iter().map(|op| (op, uniform)).collect()
    }
}

pub struct NeuralPolicy<'a> {
    target_tops: Vec<UVal>,
    plasticity: &'a Plasticity,
    allowed_ops: Vec<Op>,
}

impl<'a> NeuralPolicy<'a> {
    pub fn new(task: &TaskSpec, plasticity: &'a Plasticity, allowed_ops: Vec<Op>) -> Self {
        let mut target_tops = Vec::new();
        for case in &task.train_cases {
            if let Some(value) = case.expected_output.first() {
                if !target_tops.contains(value) {
                    target_tops.push(value.clone());
                }
            }
        }
        Self {
            target_tops,
            plasticity,
            allowed_ops,
        }
    }
}

impl<'a> CognitivePolicy<CoreMind, Op> for NeuralPolicy<'a> {
    fn evaluate(&self, state: &CoreMind) -> f32 {
        let output = state.extract_output();
        if output.is_empty() {
            return 0.0;
        }
        if let Some(top) = output.last() {
            if self.target_tops.iter().any(|target| target == top) {
                return 1.0;
            }
        }
        0.05
    }

    fn priors(&self, state: &CoreMind) -> Vec<(Op, f32)> {
        let hash = state.state_hash();
        let event = Event::ContextWithState {
            data: [0; 16],
            state_hash: hash,
        };
        self.plasticity.get_op_distribution(event, &self.allowed_ops)
    }
}

#[derive(Clone, Copy, Debug)]
struct MctsNodeFlat<A> {
    visits: u32,
    value_sum: f32,
    prior: f32,
    action: Option<A>,
    parent: u32,
    children_head: u32,
    num_children: u16,
    parent_player: i32, // The player who made the choice to reach this node
}

impl<A> MctsNodeFlat<A> {
    fn new(action: Option<A>, parent: u32, prior: f32, parent_player: i32) -> Self {
        Self {
            visits: 0,
            value_sum: 0.0,
            prior,
            action,
            parent,
            children_head: 0,
            num_children: 0,
            parent_player,
        }
    }

    fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolveStats {
    pub simulations_run: u32,
    pub elapsed_ms: f64,
    pub sims_per_sec: f64,
    pub nodes_allocated: usize,
    pub avg_selection_depth: f64,
    pub total_op_cycles: u64,
    pub op_cycles_per_sim: f64,
    pub crash_count: u32,
    pub best_score: f32,
    pub solved_early: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy Wrappers
// ─────────────────────────────────────────────────────────────────────────────

pub fn solve<P>(
    root_state: &CoreMind,
    task: &TaskSpec,
    config: &ReasoningConfig<Op>,
    policy: &P,
) -> Option<Vec<Op>>
where
    P: CognitivePolicy<CoreMind, Op>,
{
    solve_with_stats(root_state, task, config, policy).0
}

pub fn solve_with_stats<P>(
    root_state: &CoreMind,
    task: &TaskSpec,
    config: &ReasoningConfig<Op>,
    policy: &P,
) -> (Option<Vec<Op>>, SolveStats)
where
    P: CognitivePolicy<CoreMind, Op>,
{
    let world = VmWorld::new(root_state.clone(), task, config);
    solve_universal_with_stats(&world, config, policy)
}

// ─────────────────────────────────────────────────────────────────────────────
// Universal Engine! (True Absolute-Score Negamax)
// ─────────────────────────────────────────────────────────────────────────────

pub fn solve_universal<W, P, S, A>(
    root_world: &W,
    config: &ReasoningConfig<A>,
    policy: &P,
) -> Option<Vec<A>>
where
    W: UniversalWorld<State = S, Action = A>,
    P: CognitivePolicy<S, A>,
    A: Clone + Copy + PartialEq,
{
    solve_universal_with_stats(root_world, config, policy).0
}

pub fn solve_universal_with_stats<W, P, S, A>(
    root_world: &W,
    config: &ReasoningConfig<A>,
    policy: &P,
) -> (Option<Vec<A>>, SolveStats)
where
    W: UniversalWorld<State = S, Action = A>,
    P: CognitivePolicy<S, A>,
    A: Clone + Copy + PartialEq,
{
    use std::time::Instant;
    let t0 = Instant::now();

    let mut tree_storage = Vec::with_capacity(config.arena_capacity);
    tree_storage.push(MctsNodeFlat::new(None, 0, 1.0, root_world.current_player()));
    let root_node_idx = 0;

    let mut best_eureka_program: Option<(Vec<A>, f32)> = None;
    let mut sims_run: u32 = 0;
    let mut total_depth: u64 = 0;
    let mut total_op_cycles: u64 = 0;
    let mut crash_count: u32 = 0;

    for _ in 0..config.simulations {
        sims_run += 1;
        let mut node_idx = root_node_idx;
        let mut op_path = Vec::with_capacity(config.max_program_len);

        // Selection
        let mut depth = 0;
        while tree_storage[node_idx].children_head != 0 && depth < config.max_program_len {
            let current = &tree_storage[node_idx];
            let mut best_child_idx = 0;
            let mut best_score = -f32::INFINITY;
            let sqrt_visits = (current.visits.max(1) as f32).sqrt();

            let head = current.children_head as usize;
            let end = head + current.num_children as usize;

            for child_idx in head..end {
                let child = &tree_storage[child_idx];
                let score = child.q_value()
                    + config.exploration_constant
                        * child.prior
                        * (sqrt_visits / (1.0 + child.visits as f32));
                if score > best_score {
                    best_score = score;
                    best_child_idx = child_idx;
                }
            }
            node_idx = best_child_idx;
            if let Some(act) = tree_storage[node_idx].action {
                op_path.push(act);
            }
            depth += 1;
        }
        total_depth += depth as u64;

        // Rollout
        let mut state = root_world.clone();
        let mut crashed = false;
        for &op in &op_path {
            total_op_cycles += 1;
            if state.step(op).is_err() {
                crashed = true;
                crash_count += 1;
                break;
            }
        }

        let (absolute_value, terminal) = if crashed {
            // The player who made the illegal move gets punished.
            let leaf_parent_player = tree_storage[node_idx].parent_player;
            let abs_crash_score = if leaf_parent_player == 1 { -1.0 } else { 1.0 };
            (abs_crash_score, true)
        } else if state.is_terminal() || depth >= config.max_program_len {
            let (eval_score, eval_cost) = root_world.evaluate_path(&op_path);
            total_op_cycles += eval_cost;
            (eval_score, true)
        } else {
            let current_player = state.current_player();
            let priors = policy.priors(&state.current_state());
            let valid_priors: Vec<(A, f32)> = priors
                .into_iter()
                .filter(|(op, _)| config.action_space.contains(op))
                .collect();

            if !valid_priors.is_empty()
                && tree_storage.len() + valid_priors.len() < tree_storage.capacity()
            {
                let start_idx = tree_storage.len() as u32;
                let count = valid_priors.len() as u16;
                for (op, prior) in valid_priors {
                    // Assign the player whose turn it currently is as the parent_player for the child
                    tree_storage.push(MctsNodeFlat::new(
                        Some(op),
                        node_idx as u32,
                        prior,
                        current_player,
                    ));
                }
                tree_storage[node_idx].children_head = start_idx;
                tree_storage[node_idx].num_children = count;
                (policy.evaluate(&state.current_state()), false) // Absolute evaluation fallback
            } else {
                (0.0, true)
            }
        };

        // 🌟 ABSOLUTE SCORE BACKPROPAGATION 🌟
        let mut curr = node_idx;
        loop {
            let node = &mut tree_storage[curr as usize];
            node.visits += 1;
            
            // If Player 1 chose this node, they want absolute_value to be High.
            // If Player -1 chose this node, they want absolute_value to be Low.
            let aligned_value = if node.parent_player == 1 {
                absolute_value
            } else {
                -absolute_value
            };
            
            node.value_sum += aligned_value;

            if curr == root_node_idx {
                break;
            }
            curr = node.parent as usize;
        }

        if terminal {
            let root_player = tree_storage[root_node_idx].parent_player;
            let root_aligned = if root_player == 1 { absolute_value } else { -absolute_value };
            
            if root_aligned > 0.0 {
                let update = match &best_eureka_program {
                    None => true,
                    Some((_, score)) => root_aligned > *score,
                };
                if update {
                    best_eureka_program = Some((op_path.clone(), root_aligned));
                }
            }
        }
    }

    let mut robust_path = Vec::new();
    let mut curr_node = root_node_idx as usize;

    while tree_storage[curr_node].children_head != 0 && tree_storage[curr_node].num_children > 0 {
        let head = tree_storage[curr_node].children_head as usize;
        let end = head + tree_storage[curr_node].num_children as usize;

        let mut best_child = head;
        let mut max_visits = 0;
        let mut best_q = -f32::INFINITY;

        for i in head..end {
            let child = &tree_storage[i];
            if child.visits > max_visits || (child.visits == max_visits && child.q_value() > best_q)
            {
                max_visits = child.visits;
                best_q = child.q_value();
                best_child = i;
            }
        }
        
        if max_visits == 0 {
            break;
        }
        if let Some(act) = tree_storage[best_child].action {
            robust_path.push(act);
        }
        curr_node = best_child;
    }

    let robust_score = tree_storage[root_node_idx as usize].q_value();
    let final_program = if let Some((eureka_path, eureka_score)) = best_eureka_program {
        if eureka_score >= 0.999 {
            Some((eureka_path, eureka_score))
        } else {
            Some((robust_path, robust_score))
        }
    } else if !robust_path.is_empty() {
        Some((robust_path, robust_score))
    } else {
        None
    };

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let sims_per_sec = if elapsed_ms > 0.0 {
        sims_run as f64 / (elapsed_ms / 1000.0)
    } else {
        f64::INFINITY
    };

    let best_score = final_program.as_ref().map(|(_, s)| *s).unwrap_or(0.0);

    (
        final_program.map(|(p, _)| p),
        SolveStats {
            simulations_run: sims_run,
            elapsed_ms,
            sims_per_sec,
            nodes_allocated: tree_storage.len(),
            avg_selection_depth: total_depth as f64 / sims_run.max(1) as f64,
            total_op_cycles,
            op_cycles_per_sim: total_op_cycles as f64 / sims_run.max(1) as f64,
            crash_count,
            best_score,
            solved_early: false,
        },
    )
}

fn aggregate_value(
    root_state: &CoreMind,
    task: &TaskSpec,
    program: &[Op],
    config: &ReasoningConfig<Op>,
) -> f32 {
    if program.is_empty() { return 0.0; }
    let mut total = 0.0;
    let mut matched = 0usize;
    let mut runner = root_state.clone();

    for case in &task.train_cases {
        runner.reset(&case.input);
        let mut steps = 0;
        let mut crashed = false;
        
        while runner.ip() < program.len() && steps < config.max_ops_per_candidate {
            match runner.step(program[runner.ip()]) {
                StepStatus::Halt => break,
                StepStatus::Crash => { crashed = true; break; }
                _ => {}
            }
            steps += 1;
        }
        
        if crashed { return -0.5; }
        
       // Inside src/alphazero.rs -> fn aggregate_value()

        let output = runner.extract_output();
        if output == case.expected_output {
            total += 1.0;
            matched += 1;
        } else if let (Some(UVal::Number(got)), Some(UVal::Number(want))) =
            (output.last(), case.expected_output.first())
        {
            // 🌟 STRICT REGRESSION: No more partial credit for "Halt"!
            // If it is off by 1.0 (guessing 0 when the answer is 1), it gets 0.0 points.
            // If it is totally backwards (guessing 1 when the answer is -1), it gets -1.0 points!
            let diff = (*got - *want).abs();
            let accuracy = 1.0 - diff; 
            total += accuracy as f32;
            
            if diff < 0.15 { matched += 1; }
        }
    }
    
    if matched == task.train_cases.len() { return 1.0; }
    
    let mut score = total / task.train_cases.len() as f32;
    score -= config.length_penalty
        * 0.5
        * (program.len().min(config.max_program_len) as f32 / config.max_program_len.max(1) as f32);
    score.clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// WorldConfig / WorldGenerator  (built-in WorldModel implementation)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct WorldConfig {
    pub hidden_states: usize,
    pub entropy: f64,
    pub observation_noise: f64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            hidden_states: 6,
            entropy: 0.15,
            observation_noise: 0.02,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorldGenerator {
    state: usize,
    pub transitions: Vec<Vec<f64>>,
    latent_values: Vec<f64>,
    rng: u64,
    noise: f64,
}

impl WorldGenerator {
    pub fn new(config: &WorldConfig) -> Self {
        let n = config.hidden_states.max(2);
        let mut transitions = vec![vec![0.0; n]; n];
        for (s, row) in transitions.iter_mut().enumerate() {
            row[s] = (1.0 - config.entropy).clamp(0.0, 1.0);
            row[(s + 1) % n] += config.entropy * 0.7;
            row[(s + 2) % n] += config.entropy * 0.3;
        }
        let latent_values = (0..n)
            .map(|idx| {
                let base = (idx as f64 + 1.0) * 1.7;
                if (idx + 1) % 7 == 0 {
                    base + 11.0
                } else {
                    base
                }
            })
            .collect();
        Self {
            state: 0,
            transitions,
            latent_values,
            rng: 0xD1CE_CAFE_BAAD_F00D,
            noise: config.observation_noise,
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng as f64 / u64::MAX as f64).clamp(0.0, 1.0)
    }

    pub fn sample_distribution(&mut self, probs: &[f64]) -> usize {
        let r = self.next_f64();
        let mut cdf = 0.0;
        for (idx, p) in probs.iter().enumerate() {
            cdf += *p;
            if r <= cdf {
                return idx;
            }
        }
        probs.len().saturating_sub(1)
    }
}

impl WorldModel for WorldGenerator {
    type Action = Op;
    
    fn num_states(&self) -> usize {
        self.transitions.len()
    }
    
    fn current_state(&self) -> usize {
        self.state
    }
    
    fn sense(&mut self) -> f64 {
        self.latent_values[self.state] + (self.next_f64() - 0.5) * 2.0 * self.noise
    }
    
    fn step(&mut self, action: Op) -> (usize, f64) {
        let action_shift = (action as usize) % self.transitions.len();
        let base_row = &self.transitions[self.state];
        let mut rolled = vec![0.0; base_row.len()];
        for (idx, prob) in base_row.iter().enumerate() {
            rolled[(idx + action_shift) % base_row.len()] += *prob;
        }
        let next = self.sample_distribution(&rolled);
        self.state = next;
        (next, self.sense())
    }
    
    fn flip_physics(&mut self) {
        for row in self.transitions.iter_mut() {
            row.reverse();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ForwardModel  (internal world-model learned by the agent)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForwardModel {
    transition_counts: Vec<Vec<Vec<f64>>>,
    effort_decay: f64,
}

impl ForwardModel {
    pub fn new(states: usize, actions: usize) -> Self {
        Self {
            transition_counts: vec![vec![vec![1.0; states]; actions]; states],
            effort_decay: 0.0,
        }
    }

    pub fn predict_distribution(&self, state: usize, action_idx: usize) -> Vec<f64> {
        let row = &self.transition_counts[state][action_idx];
        let sum: f64 = row.iter().sum();
        if sum <= f64::EPSILON {
            return vec![1.0 / row.len() as f64; row.len()];
        }
        row.iter().map(|v| v / sum).collect()
    }

    pub fn expected_surprisal(&self, state: usize, action_idx: usize) -> f64 {
        self.predict_distribution(state, action_idx)
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| -p * p.ln())
            .sum()
    }

    pub fn update(&mut self, from: usize, action_idx: usize, to: usize, surprisal: f64) {
        let novelty = (1.0 + surprisal).clamp(1.0, 4.0);
        self.transition_counts[from][action_idx][to] += novelty;
        self.effort_decay = (self.effort_decay * 0.96 + novelty * 0.04).clamp(0.0, 10.0);
    }

    pub fn metabolic_cost(&self, action_idx: usize) -> f64 {
        self.effort_decay * (1.0 + action_idx as f64 * 0.001)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), self)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        serde_json::from_reader(BufReader::new(File::open(path)?))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ActiveInference – generic over any WorldModel
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ActiveInferenceConfig<A> {
    pub steps: usize,
    pub imagination_rollouts: usize,
    pub rollout_depth: usize,
    pub curiosity_weight: f64,
    pub effort_weight: f64,
    pub action_space: Vec<A>,
}

impl Default for ActiveInferenceConfig<Op> {
    fn default() -> Self {
        Self {
            steps: 300,
            imagination_rollouts: 24,
            rollout_depth: 6,
            curiosity_weight: 0.3,
            effort_weight: 0.08,
            action_space: vec![
                Op::In, Op::Out, Op::Add, Op::Sub, Op::Dup, Op::Swap, Op::Drop,
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ActiveInferenceReport {
    pub average_surprisal: f64,
    pub average_free_energy: f64,
    pub unique_states_seen: usize,
    pub elapsed_ms: f64,
    pub steps_per_sec: f64,
    pub total_imagination_cycles: u64,
    pub imagination_cycles_per_step: f64,
}

pub fn run_active_inference_episode<W: WorldModel>(
    world: &mut W,
    plasticity: &Plasticity,
    config: &ActiveInferenceConfig<W::Action>,
) -> ActiveInferenceReport
where
    W::Action: IntoOpcode,
{
    use std::time::Instant;
    let t0 = Instant::now();

    let mut model = ForwardModel::new(world.num_states(), config.action_space.len());
    let mut running_surprisal = 0.0;
    let mut running_fe = 0.0;
    let mut seen_states = std::collections::HashSet::new();
    let mut total_imagination_cycles: u64 = 0;

    for _ in 0..config.steps {
        total_imagination_cycles += (config.action_space.len()
            * config.imagination_rollouts
            * config.rollout_depth) as u64;
        
        let state = world.current_state();
        seen_states.insert(state);

        let action_idx = imagine_best_action::<W>(state, &model, config);
        let action = config.action_space[action_idx];

        let predicted = model.predict_distribution(state, action_idx);
        let predicted_peak = predicted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(state);

        let (next_state, _) = world.step(action);
        let prob = predicted.get(next_state).copied().unwrap_or(1e-6).max(1e-6);
        let surprisal = (-prob.ln()).clamp(0.0, 10.0);
        let prediction_error = if predicted_peak == next_state { 0.0 } else { 1.0 };

        model.update(state, action_idx, next_state, surprisal);

        let expected_uncertainty = model.expected_surprisal(state, action_idx);
        running_surprisal += surprisal;
        running_fe += surprisal - config.curiosity_weight * expected_uncertainty
            + config.effort_weight * model.metabolic_cost(action_idx);

        plasticity.observe_batch(vec![
            Event::ContextWithState {
                data: [0; 16],
                state_hash: ((state as u64) << 32) | next_state as u64,
            },
            Event::Opcode {
                opcode: action.into_opcode(),
                stack_depth: 1,
            },
            Event::Surprisal((surprisal * 100.0) as u16),
            Event::Surprisal((prediction_error * 1000.0) as u16),
        ]);
    }

    let denom = config.steps.max(1) as f64;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    
    ActiveInferenceReport {
        average_surprisal: running_surprisal / denom,
        average_free_energy: running_fe / denom,
        unique_states_seen: seen_states.len(),
        elapsed_ms,
        steps_per_sec: if elapsed_ms > 0.0 {
            config.steps as f64 / (elapsed_ms / 1000.0)
        } else {
            f64::INFINITY
        },
        total_imagination_cycles,
        imagination_cycles_per_step: total_imagination_cycles as f64 / config.steps.max(1) as f64,
    }
}

fn imagine_best_action<W: WorldModel>(
    state: usize,
    model: &ForwardModel,
    config: &ActiveInferenceConfig<W::Action>,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f64::INFINITY;

    for action_idx in 0..config.action_space.len() {
        let mut aggregate = 0.0;
        let base_dist = model.predict_distribution(state, action_idx);
        
        for rollout in 0..config.imagination_rollouts {
            let mut rollout_state =
                sample_from_distribution(&base_dist, rollout as u64 + action_idx as u64 + 1);
            let mut score = 0.0;
            
            for depth in 0..config.rollout_depth {
                let imagined_action = (action_idx + depth) % config.action_space.len();
                score += (config.effort_weight * model.metabolic_cost(imagined_action))
                    - (config.curiosity_weight * model.expected_surprisal(rollout_state, imagined_action));
                rollout_state = sample_from_distribution(
                    &model.predict_distribution(rollout_state, imagined_action),
                    (rollout + depth + 1) as u64,
                );
            }
            aggregate += score;
        }

        let mean = aggregate / config.imagination_rollouts.max(1) as f64;
        if mean < best_score {
            best_score = mean;
            best_idx = action_idx;
        }
    }
    best_idx
}

pub fn sample_from_distribution(dist: &[f64], seed: u64) -> usize {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let mut target = (x as f64 / u64::MAX as f64).clamp(0.0, 1.0);
    
    for (idx, p) in dist.iter().enumerate() {
        target -= *p;
        if target <= 0.0 {
            return idx;
        }
    }
    dist.len().saturating_sub(1)
}
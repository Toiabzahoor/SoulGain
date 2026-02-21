use crate::plasticity::{Event, Plasticity};
use crate::types::UVal;
use crate::vm::{CoreMind, Op, StepStatus};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ReasoningConfig {
    pub simulations: u32,
    pub max_depth: usize,
    pub max_program_len: usize,
    pub max_ops_per_candidate: usize,
    pub exploration_constant: f32,
    pub length_penalty: f32,
    pub loop_penalty: f32,
    pub action_space: Vec<Op>,
    pub arena_capacity: usize,
}

impl Default for ReasoningConfig {
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
                Op::Dup,
                Op::Add,
                Op::Sub,
                Op::Gt,
                Op::Not,
                Op::Jmp,
                Op::JmpIf,
                Op::Inc,
                Op::Halt,
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

pub trait CognitivePolicy {
    fn evaluate(&self, state: &CoreMind) -> f32;
    fn priors(&self, state: &CoreMind) -> Vec<(Op, f32)>;
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

impl CognitivePolicy for UniversalPolicy {
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

impl<'a> CognitivePolicy for NeuralPolicy<'a> {
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
        self.plasticity
            .get_op_distribution(event, &self.allowed_ops)
    }
}

#[derive(Clone, Copy, Debug)]
struct MctsNodeFlat {
    visits: u32,
    value_sum: f32,
    prior: f32,
    op: Op,
    parent: u32,
    children_head: u32,
    num_children: u16,
}

impl MctsNodeFlat {
    fn new(op: Op, parent: u32, prior: f32) -> Self {
        Self {
            visits: 0,
            value_sum: 0.0,
            prior,
            op,
            parent,
            children_head: 0,
            num_children: 0,
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

pub fn solve(
    root_state: &CoreMind,
    task: &TaskSpec,
    config: &ReasoningConfig,
    policy: &dyn CognitivePolicy,
) -> Option<Vec<Op>> {
    if task.train_cases.is_empty() {
        return None;
    }

    let mut tree_storage = Vec::with_capacity(config.arena_capacity);
    tree_storage.push(MctsNodeFlat::new(Op::Halt, 0, 1.0));
    let root_node_idx = 0;

    let mut best_program: Option<(Vec<Op>, f32)> = None;

    for _ in 0..config.simulations {
        let mut node_idx = root_node_idx;
        let mut op_path = Vec::with_capacity(config.max_program_len);

        let mut depth = 0;
        while tree_storage[node_idx].children_head != 0 && depth < config.max_program_len {
            let current = &tree_storage[node_idx];
            let mut best_child_idx = 0;
            let mut best_score = -f32::INFINITY;
            let parent_visits = current.visits.max(1) as f32;
            let sqrt_visits = parent_visits.sqrt();

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
            op_path.push(tree_storage[node_idx].op);
            depth += 1;
        }

        let mut state = root_state.clone();
        let mut crashed = false;
        for &op in &op_path {
            if matches!(state.step(op), StepStatus::Crash) {
                crashed = true;
                break;
            }
        }

        let (value, terminal) = if crashed {
            (-0.1, true)
        } else if state.is_halted() || depth >= config.max_program_len {
            (aggregate_value(root_state, task, &op_path, config), true)
        } else {
            let priors = policy.priors(&state);
            let valid_priors: Vec<(Op, f32)> = priors
                .into_iter()
                .filter(|(op, _)| config.action_space.contains(op))
                .collect();

            if !valid_priors.is_empty()
                && tree_storage.len() + valid_priors.len() < tree_storage.capacity()
            {
                let start_idx = tree_storage.len() as u32;
                let count = valid_priors.len() as u16;
                for (op, prior) in valid_priors {
                    tree_storage.push(MctsNodeFlat::new(op, node_idx as u32, prior));
                }
                tree_storage[node_idx].children_head = start_idx;
                tree_storage[node_idx].num_children = count;
                (policy.evaluate(&state), false)
            } else {
                (0.0, true)
            }
        };

        let mut curr = node_idx;
        loop {
            let node = &mut tree_storage[curr as usize];
            node.visits += 1;
            node.value_sum += value;
            if curr == root_node_idx {
                break;
            }
            curr = node.parent as usize;
        }

        if terminal && value > 0.0 {
            let update = match &best_program {
                None => true,
                Some((_, score)) => value > *score,
            };
            if update {
                best_program = Some((op_path, value));
                if value >= 0.999 {
                    break;
                }
            }
        }
    }

    best_program.map(|(p, _)| p)
}

fn aggregate_value(
    root_state: &CoreMind,
    task: &TaskSpec,
    program: &[Op],
    config: &ReasoningConfig,
) -> f32 {
    if program.is_empty() {
        return 0.0;
    }

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
                StepStatus::Crash => {
                    crashed = true;
                    break;
                }
                _ => {}
            }
            steps += 1;
        }

        if crashed {
            return -0.5;
        }

        let output = runner.extract_output();
        if output == case.expected_output {
            total += 1.0;
            matched += 1;
        } else {
            if let (Some(UVal::Number(got)), Some(UVal::Number(want))) =
                (output.last(), case.expected_output.first())
            {
                if *got != 0.0 && *want != 0.0 {
                    if want % got == 0.0 {
                        total += 0.4;
                    } else if got % want == 0.0 {
                        total += 0.2;
                    }
                }
                if *got == 0.0 {
                    total += 0.1;
                }
            }
        }
    }

    if matched == task.train_cases.len() {
        return 1.0;
    }

    let mut score = total / task.train_cases.len() as f32;
    let length_soft_penalty = config.length_penalty
        * 0.5
        * (program.len().min(config.max_program_len) as f32 / config.max_program_len.max(1) as f32);
    score -= length_soft_penalty;

    score.clamp(-1.0, 1.0)
}

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
    transitions: Vec<Vec<f64>>,
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

    pub fn current_state(&self) -> usize {
        self.state
    }

    pub fn sense(&mut self) -> f64 {
        let noise = (self.next_f64() - 0.5) * 2.0 * self.noise;
        self.latent_values[self.state] + noise
    }

    pub fn step(&mut self, action: Op) -> (usize, f64) {
        let action_shift = (action as usize) % self.transitions.len();
        let base_row = &self.transitions[self.state];
        let mut rolled = vec![0.0; base_row.len()];
        for (idx, prob) in base_row.iter().enumerate() {
            let target = (idx + action_shift) % base_row.len();
            rolled[target] += *prob;
        }
        let next = self.sample_distribution(&rolled);
        self.state = next;
        (next, self.sense())
    }

    // NEW: The "Catastrophe" tool
    pub fn flip_physics(&mut self) {
        for row in self.transitions.iter_mut() {
            row.reverse();
        }
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

    fn next_f64(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng as f64 / u64::MAX as f64).clamp(0.0, 1.0)
    }
}

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
        let probs = self.predict_distribution(state, action_idx);
        probs
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
        let file = File::create(path)?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let model: Self = serde_json::from_reader(BufReader::new(file))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(model)
    }
}

#[derive(Clone, Debug)]
pub struct ActiveInferenceConfig {
    pub steps: usize,
    pub imagination_rollouts: usize,
    pub rollout_depth: usize,
    pub curiosity_weight: f64,
    pub effort_weight: f64,
    pub action_space: Vec<Op>,
}

impl Default for ActiveInferenceConfig {
    fn default() -> Self {
        Self {
            steps: 300,
            imagination_rollouts: 96,
            rollout_depth: 8,
            curiosity_weight: 0.3,
            effort_weight: 0.08,
            action_space: vec![
                Op::In,
                Op::Out,
                Op::Add,
                Op::Sub,
                Op::Dup,
                Op::Swap,
                Op::Drop,
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ActiveInferenceReport {
    pub average_surprisal: f64,
    pub average_free_energy: f64,
    pub unique_states_seen: usize,
}

pub fn run_active_inference_episode(
    world: &mut WorldGenerator,
    plasticity: &Plasticity,
    config: &ActiveInferenceConfig,
) -> ActiveInferenceReport {
    let mut model = ForwardModel::new(world.transitions.len(), config.action_space.len());
    let mut running_surprisal = 0.0;
    let mut running_fe = 0.0;
    let mut seen_states = std::collections::HashSet::new();

    for _ in 0..config.steps {
        let state = world.current_state();
        seen_states.insert(state);

        let action_idx = imagine_best_action(state, &model, config);
        let action = config.action_space[action_idx];

        let predicted = model.predict_distribution(state, action_idx);
        let predicted_peak = predicted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(state);

        let (next_state, _obs) = world.step(action);
        let prob = predicted.get(next_state).copied().unwrap_or(1e-6).max(1e-6);
        let surprisal = (-prob.ln()).clamp(0.0, 10.0);
        let prediction_error = if predicted_peak == next_state {
            0.0
        } else {
            1.0
        };

        model.update(state, action_idx, next_state, surprisal);

        let expected_uncertainty = model.expected_surprisal(state, action_idx);
        let free_energy = surprisal - config.curiosity_weight * expected_uncertainty
            + config.effort_weight * model.metabolic_cost(action_idx);

        running_surprisal += surprisal;
        running_fe += free_energy;

        let state_hash = ((state as u64) << 32) | next_state as u64;
        plasticity.observe_batch(vec![
            Event::ContextWithState {
                data: [0; 16],
                state_hash,
            },
            Event::Opcode {
                opcode: action as i64,
                stack_depth: 1,
            },
            Event::Surprisal((surprisal * 100.0) as u16),
            Event::Surprisal((prediction_error * 1000.0) as u16),
        ]);
    }

    let denom = config.steps.max(1) as f64;
    ActiveInferenceReport {
        average_surprisal: running_surprisal / denom,
        average_free_energy: running_fe / denom,
        unique_states_seen: seen_states.len(),
    }
}

fn imagine_best_action(
    state: usize,
    model: &ForwardModel,
    config: &ActiveInferenceConfig,
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
                let uncertainty = model.expected_surprisal(rollout_state, imagined_action);
                let effort = model.metabolic_cost(imagined_action);
                
                // FIXED: MINIMIZING score means seeking HIGH uncertainty (negative contribution)
                score += (config.effort_weight * effort) - (config.curiosity_weight * uncertainty);
                
                let dist = model.predict_distribution(rollout_state, imagined_action);
                rollout_state = sample_from_distribution(&dist, (rollout + depth + 1) as u64);
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
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

pub type ActionId = i64;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub domain_tag: Option<u64>,
    pub features: [u64; 16],
    pub context_hash: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConceptPattern {
    pub expected_features: [u64; 16],
    pub feature_mask: [u64; 16], // 1 = bit matters, 0 = bit is ignored
    pub confidence: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TaskStats {
    pub attempts: u64,
    pub successes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionStats {
    pub attempts: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_reward_delta: f64,
    pub base_confidence: f64,
    pub last_used_tick: u64,
}

impl Default for ActionStats {
    fn default() -> Self {
        Self {
            attempts: 0,
            successes: 0,
            failures: 0,
            avg_reward_delta: 0.0,
            base_confidence: 0.5,
            last_used_tick: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureSignature {
    pub context_hash: u64,
    pub domain_tag: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionMetadata {
    pub action_id: ActionId,
    pub pattern: ConceptPattern,
    pub stats: ActionStats,
    pub per_task: HashMap<u64, TaskStats>,
    pub recent_failures: VecDeque<FailureSignature>,
}

#[derive(Clone, Debug)]
pub struct ActionOutcome {
    pub success: bool,
    pub reward_delta: f64,
    pub used_tick: u64,
    pub domain_tag: Option<u64>,
    pub features: [u64; 16],
    pub context_hash: u64,
}

#[derive(Clone, Debug)]
pub struct IntuitionWeights {
    pub w_match: f64,
    pub w_success: f64,
    pub w_reward: f64,
    pub w_conf: f64,
    pub w_decay: f64,
    pub w_explore: f64,
    pub w_task: f64,
}

impl Default for IntuitionWeights {
    fn default() -> Self {
        Self {
            w_match: 0.40,
            w_success: 0.20,
            w_reward: 0.15,
            w_conf: 0.10,
            w_decay: 0.05,
            w_explore: 0.05,
            w_task: 0.05,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct IntuitionMemory {
    pub action_meta: HashMap<ActionId, ActionMetadata>,
}

#[derive(Clone)]
pub struct UniversalIntuition {
    pub memory: IntuitionMemory, 
    pub weights: IntuitionWeights,
    pub gate_threshold: f64,
    pub deterministic_mode: bool,
    pub decay_tau_ticks: f64,
    pub last_match_score: f64,
    rng_state: u64,
}

impl Default for UniversalIntuition {
    fn default() -> Self {
        Self {
            memory: IntuitionMemory::default(),
            weights: IntuitionWeights::default(),
            gate_threshold: 0.2,
            deterministic_mode: false,
            decay_tau_ticks: 12.0,
            last_match_score: 0.0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

impl UniversalIntuition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
        bincode::serialize_into(BufWriter::new(file), &self.memory)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let file = File::open(path)?;
        let memory: IntuitionMemory = bincode::deserialize_from(BufReader::new(file))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.memory = memory;
        Ok(())
    }

    pub fn build_context(&self, features: [u64; 16], domain_tag: Option<u64>) -> ContextSnapshot {
        let mut h = 0xcbf29ce484222325u64;
        for &f in &features {
            h ^= f;
            h = h.wrapping_mul(0x100000001b3);
        }
        ContextSnapshot { domain_tag, features, context_hash: h }
    }

    pub fn ensure_action_known(&mut self, action_id: ActionId) {
        self.memory.action_meta
            .entry(action_id)
            .or_insert_with(|| ActionMetadata {
                action_id,
                pattern: ConceptPattern {
                    expected_features: [0; 16],
                    feature_mask: [0; 16],
                    confidence: 0.5,
                },
                stats: ActionStats::default(),
                per_task: HashMap::new(),
                recent_failures: VecDeque::with_capacity(16),
            });
    }

    pub fn bootstrap_pattern_if_empty(&mut self, action_id: ActionId, ctx: &ContextSnapshot) {
        self.ensure_action_known(action_id);
        if let Some(meta) = self.memory.action_meta.get_mut(&action_id) {
            if meta.pattern.feature_mask == [0; 16] {
                meta.pattern.expected_features = ctx.features;
                meta.pattern.feature_mask = [u64::MAX; 16]; 
            }
        }
    }

    // 🌟 FOR ALPHAZERO (CHESS)
    pub fn get_action_distribution<A: Copy, F: Fn(A) -> ActionId>(
        &mut self,
        ctx: &ContextSnapshot,
        allowed_actions: &[A],
        id_extractor: F,
    ) -> Vec<(A, f32)> {
        let mut priors = Vec::with_capacity(allowed_actions.len());
        let mut sum = 0.0;
        let temperature = 2.0;

        for &action in allowed_actions {
            let action_id = id_extractor(action);
            self.bootstrap_pattern_if_empty(action_id, ctx);

            let score = if let Some(meta) = self.memory.action_meta.get(&action_id) {
                let match_score = self.pattern_match(ctx, &meta.pattern);
                let confidence = meta.stats.base_confidence;
                (match_score * confidence).powf(temperature) as f32
            } else {
                0.05
            };

            priors.push((action, score));
            sum += score;
        }

        if sum > 0.0 {
            for (_, score) in &mut priors { *score /= sum; }
        } else {
            let uniform = 1.0 / allowed_actions.len() as f32;
            for (_, score) in &mut priors { *score = uniform; }
        }

        priors
    }

    // 🌟 FOR SOULGAIN VM (SKILL SELECTION)
    pub fn select_action(
        &mut self,
        ctx: &ContextSnapshot,
        candidates: &[ActionId],
        tick: u64,
    ) -> Option<ActionId> {
        let total_attempts: u64 = self
            .memory
            .action_meta
            .values()
            .map(|m| m.stats.attempts)
            .sum::<u64>()
            + 1;
            
        let mut scored: Vec<(ActionId, f64)> = Vec::new();
        let mut best_match_score = 0.0;

        for &action_id in candidates {
            self.ensure_action_known(action_id);
            let Some(meta) = self.memory.action_meta.get(&action_id) else { continue; };

            if self.matches_failure_memory(meta, ctx) { continue; }

            let pattern = self.pattern_match(ctx, &meta.pattern);
            self.last_match_score = pattern;
            
            if pattern > best_match_score { best_match_score = pattern; }
            if pattern < self.gate_threshold { continue; }
            
            let score = self.applicability_score(meta, tick, pattern, ctx.domain_tag, total_attempts);
            if score > 0.0 { scored.push((action_id, score)); }
        }

        self.last_match_score = best_match_score;
        if scored.is_empty() { return None; }

        if self.deterministic_mode {
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            return scored.first().map(|(id, _)| *id);
        }

        self.weighted_pick(&scored)
    }

    pub fn update_after_execution(&mut self, action_id: ActionId, outcome: ActionOutcome) {
        self.ensure_action_known(action_id);
        let Some(meta) = self.memory.action_meta.get_mut(&action_id) else { return; };

        meta.stats.attempts += 1;
        
        if outcome.success {
            meta.stats.successes += 1;
            meta.pattern = adapt_pattern_success(&meta.pattern, &outcome.features);
            meta.stats.base_confidence = (meta.stats.base_confidence + 0.05).clamp(0.05, 0.98);
        } else {
            meta.stats.failures += 1;
            meta.pattern.confidence = (meta.pattern.confidence - 0.05).clamp(0.05, 0.98);
            meta.stats.base_confidence = (meta.stats.base_confidence - 0.05).clamp(0.05, 0.98);

            if meta.recent_failures.len() >= 16 { let _ = meta.recent_failures.pop_front(); }
            meta.recent_failures.push_back(FailureSignature {
                context_hash: outcome.context_hash,
                domain_tag: outcome.domain_tag,
            });
        }

        if let Some(tag) = outcome.domain_tag {
            let entry = meta.per_task.entry(tag).or_default();
            entry.attempts += 1;
            if outcome.success { entry.successes += 1; }
        }

        let alpha = 0.25;
        meta.stats.avg_reward_delta =
            (1.0 - alpha) * meta.stats.avg_reward_delta + alpha * outcome.reward_delta;
        meta.stats.last_used_tick = outcome.used_tick;
    }

    fn matches_failure_memory(&self, meta: &ActionMetadata, ctx: &ContextSnapshot) -> bool {
        meta.recent_failures.iter().any(|f| {
            f.context_hash == ctx.context_hash && (f.domain_tag.is_none() || f.domain_tag == ctx.domain_tag)
        })
    }

    fn pattern_match(&self, ctx: &ContextSnapshot, pattern: &ConceptPattern) -> f64 {
        let mut total_active_bits = 0;
        let mut matching_bits = 0;

        for i in 0..16 {
            let mask = pattern.feature_mask[i];
            if mask == 0 { continue; } 

            let expected = pattern.expected_features[i] & mask;
            let actual = ctx.features[i] & mask;
            
            let active_in_word = mask.count_ones();
            total_active_bits += active_in_word;
            matching_bits += active_in_word - (expected ^ actual).count_ones();
        }

        let similarity = if total_active_bits == 0 {
            1.0 
        } else {
            matching_bits as f64 / total_active_bits as f64
        };

        (similarity * pattern.confidence).clamp(0.0, 1.0)
    }

    fn applicability_score(
        &self,
        meta: &ActionMetadata,
        tick: u64,
        pattern_match: f64,
        domain_tag: Option<u64>,
        total_attempts: u64,
    ) -> f64 {
        let success_rate = meta.stats.successes as f64 / std::cmp::max(1, meta.stats.attempts) as f64;
        let normalized_reward = (meta.stats.avg_reward_delta / 100.0).clamp(-1.0, 1.0);
        let age = tick.saturating_sub(meta.stats.last_used_tick);
        let recency_penalty = 1.0 - self.recency_decay(age);
        let task_affinity = self.task_affinity(meta, domain_tag);
        let explore = exploration_bonus(meta, total_attempts);

        self.weights.w_match * pattern_match
            + self.weights.w_success * success_rate
            + self.weights.w_reward * normalized_reward
            + self.weights.w_conf * meta.stats.base_confidence
            - self.weights.w_decay * recency_penalty
            + self.weights.w_explore * explore
            + self.weights.w_task * task_affinity
    }

    fn task_affinity(&self, meta: &ActionMetadata, domain_tag: Option<u64>) -> f64 {
        let Some(tag) = domain_tag else { return 0.0 };
        let Some(task) = meta.per_task.get(&tag) else { return 0.0; };
        task.successes as f64 / std::cmp::max(1, task.attempts) as f64
    }

    fn recency_decay(&self, age_ticks: u64) -> f64 {
        (-(age_ticks as f64) / self.decay_tau_ticks).exp()
    }

    fn weighted_pick(&mut self, scored: &[(ActionId, f64)]) -> Option<ActionId> {
        let total: f64 = scored.iter().map(|(_, s)| *s).sum();
        if total <= 0.0 {
            return scored.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(id, _)| *id);
        }

        let mut r = self.next_unit() * total;
        for (id, score) in scored {
            if r <= *score { return Some(*id); }
            r -= *score;
        }
        scored.last().map(|(id, _)| *id)
    }

    fn next_unit(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = self.rng_state >> 11;
        (v as f64) / ((u64::MAX >> 11) as f64)
    }
}

pub fn exploration_bonus(action: &ActionMetadata, total_attempts: u64) -> f64 {
    let attempts = std::cmp::max(1, action.stats.attempts) as f64;
    let total = std::cmp::max(2, total_attempts) as f64;
    let uncertainty = (1.0 - action.pattern.confidence).clamp(0.0, 1.0);
    (2.0 * total.ln() / attempts).sqrt() * uncertainty
}

fn adapt_pattern_success(pattern: &ConceptPattern, observed_features: &[u64; 16]) -> ConceptPattern {
    let mut next = pattern.clone();
    for i in 0..16 {
        let diff = next.expected_features[i] ^ observed_features[i];
        next.feature_mask[i] &= !diff; 
        next.expected_features[i] &= next.feature_mask[i];
    }
    next.confidence = (next.confidence + 0.05).clamp(0.05, 0.99);
    next
}
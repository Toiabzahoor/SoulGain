use std::collections::{HashMap, VecDeque};

use crate::types::UVal;

pub type SkillId = i64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueKind {
    Nil,
    Bool,
    Number,
    String,
    Object,
}

#[derive(Clone, Debug)]
pub struct ContextSnapshot {
    pub task_tag: Option<u64>,
    pub stack_depth: usize,
    pub top_types: [Option<ValueKind>; 3],
    pub recent_opcodes: Vec<i64>,
    pub data_hash: u64,
    pub feature_hash: u64,
    pub stack_hash: u64,
}

#[derive(Clone, Debug)]
pub struct SkillPattern {
    pub expected_types: [Option<ValueKind>; 3],
    pub expected_data_bits: u64,
    pub expected_data_mask: u64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Default)]
pub struct TaskStats {
    pub attempts: u64,
    pub successes: u64,
}

#[derive(Clone, Debug)]
pub struct SkillStats {
    pub attempts: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_reward_delta: f64,
    pub base_confidence: f64,
    pub last_used_tick: u64,
}

impl Default for SkillStats {
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

#[derive(Clone, Debug)]
pub struct FailureSignature {
    pub stack_hash: u64,
    pub task_tag: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PendingCredit {
    pub skill_id: SkillId,
    pub issued_at_tick: u64,
    pub reward_baseline: f64,
}

#[derive(Clone, Debug)]
pub struct SkillMetadata {
    pub skill_id: SkillId,
    pub pattern: SkillPattern,
    pub stats: SkillStats,
    pub per_task: HashMap<u64, TaskStats>,
    pub recent_failures: VecDeque<FailureSignature>,
}

#[derive(Clone, Debug)]
pub struct SkillOutcome {
    pub success: bool,
    pub reward_delta: f64,
    pub used_tick: u64,
    pub task_tag: Option<u64>,
    pub context_top_types: [Option<ValueKind>; 3],
    pub data_hash: u64,
    pub feature_hash: u64,
    pub stack_hash: u64,
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
            w_match: 0.38,
            w_success: 0.2,
            w_reward: 0.16,
            w_conf: 0.1,
            w_decay: 0.06,
            w_explore: 0.06,
            w_task: 0.04,
        }
    }
}

pub struct IntuitionEngine {
    pub skill_meta: HashMap<SkillId, SkillMetadata>,
    pub weights: IntuitionWeights,
    pub gate_threshold: f64,
    pub deterministic_mode: bool,
    pub decay_tau_ticks: f64,
    pub pending_credits: VecDeque<PendingCredit>,
    pub last_match_score: f64,
    rng_state: u64,
}

impl Default for IntuitionEngine {
    fn default() -> Self {
        Self {
            skill_meta: HashMap::new(),
            weights: IntuitionWeights::default(),
            gate_threshold: 0.2,
            deterministic_mode: false,
            decay_tau_ticks: 12.0,
            pending_credits: VecDeque::new(),
            last_match_score: 0.0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

impl IntuitionEngine {
    pub fn build_context(
        &self,
        stack: &[UVal],
        recent: &VecDeque<i64>,
        task_tag: Option<u64>,
    ) -> ContextSnapshot {
        let mut top_types: [Option<ValueKind>; 3] = [None, None, None];
        for (idx, value) in stack.iter().rev().take(3).enumerate() {
            top_types[idx] = Some(value_kind(value));
        }

        let data_hash = scalar_data_hash(stack);
        ContextSnapshot {
            task_tag,
            stack_depth: stack.len(),
            top_types: top_types.clone(),
            recent_opcodes: recent.iter().copied().collect(),
            data_hash,
            feature_hash: data_hash,
            stack_hash: stack_signature_hash(stack.len(), &top_types, data_hash),
        }
    }

    pub fn ensure_skill_known(&mut self, skill_id: SkillId) {
        self.skill_meta
            .entry(skill_id)
            .or_insert_with(|| SkillMetadata {
                skill_id,
                pattern: SkillPattern {
                    expected_types: [None, None, None],
                    expected_data_bits: 0,
                    expected_data_mask: 0,
                    confidence: 0.5,
                },
                stats: SkillStats::default(),
                per_task: HashMap::new(),
                recent_failures: VecDeque::with_capacity(16),
            });
    }

    pub fn bootstrap_pattern_if_empty(&mut self, skill_id: SkillId, ctx: &ContextSnapshot) {
        self.ensure_skill_known(skill_id);
        if let Some(meta) = self.skill_meta.get_mut(&skill_id) {
            if meta.pattern.expected_types == [None, None, None] {
                meta.pattern.expected_types = ctx.top_types.clone();
                meta.pattern.expected_data_bits = ctx.feature_hash;
                meta.pattern.expected_data_mask = u64::MAX;
            }
        }
    }

    pub fn select_skill(
        &mut self,
        ctx: &ContextSnapshot,
        candidates: &[SkillId],
        tick: u64,
    ) -> Option<SkillId> {
        let total_attempts: u64 = self
            .skill_meta
            .values()
            .map(|m| m.stats.attempts)
            .sum::<u64>()
            + 1;
        let mut scored: Vec<(SkillId, f64)> = Vec::new();
        let mut best_match_score = 0.0;

        for skill_id in candidates {
            self.ensure_skill_known(*skill_id);
            let Some(meta) = self.skill_meta.get(skill_id) else {
                continue;
            };

            if self.matches_failure_memory(meta, ctx) {
                continue;
            }

            let pattern = self.pattern_match(ctx, &meta.pattern);
            self.last_match_score = pattern;
            if pattern > best_match_score {
                best_match_score = pattern;
            }
            if pattern < self.gate_threshold {
                continue;
            }
            let score = self.applicability_score(meta, tick, pattern, ctx.task_tag, total_attempts);
            if score > 0.0 {
                scored.push((*skill_id, score));
            }
        }

        self.last_match_score = best_match_score;

        if scored.is_empty() {
            return None;
        }

        if self.deterministic_mode {
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            return scored.first().map(|(id, _)| *id);
        }

        self.weighted_pick(&scored)
    }

    pub fn latest_match_score(&self) -> f64 {
        self.last_match_score
    }

    pub fn issue_pending_credit(&mut self, skill_id: SkillId, tick: u64, reward_baseline: f64) {
        if self.pending_credits.len() >= 64 {
            let _ = self.pending_credits.pop_front();
        }
        self.pending_credits.push_back(PendingCredit {
            skill_id,
            issued_at_tick: tick,
            reward_baseline,
        });
    }

    pub fn settle_pending_credits(&mut self, current_tick: u64, current_total_reward: f64) {
        let mut retained = VecDeque::with_capacity(self.pending_credits.len());
        while let Some(pending) = self.pending_credits.pop_front() {
            let age = current_tick.saturating_sub(pending.issued_at_tick);
            if age > 64 {
                continue;
            }

            let decay = self.recency_decay(age);
            let delayed_reward = (current_total_reward - pending.reward_baseline) * decay;
            if delayed_reward.abs() > f64::EPSILON {
                self.ensure_skill_known(pending.skill_id);
                if let Some(meta) = self.skill_meta.get_mut(&pending.skill_id) {
                    let alpha = 0.2;
                    meta.stats.avg_reward_delta =
                        (1.0 - alpha) * meta.stats.avg_reward_delta + alpha * delayed_reward;
                }
            }

            if age < 16 {
                retained.push_back(pending);
            }
        }
        self.pending_credits = retained;
    }

    pub fn update_after_execution(&mut self, skill_id: SkillId, outcome: SkillOutcome) {
        self.ensure_skill_known(skill_id);
        let Some(meta) = self.skill_meta.get_mut(&skill_id) else {
            return;
        };

        meta.stats.attempts += 1;
        if outcome.success {
            meta.stats.successes += 1;
            meta.pattern = adapt_pattern_success(
                &meta.pattern,
                &outcome.context_top_types,
                outcome.feature_hash,
            );
            meta.stats.base_confidence = (meta.stats.base_confidence + 0.03).clamp(0.05, 0.98);
        } else {
            meta.stats.failures += 1;
            meta.pattern = adapt_pattern_failure(
                &meta.pattern,
                &outcome.context_top_types,
                outcome.feature_hash,
            );
            meta.stats.base_confidence = (meta.stats.base_confidence - 0.05).clamp(0.05, 0.98);

            if meta.recent_failures.len() >= 16 {
                let _ = meta.recent_failures.pop_front();
            }
            meta.recent_failures.push_back(FailureSignature {
                stack_hash: outcome.stack_hash,
                task_tag: outcome.task_tag,
            });
        }

        if let Some(tag) = outcome.task_tag {
            let entry = meta.per_task.entry(tag).or_default();
            entry.attempts += 1;
            if outcome.success {
                entry.successes += 1;
            }
        }

        let alpha = 0.25;
        meta.stats.avg_reward_delta =
            (1.0 - alpha) * meta.stats.avg_reward_delta + alpha * outcome.reward_delta;

        meta.stats.last_used_tick = outcome.used_tick;
    }

    fn matches_failure_memory(&self, meta: &SkillMetadata, ctx: &ContextSnapshot) -> bool {
        meta.recent_failures.iter().any(|f| {
            f.stack_hash == ctx.stack_hash && (f.task_tag.is_none() || f.task_tag == ctx.task_tag)
        })
    }

    fn pattern_match(&self, ctx: &ContextSnapshot, pattern: &SkillPattern) -> f64 {
        let mut similarity: f64 = 0.0;
        let mut total: f64 = 0.0;
        for i in 0..3 {
            if let Some(expected) = &pattern.expected_types[i] {
                total += 1.0;
                if ctx.top_types[i].as_ref() == Some(expected) {
                    similarity += 1.0;
                }
            }
        }

        let base = if total == 0.0 {
            0.5
        } else {
            similarity / total
        };
        let data_sim = if pattern.expected_data_mask == 0 {
            0.5
        } else {
            let expected = pattern.expected_data_bits & pattern.expected_data_mask;
            let got = ctx.feature_hash & pattern.expected_data_mask;
            if expected == got { 1.0 } else { 0.0 }
        };
        let combined = (0.7 * base) + (0.3 * data_sim);
        (combined * pattern.confidence).clamp(0.0, 1.0)
    }

    fn applicability_score(
        &self,
        meta: &SkillMetadata,
        tick: u64,
        pattern_match: f64,
        task_tag: Option<u64>,
        total_attempts: u64,
    ) -> f64 {
        let success_rate =
            meta.stats.successes as f64 / std::cmp::max(1, meta.stats.attempts) as f64;
        let normalized_reward = (meta.stats.avg_reward_delta / 100.0).clamp(-1.0, 1.0);
        let age = tick.saturating_sub(meta.stats.last_used_tick);
        let recency_penalty = 1.0 - self.recency_decay(age);
        let task_affinity = self.task_affinity(meta, task_tag);
        let explore = exploration_bonus(meta, total_attempts);

        self.weights.w_match * pattern_match
            + self.weights.w_success * success_rate
            + self.weights.w_reward * normalized_reward
            + self.weights.w_conf * meta.stats.base_confidence
            - self.weights.w_decay * recency_penalty
            + self.weights.w_explore * explore
            + self.weights.w_task * task_affinity
    }

    fn task_affinity(&self, meta: &SkillMetadata, task_tag: Option<u64>) -> f64 {
        let Some(tag) = task_tag else { return 0.0 };
        let Some(task) = meta.per_task.get(&tag) else {
            return 0.0;
        };
        task.successes as f64 / std::cmp::max(1, task.attempts) as f64
    }

    fn recency_decay(&self, age_ticks: u64) -> f64 {
        (-(age_ticks as f64) / self.decay_tau_ticks).exp()
    }

    fn weighted_pick(&mut self, scored: &[(SkillId, f64)]) -> Option<SkillId> {
        let total: f64 = scored.iter().map(|(_, s)| *s).sum();
        if total <= 0.0 {
            return scored
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| *id);
        }

        let mut r = self.next_unit() * total;
        for (id, score) in scored {
            if r <= *score {
                return Some(*id);
            }
            r -= *score;
        }
        scored.last().map(|(id, _)| *id)
    }

    fn next_unit(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = self.rng_state >> 11;
        (v as f64) / ((u64::MAX >> 11) as f64)
    }
}

pub fn exploration_bonus(skill: &SkillMetadata, total_attempts: u64) -> f64 {
    let attempts = std::cmp::max(1, skill.stats.attempts) as f64;
    let total = std::cmp::max(2, total_attempts) as f64;
    let uncertainty = (1.0 - skill.pattern.confidence).clamp(0.0, 1.0);
    (2.0 * total.ln() / attempts).sqrt() * uncertainty
}

fn adapt_pattern_success(
    pattern: &SkillPattern,
    observed: &[Option<ValueKind>; 3],
    feature_hash: u64,
) -> SkillPattern {
    let mut next = pattern.clone();
    for (idx, observed_ty) in observed.iter().enumerate() {
        if next.expected_types[idx].is_none() {
            next.expected_types[idx] = observed_ty.clone();
        }
    }
    if next.expected_data_mask == 0 {
        next.expected_data_bits = feature_hash;
        next.expected_data_mask = u64::MAX;
    } else {
        let diff = next.expected_data_bits ^ feature_hash;
        next.expected_data_mask &= !diff;
        next.expected_data_bits &= next.expected_data_mask;
    }
    next.confidence = (next.confidence + 0.03).clamp(0.05, 0.98);
    next
}

fn adapt_pattern_failure(
    pattern: &SkillPattern,
    observed: &[Option<ValueKind>; 3],
    _feature_hash: u64,
) -> SkillPattern {
    let mut next = pattern.clone();
    for (idx, observed_ty) in observed.iter().enumerate() {
        if let Some(expected) = &next.expected_types[idx] {
            if Some(expected) != observed_ty.as_ref() {
                next.expected_types[idx] = None;
            }
        }
    }
    // Keep mask/bits stable on failure; confidence handles down-weighting.
    next.confidence = (next.confidence - 0.04).clamp(0.05, 0.98);
    next
}

fn value_kind(v: &UVal) -> ValueKind {
    match v {
        UVal::Nil => ValueKind::Nil,
        UVal::Bool(_) => ValueKind::Bool,
        UVal::Number(_) => ValueKind::Number,
        UVal::String(_) => ValueKind::String,
        UVal::Object(_) => ValueKind::Object,
    }
}

fn stack_signature_hash(depth: usize, top_types: &[Option<ValueKind>; 3], data_hash: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    h ^= depth as u64;
    h = h.wrapping_mul(0x100000001b3);

    for t in top_types {
        let code = match t {
            None => 0u64,
            Some(ValueKind::Nil) => 1,
            Some(ValueKind::Bool) => 2,
            Some(ValueKind::Number) => 3,
            Some(ValueKind::String) => 4,
            Some(ValueKind::Object) => 5,
        };
        h ^= code;
        h = h.wrapping_mul(0x100000001b3);
    }

    h ^= data_hash;
    h = h.wrapping_mul(0x100000001b3);
    h
}

fn scalar_data_hash(stack: &[UVal]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in stack.iter().rev().take(3) {
        let code = match v {
            UVal::Bool(b) => {
                if *b {
                    0xB001_u64
                } else {
                    0xB000_u64
                }
            }
            UVal::Number(n) => {
                let bits = n.to_bits();
                let is_int = n.fract().abs() < f64::EPSILON;
                let parity_bit = if is_int {
                    (bits as i64 & 1) as u64
                } else {
                    2_u64
                };
                (bits.rotate_left(13)) ^ parity_bit
            }
            UVal::Nil => 0xA11_u64,
            UVal::String(s) => (s.len() as u64).wrapping_mul(1315423911),
            UVal::Object(_) => 0x0BEE_u64,
        };
        h ^= code;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

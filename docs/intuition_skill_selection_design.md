# Intuition Skill Selection Layer (Incremental Design)

## 1) Context representation (what Intuition can see)

Create a **small, explicit context snapshot** at the moment `Op::Intuition` executes.
This snapshot should be derived from existing VM state only (stack, IP, trace), so no VM rewrite is needed.

### Context fields
1. **Stack signature**
   - `depth_bucket`: `0..=5` (cap at 5)
   - `top_types`: type IDs for top N stack items (ex: N=3)
   - `numeric_bands`: for numeric values, bucket into bands (`neg`, `zero`, `small`, `medium`, `large`)
2. **Recent opcode history**
   - fixed ring buffer of last K opcodes (ex: K=6)
   - stored as opcode IDs only (cheap, deterministic)
3. **Input arity hint**
   - inferred from initial stack depth when program starts, or current expected depth profile
4. **Optional task tags**
   - lightweight labels (e.g., `"numeric"`, `"string"`, `"logic"`)
   - can be attached by caller/evolution layer; default empty

### Why this exists
- Gives skill selection enough signal to differentiate when skills apply.
- Keeps behavior interpretable: each feature is human-readable.
- Cheap to compute in Rust (`Vec`, arrays, small enums).

---

## 2) Skill metadata (what each skill tracks)

Each learned skill keeps metadata next to its macro/opcode sequence.

### Metadata per skill
1. **Applicability pattern**
   - accepted stack type patterns (top N)
   - min/max stack depth
   - optional required recent opcode subsequences
   - optional task tag affinity
2. **Usage statistics**
   - `attempts`, `successes`, `failures`
   - `avg_reward_delta` (EMA)
   - `avg_stack_delta_match` (optional)
3. **Recency + exploration state**
   - `last_used_tick`
   - `times_used_recent_window`
4. **Utility state**
   - `base_confidence` (starts neutral, e.g. `0.5`)
   - `learned_utility` (running score updated after outcomes)

### Why this exists
- Applicability pattern provides deterministic gating.
- Stats provide evidence for learning which skills work.
- Recency prevents one skill from dominating forever.

---

## 3) Scoring mechanism (deterministic and interpretable)

Use a transparent additive score with bounded terms:

```text
score(skill, context) =
    w_match * pattern_match(skill.pattern, context)
  + w_success * success_rate(skill)
  + w_reward * normalized_reward(skill)
  + w_conf * skill.base_confidence
  - w_decay * recency_penalty(skill)
  + w_explore * exploration_bonus(skill)
```

### Term definitions
- `pattern_match` in `[0, 1]`
  - exact top-type match: +0.5
  - depth range match: +0.2
  - recent-opcode overlap: +0.2
  - tag match: +0.1
- `success_rate = successes / max(1, attempts)`
- `normalized_reward = clamp(avg_reward_delta / reward_scale, -1, 1)`
- `recency_penalty`
  - higher when used repeatedly in recent window
- `exploration_bonus`
  - small positive bonus when attempts are low

### Why this exists
- All factors are auditable and tunable.
- No neural replacement; still macro-driven skill execution.
- Supports incremental improvements by changing weights only.

---

## 4) Selection algorithm (how Intuition picks)

Use **2-stage selection**:

1. **Gating stage (hard filter)**
   - remove skills with `pattern_match < gate_threshold` (e.g. `0.35`)
2. **Choice stage**
   - default: **weighted random over positive scores**
   - fallback: `argmax` if deterministic mode enabled

### Tradeoffs
- `argmax`
  - + stable/reproducible
  - - can lock into local optimum
- weighted random
  - + preserves exploration with interpretable probabilities
  - - slightly less reproducible unless seeded RNG
- tournament
  - + robust under noisy scores
  - - adds complexity without strong need initially

**Recommendation**: weighted random by default, deterministic seed for repeatability in tests.

---

## 5) Learning/update rules (after skill execution)

After a selected skill finishes (or errors), update metadata.

### Outcome signal
Compute a compact `SkillOutcome`:
- `success`: true if no VM error and postconditions improved
- `reward_delta`: reward events after - before
- `stack_match_after`: whether resulting stack shape moved toward expected pattern
- `used_tick`: VM logical tick

### Update policy
1. `attempts += 1`
2. if success: `successes += 1`, else `failures += 1`
3. `avg_reward_delta = ema(avg_reward_delta, reward_delta, alpha)`
4. `learned_utility += lr * signed_outcome`
   - signed_outcome positive on success, negative on failure
5. adjust `base_confidence` with clamp `[0.05, 0.95]`
6. set `last_used_tick = used_tick`

### Penalizing wrong usage
- Hard penalty when skill causes stack underflow/type error shortly after invocation.
- Increase recency penalty for repeatedly failing skill in short window.

### Strengthening correct usage
- boost confidence only when success occurs under matching context pattern.
- optional: widen applicability pattern slightly when repeated nearby contexts succeed.

---

## 6) Minimal Rust pseudocode (modular + incremental)

```rust
// types/intuition.rs
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug)]
pub enum ValueKind { Nil, Bool, Number, String, Object }

#[derive(Clone, Debug)]
pub enum NumberBand { Neg, Zero, Small, Medium, Large }

#[derive(Clone, Debug)]
pub struct ContextSnapshot {
    pub depth_bucket: u8,
    pub top_types: [Option<ValueKind>; 3],
    pub top_number_bands: [Option<NumberBand>; 3],
    pub recent_opcodes: VecDeque<i64>, // fixed-cap ring behavior
    pub input_arity_hint: u8,
    pub task_tags: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SkillPattern {
    pub min_depth: u8,
    pub max_depth: u8,
    pub required_top_types: [Option<ValueKind>; 3],
    pub preferred_recent_ops: Vec<i64>,
    pub tag_affinity: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct SkillStats {
    pub attempts: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_reward_delta: f64, // EMA
    pub base_confidence: f64,
    pub learned_utility: f64,
    pub last_used_tick: u64,
    pub times_used_recent_window: u32,
}

#[derive(Clone, Debug)]
pub struct SkillMetadata {
    pub skill_id: i64,
    pub pattern: SkillPattern,
    pub stats: SkillStats,
}

#[derive(Clone, Debug)]
pub struct SkillOutcome {
    pub success: bool,
    pub reward_delta: f64,
    pub stack_match_after: bool,
    pub used_tick: u64,
}

pub struct IntuitionEngine {
    pub skill_meta: HashMap<i64, SkillMetadata>,
    pub rng_state: u64, // deterministic simple RNG seed/state
    pub deterministic_mode: bool,
}

impl IntuitionEngine {
    pub fn build_context(&self, vm: &SoulGainVM) -> ContextSnapshot { /* ... */ }

    pub fn applicability_score(&self, ctx: &ContextSnapshot, meta: &SkillMetadata) -> f64 { /* ... */ }

    pub fn select_skill(&mut self, ctx: &ContextSnapshot, candidates: &[i64]) -> Option<i64> {
        // 1) gate by minimum pattern match
        // 2) score remaining candidates
        // 3) pick weighted-random or argmax
    }

    pub fn update_after_execution(&mut self, skill_id: i64, outcome: SkillOutcome) { /* ... */ }
}
```

### VM integration points (minimal changes)
1. Add `intuition: IntuitionEngine` field to `SoulGainVM`.
2. In `Op::Intuition` handler:
   - `ctx = intuition.build_context(self)`
   - `selected = intuition.select_skill(&ctx, learned_skill_ids)`
   - if selected, execute that skill opcode directly.
3. After skill returns / fails, call `update_after_execution`.

No core loop rewrite needed: only `Op::Intuition` branch and skill post-update hook.

---

## 7) Incremental rollout plan

1. **Phase 1: Metadata-only plumbing**
   - introduce `SkillMetadata` + default stats for existing skills.
2. **Phase 2: Deterministic gating + argmax**
   - implement `pattern_match` and pick highest score.
3. **Phase 3: Weighted random + exploration bonus**
   - add seeded RNG for controlled exploration.
4. **Phase 4: Online updates**
   - connect reward/error outcomes to confidence updates.
5. **Phase 5: Pattern adaptation (optional)**
   - cautiously generalize successful patterns.

This path keeps behavior inspectable at every step and avoids architectural jumps.

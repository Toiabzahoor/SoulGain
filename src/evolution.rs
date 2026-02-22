use crate::alphazero::{
    ReasoningConfig, TaskExample, TaskSpec, UniversalPolicy, solve as alphazero_solve,
};
use crate::hypothesis::Hypothesis;
use crate::logic::{OpCategory, ops_in_categories};
use crate::plasticity::Event;
use crate::types::UVal;
use crate::vm::CoreMind;
use crate::{Op, SKILL_OPCODE_BASE, SoulGainVM};
use rand::Rng;
use std::collections::HashSet;
#[allow(unused_imports)]
use std::fs::OpenOptions;
#[allow(unused_imports)]
use std::io::Write;

pub trait Oracle {
    fn evaluate(&self, input: Vec<UVal>) -> Vec<UVal>;
}

#[derive(Debug, Clone)]
pub struct ContextStrategy {
    pub max_stack_depth: usize,
    pub allowed_categories: Vec<OpCategory>,
    pub use_alphazero: bool,
}

impl ContextStrategy {
    pub fn from_examples(examples: &[(Vec<UVal>, Vec<UVal>)]) -> Self {
        let mut max_stack_depth = 1usize;
        let mut has_logic_signal = false;
        let mut has_boolean_output = false;

        for (input, expected) in examples {
            max_stack_depth = max_stack_depth.max(input.len()).max(expected.len());
            let numeric_input = input.iter().all(|v| matches!(v, UVal::Number(_)));
            let bool_output = expected.iter().any(|v| matches!(v, UVal::Bool(_)));
            has_boolean_output |= bool_output;
            if numeric_input && bool_output {
                has_logic_signal = true;
            }
        }

        let mut allowed_categories = vec![OpCategory::Arithmetic, OpCategory::Data];
        if has_logic_signal || has_boolean_output {
            allowed_categories.push(OpCategory::Logic);
            allowed_categories.push(OpCategory::ControlFlow);
        }

        Self {
            max_stack_depth,
            allowed_categories,
            use_alphazero: has_logic_signal || has_boolean_output || max_stack_depth > 2,
        }
    }
}

pub struct Trainer {
    pub vm: SoulGainVM,
    rng: rand::rngs::ThreadRng,
    max_program_len: usize,
    explore_rate: f64,
    program_buf: Vec<f64>,
    pub current_strategy: Option<ContextStrategy>,
}

impl Trainer {
    pub fn new(vm: SoulGainVM, max_program_len: usize) -> Self {
        Self {
            vm,
            rng: rand::thread_rng(),
            max_program_len,
            explore_rate: 0.3,
            program_buf: Vec::new(),
            current_strategy: None,
        }
    }

    fn normalize_depth(depth: usize) -> usize {
        // Treat any stack depth >= 5 as simply "5" (Enough items for complex skills)
        // This prevents overfitting to specific stack sizes.
        std::cmp::min(depth, 5)
    }

    fn detect_problem_shape(&self, input: &[UVal], expected: &[UVal]) -> u8 {
        if input.len() != expected.len() {
            return 3;
        }

        let mut has_str_to_num = false;
        let mut numeric_deltas = Vec::new();

        for (a, b) in input.iter().zip(expected.iter()) {
            match (a, b) {
                (UVal::String(_), UVal::Number(_)) => {
                    has_str_to_num = true;
                }
                (UVal::Number(x), UVal::Number(y)) => {
                    numeric_deltas.push(*y - *x);
                }
                _ => {}
            }
        }

        if has_str_to_num {
            return 5;
        }

        if !numeric_deltas.is_empty() {
            let baseline = numeric_deltas[0];
            let all_same = numeric_deltas
                .iter()
                .all(|d| (*d - baseline).abs() < f64::EPSILON);
            if !all_same {
                return 6;
            }
            if (baseline - 1.0).abs() < f64::EPSILON || (baseline + 1.0).abs() < f64::EPSILON {
                return 1;
            }
            if baseline.abs() < f64::EPSILON {
                return 2;
            }
            return 4;
        }

        0
    }

    pub fn synthesize(
        &mut self,
        examples: &[(Vec<UVal>, Vec<UVal>)],
        attempts_limit: usize,
    ) -> Option<Vec<f64>> {
        if examples.is_empty() {
            return None;
        }

        let strategy = ContextStrategy::from_examples(examples);
        self.current_strategy = Some(strategy.clone());

        if strategy.use_alphazero {
            let task = TaskSpec {
                train_cases: examples
                    .iter()
                    .map(|(input, expected)| TaskExample {
                        input: input.clone(),
                        expected_output: expected.clone(),
                    })
                    .collect(),
            };

            let mut action_space = ops_in_categories(&strategy.allowed_categories);
            if !action_space.contains(&Op::Halt) {
                action_space.push(Op::Halt);
            }
            if !action_space.contains(&Op::Literal) {
                action_space.push(Op::Literal);
            }

            let reasoning = ReasoningConfig {
                simulations: 12_000,
                max_depth: strategy.max_stack_depth.max(6),
                max_program_len: self.max_program_len.max(4),
                max_ops_per_candidate: 128,
                action_space,
                ..ReasoningConfig::default()
            };
            let policy = UniversalPolicy::from_task(&task);
            let root = CoreMind::new();
            if let Some(program) = alphazero_solve(&root, &task, &reasoning, &policy) {
                let mut out: Vec<f64> = program.into_iter().map(|op| op.as_f64()).collect();
                if out.last() != Some(&Op::Halt.as_f64()) {
                    out.push(Op::Halt.as_f64());
                }
                return Some(out);
            }
        }

        let input = examples[0].0.clone();
        let expected = examples[0].1.clone();
        let mut failed_attempts: HashSet<Vec<u64>> = HashSet::new();
        let mut best_program: Option<Vec<f64>> = None;
        let mut best_fitness = 0.0;

        let input_preamble_len = 0;
        let shape_id = self.detect_problem_shape(&input, &expected);

        for current_len in 1..=self.max_program_len {
            failed_attempts.clear();

            for level_attempt in 1..=attempts_limit {
                let r = self.rng.r#gen::<f64>();

                // Strategy Selection Logic
                let has_clue = best_program.is_some() && best_fitness > 0.0001;

                let try_hypothesis = if !has_clue {
                    // No clue? Guess wildly (Hypothesis) or use STDP (Random Build)
                    r < 0.5
                } else {
                    // Have a clue? Only guess completely new things 10% of the time.
                    r < 0.1
                };

                let try_speculation = !try_hypothesis && has_clue && r < 0.4;

                let (current_program, logic_start, strategy) = if try_hypothesis {
                    // --- HYPOTHESIS MODE (Fresh Guess) ---
                    let skills: Vec<i64> = self.vm.skills.macros.keys().cloned().collect();
                    let hypothesis = Hypothesis::generate(current_len, &skills);

                    self.program_buf.clear();
                    let start = self.program_buf.len();
                    self.program_buf.extend_from_slice(&hypothesis.logic);

                    if self.program_buf.last() != Some(&Op::Halt.as_f64()) {
                        self.program_buf.push(Op::Halt.as_f64());
                    }

                    (self.program_buf.clone(), start, "HYPOTHESIS")
                } else if try_speculation {
                    // --- SPECULATION MODE (Optimization) ---
                    let mut variant = best_program.clone().unwrap();
                    let _id = self.speculate_new_skill(&mut variant, input_preamble_len);
                    (variant, input_preamble_len, "SPEC")
                } else if has_clue {
                    // --- MUTATION / EXTEND MODE ---
                    let mut variant = best_program.clone().unwrap();

                    // If the best program is shorter than current_len, EXTEND it.
                    let logic_len = variant.len().saturating_sub(input_preamble_len);
                    if logic_len < current_len {
                        // Remove HALT
                        if variant.last() == Some(&Op::Halt.as_f64()) {
                            variant.pop();
                        }
                        // Add a random op to grow it
                        variant.push(self.choose_random_op_with_bias(input.len()) as f64);
                        variant.push(Op::Halt.as_f64());
                        (variant, input_preamble_len, "EXTEND")
                    } else {
                        // Standard mutation
                        self.mutate_program(&mut variant, input_preamble_len);
                        (variant, input_preamble_len, "MUTATE")
                    }
                } else {
                    // --- RANDOM BUILD (STDP) ---
                    let (_last_event, start) =
                        self.build_program(&input, current_len, true, shape_id);
                    (self.program_buf.clone(), start, "RANDOM")
                };

                let logic_bits: Vec<u64> = current_program[logic_start..]
                    .iter()
                    .map(|f| f.to_bits())
                    .collect();
                if failed_attempts.contains(&logic_bits) {
                    continue;
                }
                failed_attempts.insert(logic_bits);

                let logic = current_program[logic_start..].to_vec();
                let (fitness, solved_all) = self.evaluate_logic_on_examples(&logic, examples);

                self.log_logic(
                    current_len,
                    level_attempt,
                    strategy,
                    &current_program[logic_start..],
                    fitness,
                );

                // Update best even if improvement is tiny
                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_program = Some(current_program.clone());
                    // Give small rewards for ANY progress
                    self.vm
                        .plasticity
                        .observe(Event::Reward((fitness * 100.0) as i16));
                }

                // --- SUCCESS & PRUNING BLOCK ---
                if solved_all {
                    let logic_slice = current_program[logic_start..].to_vec();
                    let mut clean_logic = logic_slice;
                    if clean_logic.last() == Some(&(Op::Halt.as_f64())) {
                        clean_logic.pop();
                    }

                    // [NEW] Pruning Integration
                    use crate::hypothesis::Pruner;
                    let pruned_logic = Pruner::prune(&self.vm, &clean_logic, &input, &expected);

                    if !pruned_logic.is_empty() {
                        let skill_id = self.register_or_find_skill(pruned_logic.clone());

                        // Optional: Log the optimization
                        if clean_logic.len() > pruned_logic.len() {
                            println!(
                                "  [OPTIMIZED] Len {} -> Len {}",
                                clean_logic.len(),
                                pruned_logic.len()
                            );
                        }

                        println!(
                            "  [SUCCESS] Concept: Opcode {} | Len: {}",
                            skill_id, current_len
                        );
                        self.imprint_skill(skill_id, &input);

                        // Construct the optimized return program
                        let mut optimized = current_program[..logic_start].to_vec();
                        optimized.push(skill_id as f64);
                        optimized.push(Op::Halt.as_f64());
                        return Some(optimized);
                    }
                    return Some(current_program);
                }
            }
        }
        None
    }

    fn register_or_find_skill(&mut self, logic: Vec<f64>) -> i64 {
        // [FIX] Prevent Looping/Aliasing:
        // If the new skill logic is just a single instruction, return that instruction's ID.
        if logic.len() == 1 {
            return logic[0] as i64;
        }

        for (id, macro_logic) in &self.vm.skills.macros {
            if *macro_logic == logic {
                return *id;
            }
        }
        let new_id = self.generate_random_id();
        self.vm.skills.define_skill(new_id, logic);
        new_id
    }

    #[allow(dead_code)]
    fn generate_smart_skill_logic(&mut self, target_len: usize) -> i64 {
        let mut logic = Vec::new();
        for _ in 0..target_len {
            let op = self.choose_random_op_with_bias(2);
            logic.push(op as f64);
        }
        self.register_or_find_skill(logic)
    }

    #[allow(unused_variables)]
    fn log_logic(&self, depth: usize, level: usize, strategy: &str, logic: &[f64], fitness: f64) {
        let decoded: Vec<String> = logic
            .iter()
            .map(|&op| {
                if op == Op::Add.as_f64() {
                    "ADD".into()
                } else if op == Op::Sub.as_f64() {
                    "SUB".into()
                } else if op == Op::Mul.as_f64() {
                    "MUL".into()
                } else if op == Op::Halt.as_f64() {
                    "HALT".into()
                } else if op >= SKILL_OPCODE_BASE as f64 {
                    format!("OP_{}", op as i64)
                } else {
                    format!("LIT({})", op)
                }
            })
            .collect();
        // Commenting out logging for performance
        /*
        let mut file = OpenOptions::new().create(true).append(true).open("text.txt").unwrap();
        writeln!(
            file,
            "[{}/{}] [Strategy: {}] Fit: {:.4} | Logic: {:?}",
            depth,
            level,
            strategy,
            fitness,
            decoded
        ).unwrap();
        */
    }

    fn speculate_new_skill(&mut self, program: &mut Vec<f64>, logic_start: usize) -> Option<i64> {
        let logic_len = program.len().saturating_sub(1).saturating_sub(logic_start);
        if logic_len < 2 {
            return None;
        }
        let window_size = self.rng.gen_range(2..=std::cmp::min(5, logic_len));
        let max_start = (program.len() - 1).saturating_sub(window_size);
        if max_start < logic_start {
            return None;
        }
        let start_idx = self.rng.gen_range(logic_start..=max_start);
        let pattern = program[start_idx..start_idx + window_size].to_vec();
        let new_id = self.register_or_find_skill(pattern);
        program.drain(start_idx..start_idx + window_size);
        program.insert(start_idx, new_id as f64);
        Some(new_id)
    }

    fn mutate_program(&mut self, program: &mut Vec<f64>, logic_start: usize) {
        if program.len() <= logic_start + 1 {
            return;
        }
        let mutable_range = logic_start..program.len().saturating_sub(1);
        if mutable_range.is_empty() {
            return;
        }
        // [FIX] Clone the range because gen_range consumes it (Range is not Copy)
        let idx = self.rng.gen_range(mutable_range.clone());
        if self.rng.gen_bool(0.5) && program.len() > logic_start + 2 {
            let swap_idx = self.rng.gen_range(mutable_range.clone());
            program.swap(idx, swap_idx);
        } else {
            let op = self.choose_random_op_with_bias(program.len().saturating_sub(logic_start));
            program[idx] = op as f64;
        }
    }

    fn build_program(
        &mut self,
        input: &[UVal],
        target_len: usize,
        random_bias: bool,
        shape_id: u8,
    ) -> (Event, usize) {
        self.program_buf.clear();
        let mut stack_depth = input.len();
        let logic_start = self.program_buf.len();
        let mut last_event = Event::Opcode {
            opcode: Op::Intuition.as_i64(),
            stack_depth,
        };
        self.vm.plasticity.observe(Event::Context(shape_id as u64));

        // [FIX] Track history to prevent loops
        let mut history: Vec<i64> = Vec::new();

        for _ in 0..target_len {
            let op = if random_bias {
                self.choose_random_op_with_bias(stack_depth)
            } else {
                // [FIX] Pass the history
                self.choose_op_with_stdp(last_event, stack_depth, &history, shape_id)
            };

            self.program_buf.push(op as f64);

            // Update History (Keep last 3)
            history.push(op);
            if history.len() > 3 {
                history.remove(0);
            }

            // Rough stack tracking
            if op == Op::Literal.as_i64() {
                stack_depth += 1;
            } else {
                stack_depth = stack_depth.saturating_sub(1);
            };

            last_event = Event::Opcode {
                opcode: op,
                stack_depth,
            };
        }
        self.program_buf.push(Op::Halt.as_f64());
        (last_event, logic_start)
    }

    fn choose_op_with_stdp(
        &mut self,
        last_event: Event,
        stack_depth: usize,
        history: &[i64],
        shape_id: u8,
    ) -> i64 {
        let mut ops: Vec<i64> = vec![
            Op::Add.as_i64(),
            Op::Sub.as_i64(),
            Op::Mul.as_i64(),
            Op::Div.as_i64(),
            Op::Mod.as_i64(),
            Op::Inc.as_i64(),
            Op::Dec.as_i64(),
            Op::Parse.as_i64(),
            Op::Eq.as_i64(),
            Op::Gt.as_i64(),
            Op::Not.as_i64(),
            Op::And.as_i64(),
            Op::Or.as_i64(),
            Op::Xor.as_i64(),
            Op::IsZero.as_i64(),
            Op::Swap.as_i64(),
            Op::Dup.as_i64(),
            Op::Over.as_i64(),
            Op::Drop.as_i64(),
            Op::Store.as_i64(),
            Op::Load.as_i64(),
            Op::Jmp.as_i64(),
            Op::JmpIf.as_i64(),
            Op::Call.as_i64(),
            Op::Ret.as_i64(),
            Op::Intuition.as_i64(),
            Op::Reward.as_i64(),
            Op::Evolve.as_i64(),
        ];
        for &custom_op in self.vm.skills.macros.keys() {
            ops.push(custom_op);
        }

        if let Ok(mem) = self.vm.plasticity.memory.read() {
            let mut best_op = ops[0];
            let mut best_weight = f64::MIN;

            // [FIX] Normalize Context (Context Overfitting Fix)
            let norm_depth = Self::normalize_depth(stack_depth);

            let norm_last_event = match last_event {
                Event::Opcode {
                    opcode,
                    stack_depth: d,
                } => Event::Opcode {
                    opcode,
                    stack_depth: Self::normalize_depth(d),
                },
                _ => last_event,
            };

            for &op in &ops {
                let target = Event::Opcode {
                    opcode: op,
                    stack_depth: norm_depth,
                };

                // [FIX] Correct Nested Map Access
                let mut weight = 0.0;
                if let Some(targets) = mem.weights.get(&norm_last_event) {
                    if let Some(w) = targets.get(&target) {
                        weight = *w;
                    }
                }

                if let Some(ctx_targets) = mem.weights.get(&Event::Context(shape_id as u64)) {
                    if let Some(ctx_w) = ctx_targets.get(&target) {
                        weight += *ctx_w * 0.35;
                    }
                }

                if shape_id == 5 && op == Op::Parse.as_i64() {
                    weight += 12.0;
                }

                if op >= SKILL_OPCODE_BASE {
                    weight += 1.2;
                }

                // Penalty for looping
                if history.contains(&op) {
                    weight -= 5.0;
                }

                if weight > best_weight {
                    best_weight = weight;
                    best_op = op;
                }
            }

            if best_weight >= 9.0 && self.rng.gen_bool(0.9) {
                return best_op;
            }
        }

        if self.rng.gen_bool(self.explore_rate) {
            return ops[self.rng.gen_range(0..ops.len())];
        }
        ops[0]
    }

    fn choose_random_op_with_bias(&mut self, stack_depth: usize) -> i64 {
        if !self.vm.skills.macros.is_empty() && self.rng.gen_bool(0.3) {
            let keys: Vec<_> = self.vm.skills.macros.keys().cloned().collect();
            if let Some(id) = keys.get(self.rng.gen_range(0..keys.len())) {
                return *id;
            }
        }

        let strategy_ops: Vec<i64> = self
            .current_strategy
            .as_ref()
            .map(|strategy| {
                let mut ops = ops_in_categories(&strategy.allowed_categories);
                if stack_depth == 0 && !ops.contains(&Op::Literal) {
                    ops.push(Op::Literal);
                }
                if !ops.contains(&Op::Halt) {
                    ops.push(Op::Halt);
                }
                ops.into_iter().map(|op| op.as_i64()).collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if !strategy_ops.is_empty() {
            return strategy_ops[self.rng.gen_range(0..strategy_ops.len())];
        }

        let stack_favor = [
            Op::Add.as_i64(),
            Op::Sub.as_i64(),
            Op::Mul.as_i64(),
            Op::Div.as_i64(),
            Op::Mod.as_i64(),
            Op::Inc.as_i64(),
            Op::Dec.as_i64(),
            Op::Eq.as_i64(),
            Op::Gt.as_i64(),
            Op::Not.as_i64(),
            Op::And.as_i64(),
            Op::Or.as_i64(),
            Op::Xor.as_i64(),
            Op::IsZero.as_i64(),
            Op::Swap.as_i64(),
            Op::Dup.as_i64(),
            Op::Over.as_i64(),
            Op::Drop.as_i64(),
            Op::Store.as_i64(),
            Op::Load.as_i64(),
            Op::Jmp.as_i64(),
            Op::JmpIf.as_i64(),
            Op::Call.as_i64(),
            Op::Ret.as_i64(),
            Op::Intuition.as_i64(),
            Op::Reward.as_i64(),
            Op::Evolve.as_i64(),
            Op::Halt.as_i64(),
        ];

        let with_literal = [
            Op::Literal.as_i64(),
            Op::Parse.as_i64(),
            Op::Add.as_i64(),
            Op::Sub.as_i64(),
            Op::Mul.as_i64(),
            Op::Mod.as_i64(),
            Op::Dup.as_i64(),
            Op::Over.as_i64(),
            Op::Drop.as_i64(),
            Op::Halt.as_i64(),
        ];

        let pool = if stack_depth == 0 {
            &with_literal[..]
        } else {
            &stack_favor[..]
        };
        pool[self.rng.gen_range(0..pool.len())]
    }

    fn evaluate_logic_on_examples(
        &mut self,
        logic: &[f64],
        examples: &[(Vec<UVal>, Vec<UVal>)],
    ) -> (f64, bool) {
        let mut total = 0.0;
        let mut solved_all = true;

        for (input, expected) in examples {
            let mut program = self.materialize_program(input, logic);
            self.vm.stack.clear();
            for v in input {
                self.vm.stack.push(v.clone());
            }
            let result = self.execute_program(&mut program);
            let fitness = self.calculate_fitness(&result, expected);
            total += fitness;

            if fitness < 0.9999 {
                solved_all = false;
            }
        }

        (total / examples.len() as f64, solved_all)
    }

    fn materialize_program(&self, _input: &[UVal], logic: &[f64]) -> Vec<f64> {
        let mut program = logic.to_vec();
        if program.last() != Some(&Op::Halt.as_f64()) {
            program.push(Op::Halt.as_f64());
        }
        program
    }

    fn imprint_skill(&self, op_id: i64, sample_input: &[UVal]) {
        if let Ok(mut mem) = self.vm.plasticity.memory.write() {
            // [FIX] NORMALIZE: Save the skill as applicable to any "deep enough" stack
            let norm_depth = Self::normalize_depth(sample_input.len());

            let context = Event::Opcode {
                opcode: Op::Literal.as_i64(),
                stack_depth: norm_depth,
            };
            let target = Event::Opcode {
                opcode: op_id,
                stack_depth: norm_depth,
            };

            mem.weights
                .entry(context)
                .or_insert_with(std::collections::HashMap::new)
                .insert(target, 10.0);
        }
    }

    fn generate_random_id(&mut self) -> i64 {
        loop {
            let id = self.rng.gen_range(1000..9999);
            if !self.vm.skills.macros.contains_key(&id) {
                return id;
            }
        }
    }

    fn calculate_fitness(&self, result: &[UVal], expected: &[UVal]) -> f64 {
        if result.is_empty() || result.len() != expected.len() {
            return 0.0;
        }
        let mut score = 0.0;

        for (got, want) in result.iter().zip(expected.iter()) {
            match (got, want) {
                // Number comparison (unchanged)
                (UVal::Number(a), UVal::Number(b)) => {
                    score += 1.0 / (1.0 + (a - b).abs());
                }
                // [FIX] Strict Boolean matching. Do NOT use is_truthy() for mismatches!
                (UVal::Bool(a), UVal::Bool(b)) => {
                    if a == b {
                        score += 1.0;
                    }
                }
                // [FIX] Any other type mismatch results in 0 score for this item
                _ => {}
            }
        }
        score / expected.len() as f64
    }

    fn execute_program(&mut self, program: &mut Vec<f64>) -> Vec<UVal> {
        self.vm.ip = 0;
        let previous = std::mem::replace(&mut self.vm.program, std::mem::take(program));
        self.vm.run(10_000);
        *program = std::mem::take(&mut self.vm.program);
        self.vm.program = previous;
        self.vm.stack.clone()
    }
}

use std::collections::{HashMap, VecDeque};

use crate::hypothesis::{Hypothesis, Pruner};
use crate::plasticity::{Event, VMError};
use crate::token::Tokenizer;
use crate::types::UVal;
use crate::vm::{Op, SoulGainVM, SKILL_OPCODE_BASE};

pub struct EnglishEngine {
    pub vm: SoulGainVM,
    pub tokenizer: Tokenizer,
    epoch: usize,
    curriculum: CurriculumState,
}

struct CurriculumState {
    level: u8,
    task_performances: HashMap<String, f64>,
    consecutive_successes: usize,
    recent_results: VecDeque<bool>,
}

pub struct TrainReport {
    pub epoch: usize,
    pub accuracy: f64,
    pub synapse_count: usize,
}

#[derive(Clone, Copy, Debug)]
enum TaskTag {
    Arithmetic,
    Logic,
}

impl TaskTag {
    fn as_u64(self) -> u64 {
        match self {
            TaskTag::Arithmetic => 1,
            TaskTag::Logic => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            TaskTag::Arithmetic => "arithmetic",
            TaskTag::Logic => "logic",
        }
    }
}

#[derive(Clone, Debug)]
struct LogicTask {
    tag: TaskTag,
    input: Vec<UVal>,
    expected: Vec<UVal>,
    bytecode: Option<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MathOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

#[derive(Clone, Debug, PartialEq)]
enum MathToken {
    Number(f64),
    Op(MathOp),
    LParen,
    RParen,
}

impl EnglishEngine {
    pub fn new() -> Self {
        let vm = SoulGainVM::new(vec![
            Op::In.as_f64(),
            Op::Intuition.as_f64(),
            Op::Out.as_f64(),
            Op::Jmp.as_f64(),
            0.0,
        ]);

        let tokenizer =
            Tokenizer::load_from_file("vocab.json").unwrap_or_else(|_| Tokenizer::new());

        Self {
            vm,
            tokenizer,
            epoch: 0,
            curriculum: CurriculumState {
                level: 0,
                task_performances: HashMap::new(),
                consecutive_successes: 0,
                recent_results: VecDeque::with_capacity(100),
            },
        }
    }

    fn force_association(&mut self, prev: Event, next: Event, strength: f64) {
        if let Ok(mut mem) = self.vm.plasticity.memory.write() {
            mem.weights
                .entry(prev)
                .or_insert_with(HashMap::new)
                .insert(next, strength);
        }
    }

    fn hamming_distance(a: &[u64; 16], b: &[u64; 16]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }

    pub fn find_closest_memory_key(&self, data: [u64; 16]) -> Option<Event> {
        let pressure = self.vm.stagnation_pressure();
        let threshold = 0.0 - (pressure * 5.0);
        let mem = self.vm.plasticity.memory.read().ok()?;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_event: Option<Event> = None;

        for key in mem.weights.keys() {
            if let Event::ContextWithState { data: key_data, .. } = key {
                let distance = Self::hamming_distance(&data, key_data);
                let word_score = -(distance as f64);
                if word_score > best_score {
                    best_score = word_score;
                    best_event = Some(*key);
                }
            }
        }

        match best_event {
            Some(event) if best_score >= threshold => Some(event),
            _ => None,
        }
    }

    pub fn train_on_text(&mut self, text: &str) -> TrainReport {
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut normal_lines = Vec::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(task) = Self::parse_task_line(trimmed) {
                total += 1;
                let success = self.train_logic_task(&task);
                if success {
                    correct += 1;
                }
                continue;
            }
            normal_lines.push(trimmed.to_string());
        }

        if !normal_lines.is_empty() {
            let combined = normal_lines.join("\n");
            let (token_correct, token_total) = self.train_on_tokens(&combined);
            correct += token_correct;
            total += token_total;
        }

        if total == 0 {
            return TrainReport {
                epoch: self.epoch,
                accuracy: 0.0,
                synapse_count: 0,
            };
        }

        self.epoch += 1;

        let accuracy = (correct as f64 / total as f64) * 100.0;

        let synapse_count = self
            .vm
            .plasticity
            .memory
            .read()
            .map(|mem| mem.weights.values().map(|outgoing| outgoing.len()).sum())
            .unwrap_or(0usize);

        TrainReport {
            epoch: self.epoch,
            accuracy,
            synapse_count,
        }
    }

    fn train_on_tokens(&mut self, text: &str) -> (usize, usize) {
        let tokens = self.tokenizer.encode(text);
        if tokens.len() < 17 {
            return (0, 0);
        }

        let mut correct = 0usize;
        let mut total = 0usize;

        for window in tokens.windows(17) {
            let state_hash = self.vm.calculate_internal_state().hash();
            let prev_event = Event::ContextWithState {
                data: [
                    window[0], window[1], window[2], window[3], window[4], window[5], window[6],
                    window[7], window[8], window[9], window[10], window[11], window[12], window[13],
                    window[14], window[15],
                ],
                state_hash,
            };
            let next_event = Event::ContextWithState {
                data: [
                    window[1], window[2], window[3], window[4], window[5], window[6], window[7],
                    window[8], window[9], window[10], window[11], window[12], window[13], window[14],
                    window[15], window[16],
                ],
                state_hash,
            };

            let predicted = self.vm.plasticity.best_next_event(prev_event);

            match predicted {
                Some((event, confidence)) => {
                    self.vm.record_prediction_confidence(confidence);
                    if event == next_event {
                        self.vm.record_event(Event::Reward(100));
                        correct += 1;
                    } else {
                        self.vm
                            .record_event(Event::Error(VMError::InvalidOpcode(0)));
                        self.force_association(prev_event, next_event, 5.0);
                    }
                }
                None => {
                    self.vm.record_prediction_confidence(0.0);
                    self.vm
                        .record_event(Event::Error(VMError::InvalidOpcode(0)));
                    self.force_association(prev_event, next_event, 5.0);
                }
            }
            total += 1;
        }

        (correct, total)
    }

    pub fn prompt(&mut self, input: &str) -> String {
        if let Some(tag) = Self::detect_task_tag(input) {
            return self.prompt_logic(input, tag);
        }
        self.vm.set_task_tag(None);
        self.prompt_chat(input)
    }

    fn prompt_chat(&mut self, input: &str) -> String {
        let tokens = self.tokenizer.encode(input);
        let mut window: Vec<u64> = tokens;
        if window.len() < 16 {
            let mut padded = vec![0u64; 16 - window.len()];
            padded.append(&mut window);
            window = padded;
        }
        if window.len() > 16 {
            window = window[window.len() - 16..].to_vec();
        }

        let mut generated = Vec::new();
        for _ in 0..64 {
            self.vm.update_thermodynamics();
            let mut data = [0u64; 16];
            for (idx, token) in window.iter().enumerate().take(16) {
                data[idx] = *token;
            }
            let state_hash = self.vm.calculate_internal_state().hash();
            let prev_event = Event::ContextWithState { data, state_hash };
            let exact = Event::ContextWithState { data, state_hash };
            let key = if self
                .vm
                .plasticity
                .memory
                .read()
                .ok()
                .map(|mem| mem.weights.contains_key(&exact))
                .unwrap_or(false)
            {
                self.vm.record_prediction_confidence(1.0);
                self.vm.increment_tick();
                exact
            } else {
                match self.find_closest_memory_key(data) {
                    Some(event) => {
                        self.vm.record_prediction_confidence(0.5);
                        self.vm.increment_tick();
                        event
                    }
                    None => {
                        self.vm.record_prediction_confidence(0.0);
                        self.vm.increment_tick();
                        break;
                    }
                }
            };

            let next = self.vm.plasticity.best_next_event(key);
            match next {
                Some((Event::ContextWithState { data: next_data, .. }, _confidence)) => {
                    let next_id = next_data[15];
                    generated.push(next_id);
                    window.remove(0);
                    window.push(next_id);
                    let learning_rate = 0.1 + (self.vm.stagnation_pressure() * 2.0);
                    let mut next_window = [0u64; 16];
                    for (idx, token) in window.iter().enumerate().take(16) {
                        next_window[idx] = *token;
                    }
                    let next_event = Event::ContextWithState {
                        data: next_window,
                        state_hash,
                    };
                    self.force_association(prev_event, next_event, learning_rate);
                }
                _ => break,
            }
        }

        let output = generated
            .into_iter()
            .map(|token| self.tokenizer.decode(token))
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();

        if output.is_empty() {
            "Ready.".to_string()
        } else {
            output
        }
    }

    fn prompt_logic(&mut self, input: &str, tag: TaskTag) -> String {
        self.vm.set_task_tag(Some(tag.as_u64()));

        let Some(task) = Self::parse_logic_task(input, tag) else {
            return "Ready.".to_string();
        };

        let success = if let Some(bytecode) = task.bytecode.clone() {
            if self.run_math_bytecode(&bytecode, &task.expected) {
                true
            } else {
                let max_len = 6 + (self.curriculum.level as usize * 2);
                self.solve_logic_task(&task, max_len, 40)
            }
        } else {
            let max_len = 6 + (self.curriculum.level as usize * 2);
            self.solve_logic_task(&task, max_len, 40)
        };

        self.update_curriculum(task.tag, success);

        let response = if success {
            self.format_stack_output(&task.expected)
        } else {
            "I need more data.".to_string()
        };
        self.vm.set_task_tag(None);
        response
    }

    fn solve_logic_task(&mut self, task: &LogicTask, max_len: usize, attempts: usize) -> bool {
        self.vm.set_task_tag(Some(task.tag.as_u64()));
        let mut sandbox = self.sandbox_vm();
        sandbox.set_task_tag(Some(task.tag.as_u64()));
        sandbox.stack = task.input.clone();

        if let Some(skill_id) = sandbox.select_skill_for_context() {
            if Self::run_program_matches(&sandbox, &[skill_id as f64], &task.input, &task.expected) {
                self.vm.record_event(Event::Reward(100));
                self.vm.set_task_tag(None);
                return true;
            }
        }

        let Some(logic) =
            Self::synthesize_logic(&sandbox, &task.input, &task.expected, max_len, attempts)
        else {
            self.vm.record_event(Event::Error(VMError::InvalidOpcode(0)));
            self.vm.set_task_tag(None);
            return false;
        };

        let skill_id = self.register_skill(logic);
        self.vm.record_event(Event::Reward(100));
        let updated_sandbox = self.sandbox_vm();
        let success = Self::run_program_matches(
            &updated_sandbox,
            &[skill_id as f64],
            &task.input,
            &task.expected,
        );
        self.vm.set_task_tag(None);
        success
    }

    fn train_logic_task(&mut self, task: &LogicTask) -> bool {
        let success = if let Some(bytecode) = task.bytecode.clone() {
            if self.run_math_bytecode(&bytecode, &task.expected) {
                true
            } else {
                let max_len = 6 + (self.curriculum.level as usize * 2);
                self.solve_logic_task(task, max_len, 60)
            }
        } else {
            let max_len = 6 + (self.curriculum.level as usize * 2);
            self.solve_logic_task(task, max_len, 60)
        };
        self.update_curriculum(task.tag, success);
        success
    }

    fn register_skill(&mut self, logic: Vec<f64>) -> i64 {
        let next_id = self
            .vm
            .skills
            .macros
            .keys()
            .copied()
            .max()
            .map(|id| id + 1)
            .unwrap_or(SKILL_OPCODE_BASE);
        let skill_id = next_id.max(SKILL_OPCODE_BASE);
        self.vm.skills.define_skill(skill_id, logic);
        self.vm.intuition.ensure_skill_known(skill_id);
        skill_id
    }

    fn run_program_matches(
        base_vm: &SoulGainVM,
        program: &[f64],
        input: &[UVal],
        expected: &[UVal],
    ) -> bool {
        let mut test_vm = Self::sandbox_from(base_vm);
        test_vm.stack = input.to_vec();
        test_vm.program = program.to_vec();
        if test_vm.program.last() != Some(&Op::Halt.as_f64()) {
            test_vm.program.push(Op::Halt.as_f64());
        }
        test_vm.run(5000);
        test_vm.stack == expected
    }

    fn run_math_bytecode(&mut self, bytecode: &[f64], expected: &[UVal]) -> bool {
        let mut sandbox = self.sandbox_vm();
        sandbox.program = bytecode.to_vec();
        if sandbox.program.last() != Some(&Op::Out.as_f64()) {
            sandbox.program.push(Op::Out.as_f64());
        }
        let output = sandbox.run_until_output(5000);
        let expected_number = match expected.first() {
            Some(UVal::Number(value)) => *value,
            _ => return false,
        };
        let Some(actual) = output else {
            self.vm.record_event(Event::Error(VMError::InvalidOpcode(0)));
            return false;
        };
        if (actual - expected_number).abs() < 1e-6 {
            self.vm.record_event(Event::Reward(100));
            true
        } else {
            self.vm.record_event(Event::Error(VMError::InvalidOpcode(0)));
            false
        }
    }

    fn synthesize_logic(
        base_vm: &SoulGainVM,
        input: &[UVal],
        expected: &[UVal],
        max_len: usize,
        attempts: usize,
    ) -> Option<Vec<f64>> {
        let skills: Vec<i64> = base_vm.skills.macros.keys().copied().collect();
        for len in 1..=max_len {
            for _ in 0..attempts {
                let hypothesis = Hypothesis::generate(len, &skills);
                if Self::run_program_matches(base_vm, &hypothesis.logic, input, expected) {
                    let pruned = Pruner::prune(base_vm, &hypothesis.logic, input, expected);
                    return Some(pruned);
                }
            }
        }
        None
    }

    fn sandbox_vm(&self) -> SoulGainVM {
        Self::sandbox_from(&self.vm)
    }

    fn sandbox_from(base_vm: &SoulGainVM) -> SoulGainVM {
        let mut vm = SoulGainVM::new(Vec::new());
        vm.skills = base_vm.skills.clone();
        vm.memory = base_vm.memory.clone();
        vm.plasticity = base_vm.plasticity.clone();
        vm
    }

    fn update_curriculum(&mut self, tag: TaskTag, success: bool) {
        if self.curriculum.recent_results.len() == 100 {
            self.curriculum.recent_results.pop_front();
        }
        self.curriculum.recent_results.push_back(success);
        if success {
            self.curriculum.consecutive_successes += 1;
        } else {
            self.curriculum.consecutive_successes = 0;
        }
        let accuracy = self.curriculum.recent_results.iter().filter(|v| **v).count() as f64
            / self.curriculum.recent_results.len().max(1) as f64
            * 100.0;
        self.curriculum
            .task_performances
            .insert(tag.name().to_string(), accuracy);
        self.curriculum.check_graduation();
    }

    fn detect_task_tag(input: &str) -> Option<TaskTag> {
        let lower = input.to_lowercase();
        if lower.contains("calculate")
            || lower.contains("sum")
            || lower.contains("plus")
            || lower.contains("minus")
            || lower.contains("multiply")
            || lower.contains("divide")
            || lower.contains("power")
            || lower.contains('+')
            || lower.contains('-')
            || lower.contains('*')
            || lower.contains('/')
            || lower.contains('%')
            || lower.contains('^')
        {
            return Some(TaskTag::Arithmetic);
        }
        if lower.contains("logic") || lower.contains("puzzle") || lower.contains("solve") {
            return Some(TaskTag::Logic);
        }
        None
    }

    fn parse_task_line(line: &str) -> Option<LogicTask> {
        let trimmed = line.trim();
        if !trimmed.starts_with("[MATH]") {
            return None;
        }
        let content = trimmed.trim_start_matches("[MATH]").trim();
        let (expr, expected_raw) = content.split_once("->")?;
        let expected_value = expected_raw.trim().parse::<f64>().ok()?;
        let tokens = Self::tokenize_math_expression(expr.trim())?;
        let rpn = Self::infix_to_rpn(&tokens)?;
        let bytecode = Self::compile_rpn(&rpn);
        let input_values = Self::numbers_from_tokens(&tokens);
        Some(LogicTask {
            tag: TaskTag::Arithmetic,
            expected: vec![UVal::Number(expected_value)],
            input: input_values,
            bytecode: Some(bytecode),
        })
    }

    fn parse_logic_task(input: &str, tag: TaskTag) -> Option<LogicTask> {
        match tag {
            TaskTag::Arithmetic => {
                let tokens = Self::tokenize_math_expression(input)?;
                let rpn = Self::infix_to_rpn(&tokens)?;
                let bytecode = Self::compile_rpn(&rpn);
                let output = Self::evaluate_math_expression(&bytecode)?;
                let input_values = Self::numbers_from_tokens(&tokens);
                Some(LogicTask {
                    tag,
                    input: input_values,
                    expected: vec![UVal::Number(output)],
                    bytecode: Some(bytecode),
                })
            }
            TaskTag::Logic => None,
        }
    }

    fn evaluate_math_expression(bytecode: &[f64]) -> Option<f64> {
        let mut vm = SoulGainVM::new(Vec::new());
        vm.program = bytecode.to_vec();
        if vm.program.last() != Some(&Op::Out.as_f64()) {
            vm.program.push(Op::Out.as_f64());
        }
        vm.run_until_output(5000)
    }

    fn tokenize_math_expression(input: &str) -> Option<Vec<MathToken>> {
        let mut normalized = input.to_lowercase();
        normalized = normalized.replace("to the power of", "^");
        normalized = normalized.replace("power of", "^");
        normalized = normalized.replace("power", "^");
        normalized = normalized.replace("times", "*");
        normalized = normalized.replace("multiply", "*");
        normalized = normalized.replace("divide", "/");
        normalized = normalized.replace("mod", "%");

        let chars: Vec<char> = normalized.chars().collect();
        let mut tokens = Vec::new();
        let mut idx = 0usize;
        let mut prev_was_op = true;

        while idx < chars.len() {
            let ch = chars[idx];
            if ch.is_whitespace() {
                idx += 1;
                continue;
            }
            if ch == '(' {
                tokens.push(MathToken::LParen);
                prev_was_op = true;
                idx += 1;
                continue;
            }
            if ch == ')' {
                tokens.push(MathToken::RParen);
                prev_was_op = false;
                idx += 1;
                continue;
            }
            if "+-*/%^".contains(ch) {
                if ch == '-' && prev_was_op {
                    let (number, next) = Self::parse_number(&chars, idx)?;
                    tokens.push(MathToken::Number(number));
                    prev_was_op = false;
                    idx = next;
                    continue;
                }
                let op = Self::char_to_math_op(ch)?;
                tokens.push(MathToken::Op(op));
                prev_was_op = true;
                idx += 1;
                continue;
            }
            if ch.is_ascii_digit() || ch == '.' {
                let (number, next) = Self::parse_number(&chars, idx)?;
                tokens.push(MathToken::Number(number));
                prev_was_op = false;
                idx = next;
                continue;
            }
            idx += 1;
        }

        if tokens.is_empty() {
            None
        } else {
            Some(tokens)
        }
    }

    fn numbers_from_tokens(tokens: &[MathToken]) -> Vec<UVal> {
        tokens
            .iter()
            .filter_map(|token| match token {
                MathToken::Number(value) => Some(UVal::Number(*value)),
                _ => None,
            })
            .collect()
    }

    fn parse_number(chars: &[char], start: usize) -> Option<(f64, usize)> {
        let mut idx = start;
        let mut number = String::new();
        if chars[idx] == '-' {
            number.push('-');
            idx += 1;
        }
        while idx < chars.len() && (chars[idx].is_ascii_digit() || chars[idx] == '.') {
            number.push(chars[idx]);
            idx += 1;
        }
        let value = number.parse::<f64>().ok()?;
        Some((value, idx))
    }

    fn char_to_math_op(ch: char) -> Option<MathOp> {
        match ch {
            '+' => Some(MathOp::Add),
            '-' => Some(MathOp::Sub),
            '*' => Some(MathOp::Mul),
            '/' => Some(MathOp::Div),
            '%' => Some(MathOp::Mod),
            '^' => Some(MathOp::Pow),
            _ => None,
        }
    }

    fn infix_to_rpn(tokens: &[MathToken]) -> Option<Vec<MathToken>> {
        let mut output = Vec::new();
        let mut ops: Vec<MathToken> = Vec::new();

        for token in tokens {
            match token {
                MathToken::Number(_) => output.push(token.clone()),
                MathToken::Op(op) => {
                    while let Some(MathToken::Op(top_op)) = ops.last() {
                        if Self::op_precedence(*top_op) > Self::op_precedence(*op)
                            || (Self::op_precedence(*top_op) == Self::op_precedence(*op)
                                && !Self::is_right_associative(*op))
                        {
                            output.push(ops.pop().unwrap());
                        } else {
                            break;
                        }
                    }
                    ops.push(token.clone());
                }
                MathToken::LParen => ops.push(MathToken::LParen),
                MathToken::RParen => {
                    while let Some(top) = ops.pop() {
                        if matches!(top, MathToken::LParen) {
                            break;
                        }
                        output.push(top);
                    }
                }
            }
        }

        while let Some(top) = ops.pop() {
            if matches!(top, MathToken::LParen | MathToken::RParen) {
                return None;
            }
            output.push(top);
        }

        Some(output)
    }

    fn compile_rpn(tokens: &[MathToken]) -> Vec<f64> {
        let mut program = Vec::new();
        for token in tokens {
            match token {
                MathToken::Number(value) => {
                    program.push(Op::Literal.as_f64());
                    program.push(*value);
                }
                MathToken::Op(op) => program.push(Self::math_op_to_opcode(*op).as_f64()),
                _ => {}
            }
        }
        program.push(Op::Out.as_f64());
        program
    }

    fn math_op_to_opcode(op: MathOp) -> Op {
        match op {
            MathOp::Add => Op::Add,
            MathOp::Sub => Op::Sub,
            MathOp::Mul => Op::Mul,
            MathOp::Div => Op::Div,
            MathOp::Mod => Op::Mod,
            MathOp::Pow => Op::Pow,
        }
    }

    fn op_precedence(op: MathOp) -> u8 {
        match op {
            MathOp::Add | MathOp::Sub => 1,
            MathOp::Mul | MathOp::Div | MathOp::Mod => 2,
            MathOp::Pow => 3,
        }
    }

    fn is_right_associative(op: MathOp) -> bool {
        matches!(op, MathOp::Pow)
    }

    fn format_stack_output(&self, stack: &[UVal]) -> String {
        if stack.is_empty() {
            return "Ready.".to_string();
        }
        stack
            .iter()
            .map(|val| val.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn save_vocab(&self) -> std::io::Result<()> {
        self.tokenizer.save_to_file("vocab.json")
    }
}

impl CurriculumState {
    fn check_graduation(&mut self) {
        if self.recent_results.len() < 100 {
            return;
        }
        let accuracy = self.recent_results.iter().filter(|v| **v).count() as f64
            / self.recent_results.len().max(1) as f64
            * 100.0;
        if accuracy > 95.0 {
            self.level = self.level.saturating_add(1);
            self.recent_results.clear();
            self.consecutive_successes = 0;
        }
    }
}

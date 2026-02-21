use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::sync::Arc;

use crate::intuition::{ContextSnapshot, IntuitionEngine, SkillOutcome, ValueKind};
use crate::logic::{decode_ops_for_validation, logic_of, validate_ops};
use crate::memory::MemorySystem;
use crate::plasticity::{Event, Plasticity, VMError};
use crate::types::{InternalState, SkillLibrary, UVal};

pub const SKILL_OPCODE_BASE: i64 = 1000;
const STATE_WINDOW: usize = 32;
const CONTEXT_WINDOW: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i64)]
pub enum Op {
    Literal = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Eq = 5,
    Store = 6,
    Load = 7,
    Halt = 8,
    Gt = 9,
    Not = 10,
    Jmp = 11,
    JmpIf = 12,
    Call = 13,
    Ret = 14,
    Intuition = 15,
    Reward = 16,
    Evolve = 17,
    Swap = 18,
    Dup = 19,
    Over = 20,
    Drop = 21,
    And = 22,
    Or = 23,
    Xor = 24,
    IsZero = 25,
    Mod = 26,
    Inc = 27,
    Dec = 28,
    Parse = 29,
    In = 30,
    Out = 31,
    Pow = 32,
}

pub const CORE_STACK_SIZE: usize = 64;
pub const CORE_MEMORY_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Continue,
    Halt,
    Crash,
}

#[derive(Clone)]
pub struct CoreMind {
    stack: [UVal; CORE_STACK_SIZE],
    stack_len: usize,
    memory: [UVal; CORE_MEMORY_SIZE],
    ip: usize,
    halted: bool,
    crashed: bool,
    reading_literal: bool,
    reading_target: bool, 
}

impl CoreMind {
    pub fn new() -> Self {
        Self {
            stack: std::array::from_fn(|_| UVal::Nil),
            stack_len: 0,
            memory: std::array::from_fn(|_| UVal::Nil),
            ip: 0,
            halted: false,
            crashed: false,
            reading_literal: false,
            reading_target: false,
        }
    }

    pub fn reset(&mut self, input: &[UVal]) {
        self.stack.fill(UVal::Nil);
        self.memory.fill(UVal::Nil);
        self.stack_len = 0;
        self.ip = 0;
        self.halted = false;
        self.crashed = false;
        self.reading_literal = false;
        self.reading_target = false; 
        for value in input.iter().take(CORE_STACK_SIZE) {
            self.stack[self.stack_len] = value.clone();
            self.stack_len += 1;
        }
    }

    // [INTEGRATION POINT] 
    // Calculates a hash of the current "Situation" to query the Plasticity brain.
    pub fn context_hash(&self) -> u64 {
        let mut hash = 1469598103934665603_u64;
        
        // Factor 1: The Instruction Pointer (Location in logic)
        hash ^= self.ip as u64;
        hash = hash.wrapping_mul(1099511628211);

        // Factor 2: The Stack Top (Data Type Context)
        if let Some(top) = self.stack.get(self.stack_len.saturating_sub(1)) {
             // We hash the 'Kind' of value to allow generalization across specific numbers
             let type_id = match top {
                 UVal::Number(_) => 1,
                 UVal::Bool(_) => 2,
                 UVal::String(_) => 3,
                 UVal::Object(_) => 4,
                 UVal::Nil => 0,
             };
             hash ^= type_id;
             hash = hash.wrapping_mul(1099511628211);
        }

        // Factor 3: The Previous Stack Item (Deeper Context)
        if let Some(second) = self.stack.get(self.stack_len.saturating_sub(2)) {
             let type_id = match second {
                 UVal::Number(_) => 1,
                 UVal::Bool(_) => 2,
                 UVal::String(_) => 3,
                 UVal::Object(_) => 4,
                 UVal::Nil => 0,
             };
             hash ^= type_id << 32; // Shift to mix differently
             hash = hash.wrapping_mul(1099511628211);
        }

        hash
    }

    pub fn step(&mut self, op: Op) -> StepStatus {
        if self.reading_literal {
            self.reading_literal = false;
            let pushed = self.push(UVal::Number(op as i64 as f64));
            self.ip = self.ip.saturating_add(1);
            if matches!(pushed, StepStatus::Crash) {
                self.crashed = true;
            }
            return pushed;
        }

        if self.reading_target {
            self.reading_target = false;
            self.ip = op as usize;
            return StepStatus::Continue;
        }

        if self.halted {
            return StepStatus::Halt;
        }
        if self.crashed {
            return StepStatus::Crash;
        }

        let mut next_ip = self.ip.saturating_add(1);
        let status = match op {
            Op::Literal => {
                self.reading_literal = true;
                StepStatus::Continue
            }
            Op::Add => self.binary_numeric(|a, b| a + b),
            Op::Sub => self.binary_numeric(|a, b| a - b),
            Op::Mul => self.binary_numeric(|a, b| a * b),
            Op::Div => {
                self.binary_numeric_checked(|a, b| if b == 0.0 { None } else { Some(a / b) })
            }
            Op::Eq => self.binary_cmp(|a, b| a == b),
            Op::Gt => self.binary_cmp(|a, b| a > b),
            Op::Not => self.not(),
            Op::And => self.binary_truthy(|a, b| a && b),
            Op::Or => self.binary_truthy(|a, b| a || b),
            Op::Xor => self.binary_truthy(|a, b| a ^ b),
            Op::IsZero => self.is_zero(),
            Op::Mod => {
                self.binary_numeric_checked(|a, b| if b == 0.0 { None } else { Some(a % b) })
            }
            Op::Dup => self.dup(),
            Op::Drop => self.drop_top(),
            Op::Swap => self.swap(),
            Op::Over => self.over(),
            Op::Inc => self.unary_numeric(|n| n + 1.0),
            Op::Dec => self.unary_numeric(|n| n - 1.0),
            Op::Store => self.store(),
            Op::Load => self.load(),
            Op::Jmp => {
                self.reading_target = true;
                StepStatus::Continue
            }
            Op::JmpIf => {
                let cond = match self.pop() {
                    Some(val) => val.is_truthy(),
                    None => return StepStatus::Crash,
                };
                if cond {
                    self.reading_target = true;
                } else {
                    next_ip = self.ip.saturating_add(2);
                }
                StepStatus::Continue
            }
            Op::Call => StepStatus::Crash,
            Op::Halt => {
                self.halted = true;
                StepStatus::Halt
            }
            // For other ops like Call/Ret which are not supported in CoreMind minimal VM:
            _ => StepStatus::Crash,
        };

        if !self.reading_literal && !self.reading_target {
            self.ip = next_ip;
        }
        
        if self.reading_literal || self.reading_target {
             self.ip = self.ip.saturating_add(1);
        }

        if matches!(status, StepStatus::Crash) {
            self.crashed = true;
        }
        status
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    pub fn ip(&self) -> usize {
        self.ip
    }

    pub fn state_hash(&self) -> u64 {
        let mut hash = 1469598103934665603_u64;
        hash ^= self.ip as u64;
        hash = hash.wrapping_mul(1099511628211);
        hash ^= self.stack_len as u64;
        hash = hash.wrapping_mul(1099511628211);

        for value in &self.stack[..self.stack_len] {
            mix_uval(&mut hash, value);
        }
        for value in &self.memory {
            mix_uval(&mut hash, value);
        }
        hash ^= self.halted as u64;
        hash = hash.wrapping_mul(1099511628211);
        hash ^= self.crashed as u64;
        hash = hash.wrapping_mul(1099511628211);
        hash ^= self.reading_literal as u64;
        hash = hash.wrapping_mul(1099511628211);
        hash ^= self.reading_target as u64; 
        hash
    }

    pub fn extract_output(&self) -> Vec<UVal> {
        self.stack[..self.stack_len].to_vec()
    }

    fn pop(&mut self) -> Option<UVal> {
        if self.stack_len == 0 {
            return None;
        }
        self.stack_len -= 1;
        Some(std::mem::replace(
            &mut self.stack[self.stack_len],
            UVal::Nil,
        ))
    }

    fn push(&mut self, value: UVal) -> StepStatus {
        if self.stack_len >= CORE_STACK_SIZE {
            return StepStatus::Crash;
        }
        self.stack[self.stack_len] = value;
        self.stack_len += 1;
        StepStatus::Continue
    }

    fn binary_numeric(&mut self, op: impl FnOnce(f64, f64) -> f64) -> StepStatus {
        self.binary_numeric_checked(|a, b| Some(op(a, b)))
    }

    fn binary_numeric_checked(&mut self, op: impl FnOnce(f64, f64) -> Option<f64>) -> StepStatus {
        let rhs = match self.pop() {
            Some(UVal::Number(n)) => n,
            _ => return StepStatus::Crash,
        };
        let lhs = match self.pop() {
            Some(UVal::Number(n)) => n,
            _ => return StepStatus::Crash,
        };
        match op(lhs, rhs) {
            Some(value) => self.push(UVal::Number(value)),
            None => StepStatus::Crash,
        }
    }

    fn unary_numeric(&mut self, op: impl FnOnce(f64) -> f64) -> StepStatus {
        let value = match self.pop() {
            Some(UVal::Number(n)) => n,
            _ => return StepStatus::Crash,
        };
        self.push(UVal::Number(op(value)))
    }

    fn binary_cmp(&mut self, cmp: impl FnOnce(f64, f64) -> bool) -> StepStatus {
        let rhs = match self.pop() {
            Some(UVal::Number(n)) => n,
            _ => return StepStatus::Crash,
        };
        let lhs = match self.pop() {
            Some(UVal::Number(n)) => n,
            _ => return StepStatus::Crash,
        };
        self.push(UVal::Bool(cmp(lhs, rhs)))
    }

    fn not(&mut self) -> StepStatus {
        let value = match self.pop() {
            Some(v) => !v.is_truthy(),
            None => return StepStatus::Crash,
        };
        self.push(UVal::Bool(value))
    }

    fn binary_truthy(&mut self, op: impl FnOnce(bool, bool) -> bool) -> StepStatus {
        let rhs = match self.pop() {
            Some(value) => value.is_truthy(),
            None => return StepStatus::Crash,
        };
        let lhs = match self.pop() {
            Some(value) => value.is_truthy(),
            None => return StepStatus::Crash,
        };
        self.push(UVal::Bool(op(lhs, rhs)))
    }

    fn is_zero(&mut self) -> StepStatus {
        let value = match self.pop() {
            Some(UVal::Number(n)) => n == 0.0,
            _ => return StepStatus::Crash,
        };
        self.push(UVal::Bool(value))
    }

    fn dup(&mut self) -> StepStatus {
        if self.stack_len == 0 {
            return StepStatus::Crash;
        }
        self.push(self.stack[self.stack_len - 1].clone())
    }

    fn drop_top(&mut self) -> StepStatus {
        if self.pop().is_some() {
            StepStatus::Continue
        } else {
            StepStatus::Crash
        }
    }

    fn swap(&mut self) -> StepStatus {
        if self.stack_len < 2 {
            return StepStatus::Crash;
        }
        self.stack.swap(self.stack_len - 1, self.stack_len - 2);
        StepStatus::Continue
    }

    fn over(&mut self) -> StepStatus {
        if self.stack_len < 2 {
            return StepStatus::Crash;
        }
        self.push(self.stack[self.stack_len - 2].clone())
    }

    fn store(&mut self) -> StepStatus {
        let value = match self.pop() {
            Some(value) => value,
            None => return StepStatus::Crash,
        };
        let index = match self.pop() {
            Some(UVal::Number(n)) if n >= 0.0 && n.fract() == 0.0 => n as usize,
            _ => return StepStatus::Crash,
        };
        if index >= CORE_MEMORY_SIZE {
            return StepStatus::Crash;
        }
        self.memory[index] = value;
        StepStatus::Continue
    }

    fn load(&mut self) -> StepStatus {
        let index = match self.pop() {
            Some(UVal::Number(n)) if n >= 0.0 && n.fract() == 0.0 => n as usize,
            _ => return StepStatus::Crash,
        };
        if index >= CORE_MEMORY_SIZE {
            return StepStatus::Crash;
        }
        self.push(self.memory[index].clone())
    }
}

fn mix_uval(hash: &mut u64, value: &UVal) {
    const PRIME: u64 = 1099511628211;
    match value {
        UVal::Nil => {
            *hash ^= 0;
            *hash = hash.wrapping_mul(PRIME);
        }
        UVal::Bool(b) => {
            *hash ^= 1;
            *hash = hash.wrapping_mul(PRIME);
            *hash ^= *b as u64;
            *hash = hash.wrapping_mul(PRIME);
        }
        UVal::Number(n) => {
            *hash ^= 2;
            *hash = hash.wrapping_mul(PRIME);
            *hash ^= n.to_bits();
            *hash = hash.wrapping_mul(PRIME);
        }
        UVal::String(text) => {
            *hash ^= 3;
            *hash = hash.wrapping_mul(PRIME);
            for b in text.as_bytes() {
                *hash ^= *b as u64;
                *hash = hash.wrapping_mul(PRIME);
            }
        }
        UVal::Object(map) => {
            *hash ^= 4;
            *hash = hash.wrapping_mul(PRIME);
            *hash ^= map.len() as u64;
            *hash = hash.wrapping_mul(PRIME);
        }
    }
}

impl Default for CoreMind {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SoulGainAgent {
    pub mind: CoreMind,
    pub skills: SkillLibrary,
    pub plasticity: Plasticity,
}

impl SoulGainAgent {
    pub fn new() -> Self {
        Self {
            mind: CoreMind::new(),
            skills: SkillLibrary::new(),
            plasticity: Plasticity::new(),
        }
    }

    pub fn execute_program(&mut self, program: &[Op], input: &[UVal]) -> Option<Vec<UVal>> {
        self.mind.reset(input);
        while self.mind.ip() < program.len() {
            let op = program[self.mind.ip()];
            match self.mind.step(op) {
                StepStatus::Continue => continue,
                StepStatus::Halt => return Some(self.mind.extract_output()),
                StepStatus::Crash => return None,
            }
        }
        if self.mind.is_halted() {
            Some(self.mind.extract_output())
        } else {
            None
        }
    }
}

impl Default for SoulGainAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Op {
    pub fn from_i64(value: i64) -> Option<Self> {
        match value {
            0 => Some(Op::Literal),
            1 => Some(Op::Add),
            2 => Some(Op::Sub),
            3 => Some(Op::Mul),
            4 => Some(Op::Div),
            5 => Some(Op::Eq),
            6 => Some(Op::Store),
            7 => Some(Op::Load),
            8 => Some(Op::Halt),
            9 => Some(Op::Gt),
            10 => Some(Op::Not),
            11 => Some(Op::Jmp),
            12 => Some(Op::JmpIf),
            13 => Some(Op::Call),
            14 => Some(Op::Ret),
            15 => Some(Op::Intuition),
            16 => Some(Op::Reward),
            17 => Some(Op::Evolve),
            18 => Some(Op::Swap),
            19 => Some(Op::Dup),
            20 => Some(Op::Over),
            21 => Some(Op::Drop),
            22 => Some(Op::And),
            23 => Some(Op::Or),
            24 => Some(Op::Xor),
            25 => Some(Op::IsZero),
            26 => Some(Op::Mod),
            27 => Some(Op::Inc),
            28 => Some(Op::Dec),
            29 => Some(Op::Parse),
            30 => Some(Op::In),
            31 => Some(Op::Out),
            32 => Some(Op::Pow),
            _ => None,
        }
    }

    pub fn as_i64(self) -> i64 {
        self as i64
    }

    pub fn as_f64(self) -> f64 {
        self as i64 as f64
    }
}

pub struct SoulGainVM {
    pub program: Vec<f64>,
    pub stack: Vec<UVal>,
    pub call_stack: Vec<usize>,
    program_stack: Vec<ProgramFrame>,
    pub ip: usize,
    pub memory: MemorySystem,
    pub plasticity: Plasticity,
    pub last_event: Option<Event>,
    pub skills: SkillLibrary,
    pub intuition: IntuitionEngine,
    trace: Vec<Event>,
    recent_opcodes: VecDeque<i64>,
    tick: u64,
    total_reward: f64,
    error_count: u64,
    recent_successes: VecDeque<u8>,
    recent_errors: VecDeque<u8>,
    prediction_confidence_history: VecDeque<f64>,
    recent_stack_depths: VecDeque<usize>,
    recent_context: VecDeque<u64>,
    stagnation_pressure: f64,
    last_state: Option<InternalState>,
    pub state_override: Option<InternalState>,
    pub current_task_tag: Option<u64>,
}

#[derive(Debug)]
struct ProgramFrame {
    program: Vec<f64>,
    ip: usize,
    skill_invocation: Option<SkillInvocation>,
}

#[derive(Debug, Clone)]
struct SkillInvocation {
    skill_id: i64,
    reward_before: f64,
    errors_before: u64,
    task_tag: Option<u64>,
    context_top_types: [Option<ValueKind>; 3],
    data_hash: u64,
    feature_hash: u64,
    stack_hash: u64,
}

impl SoulGainVM {
    pub fn new(program: Vec<f64>) -> Self {
        Self {
            program,
            stack: Vec::with_capacity(256),
            call_stack: Vec::new(),
            program_stack: Vec::new(),
            ip: 0,
            memory: MemorySystem::new(),
            plasticity: Plasticity::new(),
            last_event: None,
            skills: SkillLibrary::new(),
            intuition: IntuitionEngine::default(),
            trace: Vec::with_capacity(512),
            recent_opcodes: VecDeque::with_capacity(8),
            tick: 0,
            total_reward: 0.0,
            error_count: 0,
            recent_successes: VecDeque::with_capacity(STATE_WINDOW),
            recent_errors: VecDeque::with_capacity(STATE_WINDOW),
            prediction_confidence_history: VecDeque::with_capacity(STATE_WINDOW),
            recent_stack_depths: VecDeque::with_capacity(STATE_WINDOW),
            recent_context: VecDeque::with_capacity(CONTEXT_WINDOW),
            stagnation_pressure: 0.0,
            last_state: None,
            state_override: None,
            current_task_tag: None,
        }
    }

    pub fn set_task_tag(&mut self, task_tag: Option<u64>) {
        self.current_task_tag = task_tag;
    }

    #[inline(always)]
    fn decode_opcode(raw: f64) -> Result<i64, VMError> {
        if !raw.is_finite() {
            return Err(VMError::InvalidOpcode(-1));
        }
        let rounded = raw.round();
        if (rounded - raw).abs() > 1e-9 {
            return Err(VMError::InvalidOpcode(rounded as i64));
        }
        Ok(rounded as i64)
    }

    fn push_with_limit<T>(deque: &mut VecDeque<T>, value: T, limit: usize) {
        if deque.len() == limit {
            deque.pop_front();
        }
        deque.push_back(value);
    }

    pub fn record_prediction_confidence(&mut self, confidence: f64) {
        Self::push_with_limit(
            &mut self.prediction_confidence_history,
            confidence.clamp(0.0, 1.0),
            STATE_WINDOW,
        );
    }

    fn record_stack_depth(&mut self) {
        Self::push_with_limit(
            &mut self.recent_stack_depths,
            self.stack.len(),
            STATE_WINDOW,
        );
    }

    fn record_context_token(&mut self, token: u64) {
        Self::push_with_limit(&mut self.recent_context, token, CONTEXT_WINDOW);
        if self.recent_context.len() == CONTEXT_WINDOW {
            let mut data = [0u64; CONTEXT_WINDOW];
            for (idx, value) in self.recent_context.iter().enumerate() {
                data[idx] = *value;
            }
            let state_hash = self.calculate_internal_state().hash();
            self.record_event(Event::ContextWithState { data, state_hash });
        }
    }

    pub fn calculate_internal_state(&self) -> InternalState {
        if let Some(forced) = self.state_override {
            return forced;
        }
        let successes = self.recent_successes.len() as f64;
        let errors = self.recent_errors.len() as f64;
        let total = (successes + errors).max(1.0);
        let error_rate = errors / total;
        let success_rate = successes / total;

        let mut changes = 0usize;
        let mut total_depth = 0usize;
        let mut prev_depth: Option<usize> = None;
        for depth in &self.recent_stack_depths {
            if let Some(prev) = prev_depth {
                total_depth += 1;
                if *depth != prev {
                    changes += 1;
                }
            }
            prev_depth = Some(*depth);
        }
        let stack_activity = if total_depth > 0 {
            changes as f64 / total_depth as f64
        } else {
            0.0
        };

        let avg_confidence = if self.prediction_confidence_history.is_empty() {
            0.5
        } else {
            self.prediction_confidence_history.iter().sum::<f64>()
                / self.prediction_confidence_history.len() as f64
        };

        InternalState {
            entropy: error_rate.clamp(0.0, 1.0),
            momentum: (success_rate * stack_activity).clamp(0.0, 1.0),
            coherence: self.intuition.latest_match_score().clamp(0.0, 1.0),
            plasticity: (1.0 - avg_confidence).clamp(0.0, 1.0),
        }
    }

    pub fn update_thermodynamics(&mut self) -> f64 {
        let current = self.calculate_internal_state();
        let threshold = 0.01;

        if let Some(previous) = self.last_state {
            let total_change = (current.entropy - previous.entropy).abs()
                + (current.momentum - previous.momentum).abs()
                + (current.coherence - previous.coherence).abs()
                + (current.plasticity - previous.plasticity).abs();
            if total_change > threshold {
                self.stagnation_pressure = 0.0;
            } else {
                self.stagnation_pressure += 1.0;
            }
        }

        self.last_state = Some(current);
        self.stagnation_pressure
    }

    pub fn stagnation_pressure(&self) -> f64 {
        self.stagnation_pressure
    }

    pub fn increment_tick(&mut self) {
        self.tick = self.tick.saturating_add(1);
    }

    pub fn build_context_snapshot(&self) -> ContextSnapshot {
        self.intuition
            .build_context(&self.stack, &self.recent_opcodes, self.current_task_tag)
    }

    pub fn select_skill_for_context(&mut self) -> Option<i64> {
        let candidates: Vec<i64> = self.skills.macros.keys().copied().collect();
        let ctx = self.build_context_snapshot();
        self.intuition.select_skill(&ctx, &candidates, self.tick)
    }

    pub fn invoke_skill(&mut self, skill_id: i64) {
        self.execute_skill(skill_id);
    }

    pub fn record_event(&mut self, event: Event) {
        self.last_event = Some(event);
        self.trace.push(event);
        match event {
            Event::Reward(_) | Event::Opcode { .. } => {
                Self::push_with_limit(&mut self.recent_successes, 1, STATE_WINDOW);
            }
            Event::Error(_) => {
                Self::push_with_limit(&mut self.recent_errors, 1, STATE_WINDOW);
            }
            _ => {}
        }
        self.record_stack_depth();
    }

    fn record_error(&mut self, error: VMError) {
        self.error_count = self.error_count.saturating_add(1);
        self.record_event(Event::Error(error));
        self.flush_trace();
    }

    fn flush_trace(&mut self) {
        if self.trace.is_empty() {
            return;
        }
        let batch = std::mem::take(&mut self.trace);
        self.plasticity.observe_batch(batch);
    }

    fn restore_program(&mut self) -> bool {
        if let Some(frame) = self.program_stack.pop() {
            if let Some(invocation) = frame.skill_invocation {
                let success = self.error_count == invocation.errors_before;
                let reward_delta = self.total_reward - invocation.reward_before;
                self.intuition
                    .settle_pending_credits(self.tick, self.total_reward);
                self.intuition.update_after_execution(
                    invocation.skill_id,
                    SkillOutcome {
                        success,
                        reward_delta,
                        used_tick: self.tick,
                        task_tag: invocation.task_tag,
                        context_top_types: invocation.context_top_types,
                        data_hash: invocation.data_hash,
                        feature_hash: invocation.feature_hash,
                        stack_hash: invocation.stack_hash,
                    },
                );
            }
            self.program = frame.program;
            self.ip = frame.ip;
            true
        } else {
            false
        }
    }

    pub fn run(&mut self, max_cycles: usize) {
        let mut cycles = 0usize;
        while cycles < max_cycles {
            if self.ip >= self.program.len() {
                if self.restore_program() {
                    continue;
                }
                self.flush_trace();
                break;
            }
            let raw = unsafe { *self.program.get_unchecked(self.ip) };
            self.ip += 1;
            cycles += 1;
            self.tick = self.tick.saturating_add(1);

            let opcode = match Self::decode_opcode(raw) {
                Ok(op) => op,
                Err(e) => {
                    self.record_error(e);
                    continue;
                }
            };

            if opcode >= SKILL_OPCODE_BASE {
                let opcode_event = Event::Opcode {
                    opcode,
                    stack_depth: self.stack.len(),
                };
                self.record_event(opcode_event);
                self.push_recent_opcode(opcode);
                self.execute_skill(opcode);
                continue;
            }

            match Op::from_i64(opcode) {
                Some(op) => {
                    if !self.execute_opcode(op) {
                        break;
                    }
                }
                None => self.record_error(VMError::InvalidOpcode(opcode)),
            }
        }
    }

    pub fn run_until_output(&mut self, max_cycles: usize) -> Option<f64> {
        let mut cycles = 0usize;
        while cycles < max_cycles {
            if self.ip >= self.program.len() {
                if self.restore_program() {
                    continue;
                }
                self.flush_trace();
                return None;
            }
            let raw = unsafe { *self.program.get_unchecked(self.ip) };
            self.ip += 1;
            cycles += 1;
            self.tick = self.tick.saturating_add(1);

            let opcode = match Self::decode_opcode(raw) {
                Ok(op) => op,
                Err(e) => {
                    self.record_error(e);
                    continue;
                }
            };

            if opcode >= SKILL_OPCODE_BASE {
                let opcode_event = Event::Opcode {
                    opcode,
                    stack_depth: self.stack.len(),
                };
                self.record_event(opcode_event);
                self.push_recent_opcode(opcode);
                self.execute_skill(opcode);
                continue;
            }

            match Op::from_i64(opcode) {
                Some(op) => match op {
                    Op::Out => {
                        let opcode_event = Event::Opcode {
                            opcode: op.as_i64(),
                            stack_depth: self.stack.len(),
                        };
                        self.record_event(opcode_event);
                        self.push_recent_opcode(op.as_i64());

                        let val = match self.stack.pop() {
                            Some(value) => value,
                            None => {
                                self.record_error(VMError::StackUnderflow);
                                return None;
                            }
                        };

                        let number = match val {
                            UVal::Number(n) => n,
                            UVal::Bool(b) => {
                                if b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            UVal::Nil => f64::NAN,
                            UVal::String(s) => {
                                s.as_bytes().first().map(|b| *b as f64).unwrap_or(f64::NAN)
                            }
                            UVal::Object(_) => f64::NAN,
                        };
                        if number >= 32.0 && number <= 126.0 && number.fract() == 0.0 {
                            self.record_context_token(number as u64);
                        }
                        return Some(number);
                    }
                    Op::In => {
                        let opcode_event = Event::Opcode {
                            opcode: op.as_i64(),
                            stack_depth: self.stack.len(),
                        };
                        self.record_event(opcode_event);
                        self.push_recent_opcode(op.as_i64());
                        return None;
                    }
                    _ => {
                        if !self.execute_opcode(op) {
                            return None;
                        }
                    }
                },
                None => self.record_error(VMError::InvalidOpcode(opcode)),
            }
        }
        None
    }

    fn execute_skill(&mut self, opcode: i64) {
        if let Some(macro_code) = self.skills.get_skill(opcode).cloned() {
            let ctx = self.intuition.build_context(
                &self.stack,
                &self.recent_opcodes,
                self.current_task_tag,
            );
            self.intuition.bootstrap_pattern_if_empty(opcode, &ctx);
            self.intuition
                .issue_pending_credit(opcode, self.tick, self.total_reward);

            let frame = ProgramFrame {
                program: std::mem::take(&mut self.program),
                ip: self.ip,
                skill_invocation: Some(SkillInvocation {
                    skill_id: opcode,
                    reward_before: self.total_reward,
                    errors_before: self.error_count,
                    task_tag: ctx.task_tag,
                    context_top_types: ctx.top_types,
                    data_hash: ctx.data_hash,
                    feature_hash: ctx.feature_hash,
                    stack_hash: ctx.stack_hash,
                }),
            };
            self.program_stack.push(frame);
            self.program = macro_code;
            self.ip = 0;
        } else {
            self.record_error(VMError::InvalidOpcode(opcode));
        }
    }

    fn push_recent_opcode(&mut self, opcode: i64) {
        if self.recent_opcodes.len() >= 6 {
            let _ = self.recent_opcodes.pop_front();
        }
        self.recent_opcodes.push_back(opcode);
    }

    #[inline(always)]
    fn execute_opcode(&mut self, opcode: Op) -> bool {
        let info = logic_of(opcode);
        if info.stack_delta < 0 && self.stack.len() < (-info.stack_delta) as usize {
            self.record_error(VMError::StackUnderflow);
            return true;
        }

        let opcode_event = Event::Opcode {
            opcode: opcode.as_i64(),
            stack_depth: self.stack.len(),
        };
        self.record_event(opcode_event);
        self.push_recent_opcode(opcode.as_i64());

        match opcode {
            Op::Literal => {
                if self.ip >= self.program.len() {
                    return false;
                }
                let v = unsafe { *self.program.get_unchecked(self.ip) };
                self.ip += 1;
                self.stack.push(UVal::Number(v));
            }
            Op::Add => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                match (a, b) {
                    (UVal::Number(na), UVal::Number(nb)) => self.stack.push(UVal::Number(na + nb)),
                    (UVal::String(sa), UVal::String(sb)) => {
                        let mut new_s = (*sa).clone();
                        new_s.push_str(&sb);
                        self.stack.push(UVal::String(Arc::new(new_s)));
                    }
                    _ => self.record_error(VMError::InvalidOpcode(opcode.as_i64())),
                }
            }
            Op::Sub => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                if let (UVal::Number(na), UVal::Number(nb)) = (a, b) {
                    self.stack.push(UVal::Number(na - nb));
                }
            }
            Op::Mul => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                if let (UVal::Number(na), UVal::Number(nb)) = (a, b) {
                    self.stack.push(UVal::Number(na * nb));
                }
            }
            Op::Div => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                if let (UVal::Number(na), UVal::Number(nb)) = (a, b) {
                    if nb == 0.0 {
                        self.record_error(VMError::InvalidOpcode(opcode.as_i64()));
                    } else {
                        self.stack.push(UVal::Number(na / nb));
                    }
                } else {
                    self.record_error(VMError::InvalidOpcode(opcode.as_i64()));
                }
            }
            Op::Eq => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                self.stack.push(UVal::Bool(a == b));
            }
            Op::Gt => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                if let (UVal::Number(na), UVal::Number(nb)) = (a, b) {
                    self.stack.push(UVal::Bool(na > nb));
                }
            }
            Op::Not => {
                if let Some(val) = self.stack.pop() {
                    self.stack.push(UVal::Bool(!val.is_truthy()));
                } else {
                    self.record_error(VMError::StackUnderflow);
                }
            }
            Op::Store => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let val = self.stack.pop().unwrap();
                let addr_val = self.stack.pop().unwrap();
                if let UVal::Number(addr) = addr_val {
                    if self.memory.write(addr, val) {
                        self.record_event(Event::MemoryWrite);
                    }
                } else {
                    self.record_error(VMError::InvalidOpcode(opcode.as_i64()));
                }
            }
            Op::Load => {
                if let Some(UVal::Number(addr)) = self.stack.pop() {
                    if let Some(v) = self.memory.read(addr) {
                        self.stack.push(v);
                        self.record_event(Event::MemoryRead);
                    } else {
                        self.stack.push(UVal::Nil);
                    }
                } else {
                    self.record_error(VMError::StackUnderflow);
                }
            }
            Op::Intuition => {
                let candidates: Vec<i64> = self.skills.macros.keys().copied().collect();
                let ctx = self.intuition.build_context(
                    &self.stack,
                    &self.recent_opcodes,
                    self.current_task_tag,
                );
                if let Some(skill_id) = self.intuition.select_skill(&ctx, &candidates, self.tick) {
                    self.execute_skill(skill_id);
                }
            }
            Op::Jmp => {
                if self.ip >= self.program.len() {
                    return false;
                }
                let target = self.program[self.ip];
                self.ip += 1;
                if !target.is_finite() || target < 0.0 {
                    self.record_error(VMError::InvalidJump(-1));
                    return true;
                }
                let new_ip = target.round() as usize;
                if new_ip >= self.program.len() {
                    self.record_error(VMError::InvalidJump(new_ip as i64));
                    return true;
                }
                self.ip = new_ip;
            }
            Op::JmpIf => {
                if self.ip >= self.program.len() {
                    self.record_error(VMError::InvalidJump(-1));
                    return false;
                }
                if self.stack.is_empty() {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let target = self.program[self.ip];
                self.ip += 1;
                let condition = self.stack.pop().unwrap();
                if condition.is_truthy() {
                    if !target.is_finite() || target < 0.0 {
                        self.record_error(VMError::InvalidJump(-1));
                        return true;
                    }
                    let new_ip = target.round() as usize;
                    if new_ip >= self.program.len() {
                        self.record_error(VMError::InvalidJump(new_ip as i64));
                        return true;
                    }
                    self.ip = new_ip;
                }
            }
            Op::Call => {
                if self.ip >= self.program.len() {
                    return false;
                }
                let target = self.program[self.ip];
                self.ip += 1;
                if !target.is_finite() || target < 0.0 {
                    self.record_error(VMError::InvalidJump(-1));
                    return true;
                }
                let new_ip = target.round() as usize;
                if new_ip >= self.program.len() {
                    self.record_error(VMError::InvalidJump(new_ip as i64));
                    return true;
                }
                self.call_stack.push(self.ip);
                self.ip = new_ip;
            }
            Op::Ret => {
                if let Some(return_ip) = self.call_stack.pop() {
                    self.ip = return_ip;
                } else {
                    self.record_error(VMError::ReturnStackUnderflow);
                }
            }
            Op::Reward => {
                self.total_reward += 100.0;
                self.record_event(Event::Reward(100));
                self.flush_trace();
            }
            Op::Evolve => {
                if let Some(UVal::Number(id)) = self.stack.pop() {
                    let skill_program = self.program.clone();
                    match decode_ops_for_validation(&skill_program).and_then(|ops| {
                        validate_ops(&ops).map_err(|_| VMError::InvalidEvolve(id as i64))
                    }) {
                        Ok(_) => {
                            self.skills.define_skill(id as i64, skill_program);
                            self.record_event(Event::Reward(100));
                            self.flush_trace();
                        }
                        Err(err) => self.record_error(err),
                    }
                } else {
                    self.record_error(VMError::InvalidEvolve(-1));
                }
            }
            Op::Halt => {
                self.flush_trace();
                if self.restore_program() {
                    return true;
                }
                return false;
            }
            Op::Swap => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let len = self.stack.len();
                self.stack.swap(len - 1, len - 2);
            }
            Op::Dup => {
                if let Some(val) = self.stack.last().cloned() {
                    self.stack.push(val);
                } else {
                    self.record_error(VMError::StackUnderflow);
                }
            }
            Op::Over => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let len = self.stack.len();
                let val = self.stack[len - 2].clone();
                self.stack.push(val);
            }
            Op::Drop => {
                if self.stack.pop().is_none() {
                    self.record_error(VMError::StackUnderflow);
                }
            }
            Op::And => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                self.stack.push(UVal::Bool(a.is_truthy() && b.is_truthy()));
            }
            Op::Or => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                self.stack.push(UVal::Bool(a.is_truthy() || b.is_truthy()));
            }
            Op::Xor => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                let result = a.is_truthy() ^ b.is_truthy();
                self.stack.push(UVal::Bool(result));
            }
            Op::IsZero => {
                if let Some(val) = self.stack.pop() {
                    self.stack.push(UVal::Bool(!val.is_truthy()));
                } else {
                    self.record_error(VMError::StackUnderflow);
                }
            }
            Op::Mod => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                if let (UVal::Number(na), UVal::Number(nb)) = (a, b) {
                    self.stack.push(UVal::Number(na % nb));
                } else {
                    self.record_error(VMError::InvalidOpcode(opcode.as_i64()));
                }
            }
            Op::Pow => {
                if self.stack.len() < 2 {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                }
                let b = self.stack.pop().unwrap();
                let a = self.stack.pop().unwrap();
                if let (UVal::Number(base), UVal::Number(exp)) = (a, b) {
                    self.stack.push(UVal::Number(base.powf(exp)));
                } else {
                    self.record_error(VMError::InvalidOpcode(opcode.as_i64()));
                }
            }
            Op::Inc => match self.stack.pop() {
                Some(UVal::Number(n)) => self.stack.push(UVal::Number(n + 1.0)),
                Some(_) => self.record_error(VMError::InvalidOpcode(opcode.as_i64())),
                None => self.record_error(VMError::StackUnderflow),
            },
            Op::Dec => match self.stack.pop() {
                Some(UVal::Number(n)) => self.stack.push(UVal::Number(n - 1.0)),
                Some(_) => self.record_error(VMError::InvalidOpcode(opcode.as_i64())),
                None => self.record_error(VMError::StackUnderflow),
            },
            Op::Parse => match self.stack.pop() {
                Some(UVal::String(text)) => match text.parse::<f64>() {
                    Ok(n) => self.stack.push(UVal::Number(n)),
                    Err(_) => self.stack.push(UVal::Nil),
                },
                Some(UVal::Number(n)) => self.stack.push(UVal::Number(n)),
                Some(_) => self.stack.push(UVal::Nil),
                None => self.record_error(VMError::StackUnderflow),
            },
            Op::In => {
                let _ = io::stdout().flush();
                let mut buf = [0u8; 1];
                match io::stdin().read_exact(&mut buf) {
                    Ok(()) => {
                        self.stack.push(UVal::Number(buf[0] as f64));
                        self.record_context_token(buf[0] as u64);
                    }
                    Err(_) => self.stack.push(UVal::Nil),
                }
            }
            Op::Out => {
                let Some(val) = self.stack.pop() else {
                    self.record_error(VMError::StackUnderflow);
                    return true;
                };
                match val {
                    UVal::Number(n) => {
                        if n >= 32.0 && n <= 126.0 && n.fract() == 0.0 {
                            self.record_context_token(n as u64);
                            print!("{}", n as u8 as char);
                        } else if n == 10.0 {
                            print!("{}", n as u8 as char);
                        } else {
                            print!("[{}]", n);
                        }
                    }
                    UVal::String(s) => {
                        print!("{}", s);
                    }
                    _ => {
                        print!("{:?}", val);
                    }
                }
                let _ = io::stdout().flush();
            }
        }

        true
    }

    fn find_next_opcode(&self, target_opcode: i64) -> Option<usize> {
        self.program.iter().enumerate().find_map(|(idx, &raw)| {
            if raw == target_opcode as f64 {
                Some(idx)
            } else {
                None
            }
        })
    }
}
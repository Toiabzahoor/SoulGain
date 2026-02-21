use crate::plasticity::VMError;
use crate::vm::{Op, SKILL_OPCODE_BASE};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicInfo {
    pub stack_delta: isize,
    pub may_branch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicValidationError {
    MissingHalt,
    StackUnderflow {
        index: usize,
        op: Op,
        needed: usize,
        available: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCategory {
    Arithmetic,
    Logic,
    Data,
    ControlFlow,
    Memory,
    Meta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceLogicSummary {
    pub net_stack_delta: isize,
    pub has_branch: bool,
}

pub fn logic_of(op: Op) -> LogicInfo {
    match op {
        Op::Literal => LogicInfo {
            stack_delta: 1,
            may_branch: false,
        },
        Op::Add => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Sub => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Mul => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Div => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Eq => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Store => LogicInfo {
            stack_delta: -2,
            may_branch: false,
        },
        Op::Load => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::Halt => LogicInfo {
            stack_delta: 0,
            may_branch: true,
        },
        Op::Gt => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Not => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::Jmp => LogicInfo {
            stack_delta: 0,
            may_branch: true,
        },
        Op::JmpIf => LogicInfo {
            stack_delta: -1,
            may_branch: true,
        },
        Op::Call => LogicInfo {
            stack_delta: 0,
            may_branch: true,
        },
        Op::Ret => LogicInfo {
            stack_delta: 0,
            may_branch: true,
        },
        Op::Intuition => LogicInfo {
            stack_delta: 0,
            may_branch: true,
        },
        Op::Reward => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::Evolve => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Swap => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::Dup => LogicInfo {
            stack_delta: 1,
            may_branch: false,
        },
        Op::Over => LogicInfo {
            stack_delta: 1,
            may_branch: false,
        },
        Op::Drop => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::And => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Or => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Xor => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::IsZero => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::Mod => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Pow => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
        Op::Inc => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::Dec => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::Parse => LogicInfo {
            stack_delta: 0,
            may_branch: false,
        },
        Op::In => LogicInfo {
            stack_delta: 1,
            may_branch: false,
        },
        Op::Out => LogicInfo {
            stack_delta: -1,
            may_branch: false,
        },
    }
}

fn min_stack_required(op: Op) -> usize {
    match op {
        Op::Literal | Op::Halt | Op::Jmp | Op::Call | Op::Intuition | Op::Reward | Op::In => 0,
        Op::Load
        | Op::Not
        | Op::JmpIf
        | Op::Evolve
        | Op::Dup
        | Op::Drop
        | Op::IsZero
        | Op::Inc
        | Op::Dec
        | Op::Parse
        | Op::Out => 1,
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Eq
        | Op::Store
        | Op::Gt
        | Op::Swap
        | Op::Over
        | Op::And
        | Op::Or
        | Op::Xor
        | Op::Mod
        | Op::Pow => 2,
        Op::Ret => 0,
    }
}

pub fn validate_ops(program: &[Op]) -> Result<(), LogicValidationError> {
    let mut depth: isize = 0;
    let mut has_halt = false;

    for (index, &op) in program.iter().enumerate() {
        let needed = min_stack_required(op);
        if depth < needed as isize {
            return Err(LogicValidationError::StackUnderflow {
                index,
                op,
                needed,
                available: depth.max(0) as usize,
            });
        }

        let info = logic_of(op);
        depth += info.stack_delta;
        if depth < 0 {
            return Err(LogicValidationError::StackUnderflow {
                index,
                op,
                needed: (-info.stack_delta) as usize,
                available: 0,
            });
        }

        if op == Op::Halt {
            has_halt = true;
        }
    }

    if !has_halt {
        return Err(LogicValidationError::MissingHalt);
    }

    Ok(())
}

pub fn category_of(op: Op) -> OpCategory {
    match op {
        Op::Literal | Op::Dup | Op::Over | Op::Drop | Op::Swap | Op::Parse | Op::In | Op::Out => {
            OpCategory::Data
        }
        Op::Store | Op::Load => OpCategory::Memory,
        Op::Jmp | Op::JmpIf | Op::Call | Op::Ret | Op::Halt | Op::Intuition => {
            OpCategory::ControlFlow
        }
        Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Mod | Op::Pow | Op::Inc | Op::Dec => {
            OpCategory::Arithmetic
        }
        Op::Eq | Op::Gt | Op::Not | Op::And | Op::Or | Op::Xor | Op::IsZero => OpCategory::Logic,
        Op::Reward | Op::Evolve => OpCategory::Meta,
    }
}

pub fn aggregate_trace_logic(trace: &[Op]) -> TraceLogicSummary {
    let mut net_stack_delta = 0isize;
    let mut has_branch = false;

    for &op in trace {
        let info = logic_of(op);
        net_stack_delta += info.stack_delta;
        has_branch |= info.may_branch;
    }

    TraceLogicSummary {
        net_stack_delta,
        has_branch,
    }
}

pub fn decode_ops_for_validation(program: &[f64]) -> Result<Vec<Op>, VMError> {
    let mut ops = Vec::new();
    let mut ip = 0usize;

    while ip < program.len() {
        let raw = program[ip];
        ip += 1;

        if !raw.is_finite() {
            return Err(VMError::InvalidOpcode(-1));
        }
        let rounded = raw.round();
        if (rounded - raw).abs() > 1e-9 {
            return Err(VMError::InvalidOpcode(rounded as i64));
        }

        let opcode = rounded as i64;
        if opcode >= SKILL_OPCODE_BASE {
            continue;
        }

        let Some(op) = Op::from_i64(opcode) else {
            return Err(VMError::InvalidOpcode(opcode));
        };

        ops.push(op);

        if matches!(op, Op::Literal | Op::Jmp | Op::JmpIf | Op::Call) {
            if ip >= program.len() {
                return Err(VMError::InvalidOpcode(opcode));
            }
            ip += 1;
        }
    }

    Ok(ops)
}

pub fn ops_in_categories(categories: &[OpCategory]) -> Vec<Op> {
    all_ops()
        .iter()
        .copied()
        .filter(|op| categories.contains(&category_of(*op)))
        .collect()
}

pub fn all_ops() -> &'static [Op] {
    &[
        Op::Literal,
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Eq,
        Op::Store,
        Op::Load,
        Op::Halt,
        Op::Gt,
        Op::Not,
        Op::Jmp,
        Op::JmpIf,
        Op::Call,
        Op::Ret,
        Op::Intuition,
        Op::Reward,
        Op::Evolve,
        Op::Swap,
        Op::Dup,
        Op::Over,
        Op::Drop,
        Op::And,
        Op::Or,
        Op::Xor,
        Op::IsZero,
        Op::Mod,
        Op::Pow,
        Op::Inc,
        Op::Dec,
        Op::Parse,
        Op::In,
        Op::Out,
    ]
}

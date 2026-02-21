pub mod alphazero;
pub mod eng;
pub mod evolution;
pub mod memory;
pub mod plasticity;
pub mod types;
pub mod vm;
// Add this line to src/lib.rs
pub mod hypothesis;
pub mod intuition;
pub mod logic;
pub mod run;
pub mod token;
pub use memory::MemorySystem;
pub use plasticity::{Event, Plasticity, VMError};
pub use types::{SkillLibrary, UVal};
pub use vm::{CoreMind, Op, SKILL_OPCODE_BASE, SoulGainAgent, SoulGainVM, StepStatus};

pub use logic::{
    LogicInfo, LogicValidationError, OpCategory, TraceLogicSummary, aggregate_trace_logic,
    category_of, logic_of, ops_in_categories, validate_ops,
};

use crate::types::UVal;
use crate::vm::{Op, SoulGainVM};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Hypothesis {
    pub logic: Vec<f64>,
}

impl Hypothesis {
    pub fn generate(target_len: usize, available_skills: &[i64]) -> Self {
        let mut rng = rand::thread_rng();
        let mut logic = Vec::with_capacity(target_len + 4);

        let primitives = [
            Op::Add.as_i64(),
            Op::Sub.as_i64(),
            Op::Mul.as_i64(),
            Op::Div.as_i64(),
            Op::Mod.as_i64(),
            Op::Pow.as_i64(),
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

        while logic.len() < target_len {
            let remaining = target_len - logic.len();

            if remaining >= 6 && rng.gen_bool(0.12) {
                // Compare & Swap template: [Over, Over, Gt, JmpIf(2), Swap]
                logic.push(Op::Over.as_f64());
                logic.push(Op::Over.as_f64());
                logic.push(Op::Gt.as_f64());
                logic.push(Op::JmpIf.as_f64());
                logic.push(2.0);
                logic.push(Op::Swap.as_f64());
                continue;
            }

            if remaining >= 1 && rng.gen_bool(0.12) {
                // Type Cast template: [Parse]
                logic.push(Op::Parse.as_f64());
                continue;
            }

            if !available_skills.is_empty() && rng.gen_bool(0.3) {
                let skill_id = available_skills[rng.gen_range(0..available_skills.len())];
                logic.push(skill_id as f64);
            } else {
                let op = primitives[rng.gen_range(0..primitives.len())];
                logic.push(op as f64);
            }
        }

        Self { logic }
    }
}

pub struct Pruner;

impl Pruner {
    /// Iteratively removes instructions to find the shortest valid logic sequence.
    pub fn prune(
        base_vm: &SoulGainVM,
        original_logic: &[f64],
        input: &[UVal],
        expected: &[UVal],
    ) -> Vec<f64> {
        let mut best_logic = original_logic.to_vec();
        let mut i = 0;

        // Loop through the logic. Try to remove the instruction at index `i`.
        while i < best_logic.len() {
            // Don't prune if we are down to 1 instruction
            if best_logic.len() <= 1 {
                break;
            }

            let mut candidate = best_logic.clone();
            candidate.remove(i);

            // Check if the shorter candidate still produces the EXACT expected output
            if Self::validates(base_vm, &candidate, input, expected) {
                // Success! The instruction was useless (Junk DNA).
                // Keep the shorter version.
                // We do NOT increment 'i' because the next instruction shifted into slot 'i'.
                best_logic = candidate;
            } else {
                // The instruction was necessary. Keep it and move to the next one.
                i += 1;
            }
        }

        best_logic
    }

    fn validates(base_vm: &SoulGainVM, logic: &[f64], input: &[UVal], expected: &[UVal]) -> bool {
        // Create a lightweight VM for testing
        let mut test_vm = SoulGainVM::new(Vec::new());

        // Clone the brain (skills/memory) so the logic has context
        test_vm.skills = base_vm.skills.clone();
        test_vm.memory = base_vm.memory.clone();
        test_vm.plasticity = base_vm.plasticity.clone();

        // Load Input
        for val in input {
            test_vm.stack.push(val.clone());
        }

        // Setup Program
        test_vm.program = logic.to_vec();

        // Ensure Halt exists for safety
        if test_vm.program.last() != Some(&Op::Halt.as_f64()) {
            test_vm.program.push(Op::Halt.as_f64());
        }

        test_vm.run(5000); // Give it enough fuel

        // Strict Check: Stack must match expected output EXACTLY
        if test_vm.stack.len() != expected.len() {
            return false;
        }
        for (a, b) in test_vm.stack.iter().zip(expected.iter()) {
            if a != b {
                return false;
            }
        }
        true
    }
}

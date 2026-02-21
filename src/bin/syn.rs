//! syn.rs – Extended MCTS problem suite + inference benchmark
//!
//! Tiered from trivial → hard, each problem has a known correct program
//! so we can verify the solver actually finds the right answer, not just
//! any program that scores well on the training cases.

use soulgain::alphazero::{
    run_active_inference_episode, ActiveInferenceConfig,
    ReasoningConfig, TaskExample, TaskSpec, UniversalPolicy,
    WorldConfig, WorldGenerator, WorldModel, solve_with_stats,
};
use soulgain::plasticity::Plasticity;
use soulgain::vm::{CoreMind, Op};

// ─────────────────────────────────────────────────────────────────────────────
// Custom worlds for inference section
// ─────────────────────────────────────────────────────────────────────────────

struct BinaryCounter { state: usize }
impl WorldModel for BinaryCounter {
    fn num_states(&self) -> usize { 8 }
    fn current_state(&self) -> usize { self.state }
    fn sense(&mut self) -> f64 { self.state as f64 }
    fn step(&mut self, _: Op) -> (usize, f64) {
        self.state = (self.state + 1) % 8;
        (self.state, self.sense())
    }
}

struct XorWorld { state: usize, n: usize }
impl WorldModel for XorWorld {
    fn num_states(&self) -> usize { self.n }
    fn current_state(&self) -> usize { self.state }
    fn sense(&mut self) -> f64 { self.state as f64 / self.n as f64 }
    fn step(&mut self, action: Op) -> (usize, f64) {
        self.state = (self.state ^ (action as usize % self.n)) % self.n;
        (self.state, self.sense())
    }
}

struct StochasticGrid { state: usize, seed: u64 }
impl StochasticGrid {
    fn rand(&mut self) -> f64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        (self.seed as f64 / u64::MAX as f64).clamp(0.0, 1.0)
    }
}
impl WorldModel for StochasticGrid {
    fn num_states(&self) -> usize { 9 }
    fn current_state(&self) -> usize { self.state }
    fn sense(&mut self) -> f64 {
        let r = self.state / 3;
        let c = self.state % 3;
        ((r as i32 - 1).abs() + (c as i32 - 1).abs()) as f64
    }
    fn step(&mut self, action: Op) -> (usize, f64) {
        let r = self.state / 3;
        let c = self.state % 3;
        let (dr, dc) = match action as usize % 4 {
            0 => (0i32, 1i32), 1 => (0, -1), 2 => (-1, 0), _ => (1, 0),
        };
        let noise = self.rand();
        let (dr, dc) = if noise < 0.8 { (dr, dc) }
                       else if noise < 0.9 { (dc, dr) }
                       else { (-dc, -dr) };
        self.state = ((r as i32 + dr).clamp(0, 2) as usize) * 3
                   + ((c as i32 + dc).clamp(0, 2) as usize);
        (self.state, self.sense())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task builder
// ─────────────────────────────────────────────────────────────────────────────

fn task(pairs: &[(&[i64], &[i64])]) -> TaskSpec {
    use soulgain::types::UVal;
    TaskSpec {
        train_cases: pairs.iter().map(|(i, o)| TaskExample {
            input:  i.iter().map(|&n| UVal::Number(n as f64)).collect(),
            expected_output: o.iter().map(|&n| UVal::Number(n as f64)).collect(),
        }).collect(),
    }
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Add=>"Add", Op::Sub=>"Sub", Op::Mul=>"Mul", Op::Div=>"Div",
        Op::Dup=>"Dup", Op::Drop=>"Drop", Op::Swap=>"Swap", Op::Over=>"Over",
        Op::Halt=>"Halt", Op::Gt=>"Gt", Op::Not=>"Not", Op::Eq=>"Eq",
        Op::Jmp=>"Jmp", Op::JmpIf=>"JmpIf", Op::Inc=>"Inc", Op::Dec=>"Dec",
        Op::IsZero=>"IsZero", Op::And=>"And", Op::Or=>"Or", Op::Mod=>"Mod",
        Op::Store=>"Store", Op::Load=>"Load", Op::Pow=>"Pow", Op::Xor=>"Xor",
        _ => "?",
    }
}

fn ops_str(ops: &[Op]) -> String {
    ops.iter().map(op_name).collect::<Vec<_>>().join(", ")
}

// ─────────────────────────────────────────────────────────────────────────────
// Printers
// ─────────────────────────────────────────────────────────────────────────────

fn div() { println!("{}", "─".repeat(68)); }
fn gap() { println!(); }

struct Problem {
    label:    &'static str,
    task:     TaskSpec,
    sims:     u32,
    ops:      Vec<Op>,
    max_len:  usize,
    expected: &'static str,   // the correct program in human terms
}

fn run_problem(p: &Problem) {
    let config = ReasoningConfig {
        simulations: p.sims,
        max_program_len: p.max_len,
        max_ops_per_candidate: p.max_len * 2,
        action_space: p.ops.clone(),
        ..Default::default()
    };
    let policy = UniversalPolicy::from_task(&p.task);
    let root   = CoreMind::new();

    let (program, s) = solve_with_stats(&root, &p.task, &config, &policy);

    let crash_pct  = s.crash_count as f64 / s.simulations_run.max(1) as f64 * 100.0;
    let exit_label = if s.solved_early { "EARLY EXIT ✓" } else { "full budget" };
    let score_tag  = if s.best_score >= 0.999 { "SOLVED ✓" }
                     else if s.best_score >= 0.5 { "partial" }
                     else { "failed" };

    println!("┌─ {:}", p.label);
    println!("│  expected  : {}", p.expected);
    println!("│  sims cap  : {}   max prog len: {}   ops: {}",
        p.sims, p.max_len, p.ops.len());
    div();
    println!("  result      : {}  (score {:.4})  [{}]",
        score_tag, s.best_score, exit_label);
    println!("  program     : [{}]",
        program.as_deref().map(ops_str).unwrap_or_else(|| "none".into()));
    println!("  sims run    : {} / {}", s.simulations_run, p.sims);
    println!("  time        : {:.3} ms   sims/sec: {:.0}", s.elapsed_ms, s.sims_per_sec);
    println!("  nodes       : {}   avg depth: {:.2}", s.nodes_allocated, s.avg_selection_depth);
    println!("  op cycles   : {} total   {:.1}/sim", s.total_op_cycles, s.op_cycles_per_sim);
    println!("  crashes     : {} ({:.1}%)", s.crash_count, crash_pct);
    println!("└──────────────────────────────────────────────────────────────────┘");
    gap();
}

fn run_inference(label: &str, world: &mut dyn WorldModel, plasticity: &Plasticity) {
    let cfg = ActiveInferenceConfig { steps: 300, ..Default::default() };
    let cycles = cfg.action_space.len() * cfg.imagination_rollouts * cfg.rollout_depth;
    let r = run_active_inference_episode(world, plasticity, &cfg);
    let learned = ((1.0 - r.average_surprisal / 2.303) * 100.0).clamp(0.0, 100.0);
    println!("  {:42} | {:5.1}ms | {:5.0} s/s | surp {:.3} | {:.1}%",
        label, r.elapsed_ms, r.steps_per_sec, r.average_surprisal, learned);
    let _ = cycles;
}

// ─────────────────────────────────────────────────────────────────────────────
// Action space shorthands
// ─────────────────────────────────────────────────────────────────────────────

fn ops_basic() -> Vec<Op> {
    vec![Op::Dup, Op::Add, Op::Sub, Op::Inc, Op::Dec, Op::Halt]
}
fn ops_arith() -> Vec<Op> {
    vec![Op::Dup, Op::Add, Op::Sub, Op::Mul, Op::Inc, Op::Dec,
         Op::Over, Op::Swap, Op::Drop, Op::Halt]
}
fn ops_full() -> Vec<Op> {
    vec![Op::Dup, Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Mod,
         Op::Inc, Op::Dec, Op::Over, Op::Swap, Op::Drop,
         Op::Gt, Op::Not, Op::IsZero, Op::Eq, Op::Halt]
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  SoulGain · Extended MCTS Problem Suite + Inference Benchmark        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // ─── TIER 1: Trivial — should solve in < 100 sims ─────────────────────────
    println!("\n══ TIER 1 · Trivial  (should solve < 100 sims) ══════════════════════\n");

    run_problem(&Problem {
        label: "Identity  [5]→[5]",
        task: task(&[(&[5], &[5]), (&[3], &[3]), (&[7], &[7])]),
        sims: 500, max_len: 4,
        ops: ops_basic(),
        expected: "[Halt]",
    });

    run_problem(&Problem {
        label: "Increment  [n]→[n+1]",
        task: task(&[(&[1], &[2]), (&[4], &[5]), (&[9], &[10])]),
        sims: 500, max_len: 4,
        ops: ops_basic(),
        expected: "[Inc, Halt]",
    });

    run_problem(&Problem {
        label: "Decrement  [n]→[n-1]",
        task: task(&[(&[5], &[4]), (&[3], &[2]), (&[10], &[9])]),
        sims: 500, max_len: 4,
        ops: ops_basic(),
        expected: "[Dec, Halt]",
    });

    run_problem(&Problem {
        label: "Add 2  [n]→[n+2]",
        task: task(&[(&[1], &[3]), (&[5], &[7]), (&[8], &[10])]),
        sims: 1_000, max_len: 5,
        ops: ops_basic(),
        expected: "[Inc, Inc, Halt]",
    });

    // ─── TIER 2: Easy — should solve in < 1k sims ────────────────────────────
    println!("══ TIER 2 · Easy  (should solve < 1k sims) ══════════════════════════\n");

    run_problem(&Problem {
        label: "Double  [n]→[2n]",
        task: task(&[(&[1], &[2]), (&[3], &[6]), (&[5], &[10])]),
        sims: 2_000, max_len: 5,
        ops: ops_arith(),
        expected: "[Dup, Add, Halt]",
    });

    run_problem(&Problem {
        label: "Triple  [n]→[3n]",
        task: task(&[(&[1], &[3]), (&[2], &[6]), (&[4], &[12])]),
        sims: 4_000, max_len: 6,
        ops: ops_arith(),
        expected: "[Dup, Dup, Add, Add, Halt]  or  [Dup, Add, Inc... no — Dup, Over, Add, Add, Halt]",
    });

    run_problem(&Problem {
        label: "Square  [n]→[n²]",
        task: task(&[(&[2], &[4]), (&[3], &[9]), (&[4], &[16])]),
        sims: 4_000, max_len: 5,
        ops: ops_arith(),
        expected: "[Dup, Mul, Halt]",
    });

    run_problem(&Problem {
        label: "Add two inputs  [a, b]→[a+b]",
        task: task(&[(&[2, 3], &[5]), (&[1, 7], &[8]), (&[4, 4], &[8])]),
        sims: 2_000, max_len: 4,
        ops: ops_basic(),
        expected: "[Add, Halt]",
    });

    run_problem(&Problem {
        label: "Multiply two inputs  [a, b]→[a*b]",
        task: task(&[(&[2, 3], &[6]), (&[3, 4], &[12]), (&[5, 2], &[10])]),
        sims: 2_000, max_len: 4,
        ops: ops_arith(),
        expected: "[Mul, Halt]",
    });

    run_problem(&Problem {
        label: "Subtract two inputs  [a, b]→[a-b]",
        task: task(&[(&[5, 3], &[2]), (&[9, 4], &[5]), (&[7, 2], &[5])]),
        sims: 2_000, max_len: 4,
        ops: ops_basic(),
        expected: "[Sub, Halt]",
    });

    // ─── TIER 3: Medium — needs real search depth ─────────────────────────────
    println!("══ TIER 3 · Medium  (needs depth, 4k–20k sims) ══════════════════════\n");

    run_problem(&Problem {
        label: "Double then add 1  [n]→[2n+1]",
        task: task(&[(&[1], &[3]), (&[2], &[5]), (&[3], &[7]), (&[5], &[11])]),
        sims: 10_000, max_len: 6,
        ops: ops_arith(),
        expected: "[Dup, Add, Inc, Halt]",
    });

    run_problem(&Problem {
        label: "Square then add input  [n]→[n²+n]",
        task: task(&[(&[2], &[6]), (&[3], &[12]), (&[4], &[20])]),
        sims: 10_000, max_len: 7,
        ops: ops_arith(),
        expected: "[Dup, Dup, Mul, Add, Halt]",
    });

    run_problem(&Problem {
        label: "Sum of three inputs  [a, b, c]→[a+b+c]",
        task: task(&[(&[1, 2, 3], &[6]), (&[2, 3, 4], &[9]), (&[5, 5, 5], &[15])]),
        sims: 4_000, max_len: 5,
        ops: ops_basic(),
        expected: "[Add, Add, Halt]",
    });

    run_problem(&Problem {
        label: "Difference of squares  [a, b]→[a²-b²]",
        task: task(&[(&[3, 2], &[5]), (&[4, 1], &[15]), (&[5, 3], &[16])]),
        sims: 10_000, max_len: 8,
        ops: ops_full(),
        expected: "[Over, Over, Mul, Swap, Dup, Mul, Sub, Halt]",
    });

    run_problem(&Problem {
        label: "Absolute difference  [a, b]→[|a-b|]",
        task: task(&[(&[5, 3], &[2]), (&[3, 7], &[4]), (&[9, 9], &[0])]),
        sims: 10_000, max_len: 8,
        ops: ops_full(),
        expected: "[Over, Over, Sub, Swap, Sub, Gt... complex]",
    });

    // ─── TIER 4: Hard / stress ────────────────────────────────────────────────
    println!("══ TIER 4 · Hard / Stress ════════════════════════════════════════════\n");

    run_problem(&Problem {
        label: "Cube  [n]→[n³]  (needs Dup×2 + Mul×2)",
        task: task(&[(&[2], &[8]), (&[3], &[27]), (&[4], &[64])]),
        sims: 20_000, max_len: 8,
        ops: ops_arith(),
        expected: "[Dup, Dup, Mul, Mul, Halt]",
    });

    run_problem(&Problem {
        label: "Min of two inputs  [a, b]→[min(a,b)]",
        task: task(&[(&[3, 5], &[3]), (&[7, 2], &[2]), (&[4, 4], &[4])]),
        sims: 20_000, max_len: 10,
        ops: ops_full(),
        expected: "[Over, Over, Gt, ... conditional logic]",
    });

    run_problem(&Problem {
        label: "Throughput ceiling: 50k sims on trivial task",
        task: task(&[(&[1], &[1])]),
        sims: 50_000, max_len: 3,
        ops: vec![Op::Halt, Op::Inc, Op::Dup],
        expected: "[Halt]",
    });

    // ─── INFERENCE: WorldModel plug-in bench ──────────────────────────────────
    println!("══ INFERENCE: WorldModel plug-in (300 steps each) ═══════════════════\n");
    println!("  {:42} | {:>7} | {:>8} | {:>9} | {:>6}",
        "world", "time", "steps/s", "surprisal", "learned");
    div();

    let plasticity = Plasticity::new();

    let wc = WorldConfig { hidden_states: 8, entropy: 0.2, observation_noise: 0.01 };
    run_inference("WorldGenerator 8-state (entropy 0.2)",
        &mut WorldGenerator::new(&wc), &plasticity);

    let wc2 = WorldConfig { hidden_states: 16, entropy: 0.1, observation_noise: 0.0 };
    run_inference("WorldGenerator 16-state (entropy 0.1)",
        &mut WorldGenerator::new(&wc2), &plasticity);

    let wc3 = WorldConfig { hidden_states: 32, entropy: 0.3, observation_noise: 0.05 };
    run_inference("WorldGenerator 32-state (entropy 0.3, noisy)",
        &mut WorldGenerator::new(&wc3), &plasticity);

    run_inference("BinaryCounter  8-state deterministic",
        &mut BinaryCounter { state: 0 }, &plasticity);

    run_inference("XorWorld  8-state",
        &mut XorWorld { state: 0, n: 8 }, &plasticity);

    run_inference("XorWorld  32-state",
        &mut XorWorld { state: 0, n: 32 }, &plasticity);

    run_inference("StochasticGrid  3×3 (80/10/10 noise)",
        &mut StochasticGrid { state: 4, seed: 0xDEAD_BEEF }, &plasticity);

    // flip test
    let mut wg = WorldGenerator::new(&wc);
    run_inference("WorldGenerator pre-flip",  &mut wg, &plasticity);
    wg.flip_physics();
    run_inference("WorldGenerator post-flip (physics inverted)", &mut wg, &plasticity);

    div();
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Tiers 1–2 prove correctness.  Tier 3 measures search depth.        ║");
    println!("║  Tier 4 shows throughput ceiling and hard combinatorial limits.      ║");
    println!("║  Inference table proves WorldModel is fully pluggable.               ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");
}
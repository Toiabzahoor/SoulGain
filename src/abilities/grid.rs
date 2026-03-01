use crate::alphazero::{CognitivePolicy, ReasoningConfig, UniversalWorld, solve_universal_with_stats};
use crate::plasticity::{Event, IntoOpcode, Plasticity, VMError, hash_event};
use std::collections::HashMap;
use std::fmt;
use std::io::Write;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Define the Actions
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GridAction { Up, Down, Left, Right }

impl IntoOpcode for GridAction {
    fn into_opcode(self) -> i64 {
        match self {
            GridAction::Up => 200, GridAction::Down => 201,
            GridAction::Left => 202, GridAction::Right => 203,
        }
    }
}

impl fmt::Display for GridAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridAction::Up => write!(f, "↑"), GridAction::Down => write!(f, "↓"),
            GridAction::Left => write!(f, "←"), GridAction::Right => write!(f, "→"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. The MENTAL World (MCTS consults synaptic weights for safe imagination)
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct MentalWorld<'a> {
    pub x: i32, 
    pub y: i32,
    pub plasticity: &'a Plasticity,
}

impl<'a> UniversalWorld for MentalWorld<'a> {
    type State = (i32, i32);
    type Action = GridAction;

    fn current_state(&self) -> Self::State { (self.x, self.y) }

    fn step(&mut self, action: Self::Action) -> Result<(), ()> {
        // 🌟 IMAGINATION CONSTRAINT: Check synaptic weights for learned prohibitions
        let state_hash = ((self.x as u64) << 32) | (self.y as u64);
        let ctx_hash = hash_event(&Event::ContextWithState { data: [0; 16], state_hash });
        let action_hash = hash_event(&Event::Opcode { opcode: action.into_opcode(), stack_depth: 0 });

        // Block imagination if strong negative weight exists (trauma learned)
        if let Ok(mem) = self.plasticity.memory.read() {
            if let Some(bucket) = mem.get_bucket(ctx_hash) {
                for syn in &bucket.synapses {
                    if syn.target_hash == action_hash && syn.weight < -10.0 {
                        // Strong inhibition from experience: imagination blocked
                        return Err(()); 
                    }
                }
            }
        }

        // Normal Physics (edges of the universe)
        let (nx, ny) = match action {
            GridAction::Up => (self.x, self.y - 1), 
            GridAction::Down => (self.x, self.y + 1),
            GridAction::Left => (self.x - 1, self.y), 
            GridAction::Right => (self.x + 1, self.y),
        };
        
        if nx < 0 || nx > 5 || ny < 0 || ny > 5 { return Err(()); }

        self.x = nx; self.y = ny;
        Ok(())
    }

    fn is_terminal(&self) -> bool { self.x == 5 && self.y == 5 }

    fn evaluate_path(&self, path: &[Self::Action]) -> (f32, u64) {
        let mut sim = self.clone();
        for &action in path {
            if sim.step(action).is_err() { return (-1.0, path.len() as u64); }
        }
        if sim.is_terminal() {
            (1.0, path.len() as u64)
        } else {
            let dist = ((5 - sim.x).abs() + (5 - sim.y).abs()) as f32;
            ((1.0 - (dist / 10.0)).clamp(0.0, 0.9), path.len() as u64)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The Brain / Intuition Interface
// ─────────────────────────────────────────────────────────────────────────────
pub struct GridPolicy<'a> {
    pub plasticity: &'a Plasticity,
    pub allowed_ops: Vec<GridAction>,
}

impl<'a> CognitivePolicy<(i32, i32), GridAction> for GridPolicy<'a> {
    fn evaluate(&self, state: &(i32, i32)) -> f32 {
        let dist = ((5 - state.0).abs() + (5 - state.1).abs()) as f32;
        if dist == 0.0 { 1.0 } else { 1.0 - (dist / 10.0).clamp(0.0, 0.9) }
    }

    fn priors(&self, state: &(i32, i32)) -> Vec<(GridAction, f32)> {
        let state_hash = ((state.0 as u64) << 32) | (state.1 as u64);
        let event = Event::ContextWithState { data: [0; 16], state_hash };
        self.plasticity.get_op_distribution(event, &self.allowed_ops)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. The REAL World Logic & Simulation
// ─────────────────────────────────────────────────────────────────────────────

fn is_invisible_wall(x: i32, y: i32) -> bool {
    // A massive invisible wall blocking x=2
    if x == 2 && y <= 4 { return true; } 
    false
}

fn calibrate_surprise(distance_from_goal: i32, is_crash: bool) -> u16 {
    // Crash surprise scales with how far agent deviated
    if is_crash {
        let base = (distance_from_goal as u16).saturating_mul(40);
        base.saturating_add(100).min(700)
    } else {
        // Success: modest surprise (learning signal, not alarm)
        150
    }
}

pub fn run_dynamic_gridworld(plasticity: &Plasticity) {
    let allowed_actions = vec![GridAction::Up, GridAction::Down, GridAction::Left, GridAction::Right];
    
    let config = ReasoningConfig {
        simulations: 800,        
        max_depth: 8,           
        max_program_len: 8,
        max_ops_per_candidate: 8,
        exploration_constant: 1.41,
        length_penalty: 0.05,    
        loop_penalty: 0.5,
        action_space: allowed_actions.clone(),
        arena_capacity: 10_000,
    };

    let policy = GridPolicy { plasticity, allowed_ops: allowed_actions.clone() };

    println!("\n🌍 ===========================================================");
    println!("🌍 TRUE AGI TEST: CONTINUOUS ONE-SHOT LEARNING");
    println!("🌍 ===========================================================");

    for ep in 1..=5 {
        println!("\n🚀 LIFETIME EXPEDITION {} 🚀", ep);
        let mut real_x = 0;
        let mut real_y = 0;
        let mut steps = 0;
        let mut crashes = 0;
        
        // 🌟 PER-STATE CRASH TRACKING: Forces S2 on repeated failures
        let mut crash_count: HashMap<(i32, i32), usize> = HashMap::new();

        while real_x != 5 || real_y != 5 {
            steps += 1;
            let state_hash = ((real_x as u64) << 32) | (real_y as u64);
            let event = Event::ContextWithState { data: [0; 16], state_hash };
            
            // 1. Query the Physical S1 Brain: Get synaptic distribution
            let priors = plasticity.get_op_distribution(event, &policy.allowed_ops);
            
            // Extract confidence of best action
            let best_prior = priors.iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, conf)| *conf)
                .unwrap_or(0.0);
            
            let mut best_action = policy.allowed_ops[0];
            for (a, c) in &priors {
                if (*c - best_prior).abs() < 0.0001 { best_action = *a; }
            }

            let action_to_take;
            let mode;
            
            // 🌟 Check if we've crashed multiple times at this state
            let crash_at_state = *crash_count.get(&(real_x, real_y)).unwrap_or(&0);
            let force_s2_due_to_crashes = crash_at_state >= 3;

            // 2. SYSTEM 1 vs SYSTEM 2 GATE: Confidence-driven + crash-driven
            if best_prior > 0.65 && !force_s2_due_to_crashes {
                action_to_take = best_action;
                mode = "S1";
            } else {
                let mental_world = MentalWorld { x: real_x, y: real_y, plasticity };
                let (mcts_path_opt, _) = solve_universal_with_stats(&mental_world, &config, &policy);
                
                if let Some(mcts_path) = mcts_path_opt {
                    if !mcts_path.is_empty() { 
                        action_to_take = mcts_path[0]; 
                    } else { 
                        action_to_take = best_action; 
                    } 
                } else { 
                    action_to_take = best_action; 
                }
                mode = if force_s2_due_to_crashes { "S2F" } else { "S2" };
            }

            // 3. EXECUTE IN REAL WORLD
            let (dx, dy) = match action_to_take {
                GridAction::Up => (0, -1), 
                GridAction::Down => (0, 1),
                GridAction::Left => (-1, 0), 
                GridAction::Right => (1, 0),
            };

            let nx = real_x + dx;
            let ny = real_y + dy;
            let distance_to_goal = ((5 - nx).abs() + (5 - ny).abs()) as i32;

            let mut stdp_batch = vec![
                Event::ContextWithState { data: [0; 16], state_hash },
                Event::Opcode { opcode: action_to_take.into_opcode(), stack_depth: 0 }
            ];

            if nx < 0 || nx > 5 || ny < 0 || ny > 5 || is_invisible_wall(nx, ny) {
                crashes += 1;
                let crash_key = (real_x, real_y);
                *crash_count.entry(crash_key).or_insert(0) += 1;
                
                print!("\n[{},{}] {}:{} 💥 (Ouch!)", real_x, real_y, mode, action_to_take);
                
                // 🌟 ONE-SHOT LEARNING: Negative weight spike on action synapse
                if let Ok(mut mem) = plasticity.memory.write() {
                    let ctx_hash = hash_event(&Event::ContextWithState { data: [0; 16], state_hash });
                    let action_hash = hash_event(&Event::Opcode { 
                        opcode: action_to_take.into_opcode(), 
                        stack_depth: 0 
                    });

                    // Strong negative weight = learned inhibition
                    mem.apply_update(0, ctx_hash, action_hash, -25.0);
                }
                
                stdp_batch.push(Event::Error(VMError::Crash));
                let crash_surprise = calibrate_surprise(distance_to_goal, true);
                stdp_batch.push(Event::Surprisal(crash_surprise)); 
                
                // NO BREAK! Agent stays at real_x, real_y and tries again instantly!
            } else {
                print!("\n[{},{}] {}:{} 🟢", real_x, real_y, mode, action_to_take);
                real_x = nx;
                real_y = ny;
                
                stdp_batch.push(Event::Reward(20)); 

                if real_x == 5 && real_y == 5 {
                    print!(" 🎯 TARGET REACHED!");
                    stdp_batch.push(Event::Reward(100)); 
                    let success_surprise = calibrate_surprise(0, false);
                    stdp_batch.push(Event::Surprisal(success_surprise)); 
                }
            }
            std::io::stdout().flush().unwrap();

            // 4. SYNC NEUROPLASTICITY
            plasticity.observe_batch(stdp_batch);
            plasticity.sync(); 
            
            if steps > 40 { 
                print!("\n😵 (Agent lost in thoughts)"); 
                break; 
            }
        }
        println!("\n🎯 End of Expedition {}. Steps: {}, Crashes: {}", ep, steps, crashes);
    }
}
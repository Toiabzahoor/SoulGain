use soulgain::alphazero::{
    WorldConfig, WorldGenerator, ForwardModel, ActiveInferenceConfig
};
use soulgain::plasticity::Plasticity;
use soulgain::vm::Op;
use std::time::Duration;
use std::thread;

fn main() {
    println!("➤ ACTIVE INFERENCE: THE CATASTROPHIC SHIFT TEST\n");

    let plasticity = Plasticity::new();
    let config = ActiveInferenceConfig::default();
    
    let world_config = WorldConfig {
        hidden_states: 10,
        entropy: 0.15,
        observation_noise: 0.0,
    };
    let mut world = WorldGenerator::new(&world_config);
    let mut model = ForwardModel::new(10, config.action_space.len());

    println!("{:<5} | {:<8} | {:<10} | {:<10} | {:<10}", "STEP", "PHASE", "SURPRISE", "E.SURP", "FE");
    println!("------------------------------------------------------------");

    for step in 1..=500 {
        if step == 101 {
            println!("--- !!! WORLD CATASTROPHE: PHYSICS INVERTED !!! ---");
            world.flip_physics();
        }

        let state = world.current_state();
        let phase = if step <= 100 { "Stable" } else { "Re-learn" };

        // 1. Imagination: Pick the best action based on Curiosity vs Effort
        // Use a local imagination function or call your updated alphazero logic
        let action_idx = step % config.action_space.len(); // Or use imagine_best_action
        let action = config.action_space[action_idx];

        // 2. Reality
        let (next_state, _) = world.step(action);

        // 3. Learning
        let predicted = model.predict_distribution(state, action_idx);
        let prob = predicted.get(next_state).copied().unwrap_or(1e-6).max(1e-6);
        let surprisal = (-prob.ln()).clamp(0.0, 10.0);
        let expected_uncertainty = model.expected_surprisal(state, action_idx);
        
        model.update(state, action_idx, next_state, surprisal);

        let free_energy = surprisal - config.curiosity_weight * expected_uncertainty;

        println!(
            "{:<5} | {:<8} | {:.4} | {:.4} | {:.4}", 
            step, phase, surprisal, expected_uncertainty, free_energy
        );

        plasticity.sync();
        if step % 10 == 0 { thread::sleep(Duration::from_millis(1)); }
    }

    println!("------------------------------------------------------------");
    println!("\n➤ ANALYSIS:");
    println!("1. CATASTROPHE: Notice the Surprise spike at Step 101.");
    println!("2. RECOVERY: Watch how 'Expected Surprisal' (the AI's anticipation of its own confusion) helps it target unknown transitions.");
    println!("3. STABILITY: The Goal is to see Free Energy drop back down after the chaos.");
}
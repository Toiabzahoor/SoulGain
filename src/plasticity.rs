use crate::memory::PersistentMemory;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};

// Any module can implement this to talk to the brain
pub trait IntoOpcode {
    fn into_opcode(self) -> i64;
}

impl IntoOpcode for usize {
    fn into_opcode(self) -> i64 {
        self as i64
    }
}

// --- STDP CONSTANTS ---
const A_PLUS: f32 = 0.08;
const A_MINUS: f32 = 0.10;
const TAU: f32 = 0.025;
const WINDOW_S: f64 = 0.15;
const REWARD_BOOST: f32 = 0.6;
const SURPRISE_LOW: f64 = 0.25;
const SURPRISE_HIGH: f64 = 0.75;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum VMError {
    StackUnderflow,
    InvalidOpcode(i64),
    InvalidJump(i64),
    ReturnStackUnderflow,
    InvalidEvolve(i64),
    Crash,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Event {
    Opcode { opcode: i64, stack_depth: usize },
    Context(u64),
    ContextWithState { data: [u64; 16], state_hash: u64 },
    MemoryRead,
    MemoryWrite,
    Reward(i16),
    Surprisal(u16),
    Error(VMError),
}

pub fn hash_event(event: &Event) -> u64 {
    let mut hasher = DefaultHasher::new();
    event.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone)]
pub struct Plasticity {
    sender: mpsc::Sender<PlasticityMessage>,
    pub memory: Arc<RwLock<PersistentMemory>>,
}

enum PlasticityMessage {
    Single(Event, Instant),
    Batch(Vec<Event>),
    Sync(mpsc::Sender<()>),
}

impl Plasticity {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<PlasticityMessage>();
        
        let memory: Arc<RwLock<PersistentMemory>> = match PersistentMemory::new() {
            Ok(mem) => Arc::new(RwLock::new(mem)),
            Err(e) => {
                println!("🚨 Failed to initialize brain arena: {}", e);
                panic!("Cannot start without persistent memory");
            }
        };
        
        let mem_clone = memory.clone();

        thread::spawn(move || {
            let mut recent_events: Vec<(Event, Instant)> = Vec::new();
            let mut surprise_gate = 0.5;

            let mut process_event = |current_event: Event, current_time: Instant, recent_events: &mut Vec<(Event, Instant)>| {
                recent_events.retain(|(_, t)| current_time.duration_since(*t).as_secs_f64() < WINDOW_S);

                if let Event::Surprisal(raw) = current_event {
                    surprise_gate = (raw as f64 / 1000.0).clamp(0.0, 1.0);
                    recent_events.push((current_event, current_time));
                    return;
                }

                let mut updates: Vec<(Event, Event, f32)> = Vec::new();

                let (ltp_multiplier, ltd_multiplier) = if surprise_gate <= SURPRISE_LOW {
                    (0.3, 2.5)
                } else if surprise_gate >= SURPRISE_HIGH {
                    (3.0, 0.2)
                } else {
                    (1.0, 1.0)
                };

                for (past_event, past_time) in recent_events.iter() {
                    let delta_t = current_time.duration_since(*past_time).as_secs_f64() as f32;
                    if delta_t <= 0.0 || delta_t >= WINDOW_S as f32 { continue; }

                    match current_event {
                        Event::Reward(intensity) => {
                            let scale = (intensity.abs() as f32 / 100.0).clamp(0.0, 1.0);
                            if intensity > 0 {
                                let reward_change = (REWARD_BOOST * scale) * (-delta_t / TAU).exp();
                                updates.push((*past_event, current_event, reward_change));
                            }
                            continue;
                        }
                        Event::Error(_) => {
                            let penalty = -REWARD_BOOST * (-delta_t / TAU).exp();
                            updates.push((*past_event, current_event, penalty));
                            continue;
                        }
                        _ => {}
                    }

                    let ltp_change = A_PLUS * ltp_multiplier * (-delta_t / TAU).exp();
                    updates.push((*past_event, current_event, ltp_change));

                    let ltd_change = A_MINUS * ltd_multiplier * (-delta_t / TAU).exp();
                    updates.push((current_event, *past_event, -ltd_change));
                }

                if !updates.is_empty() {
                    if let Ok(mut mem) = mem_clone.write() {
                        for (from, to, delta) in updates {
                            let from_hash = hash_event(&from);
                            let to_hash = hash_event(&to);
                            
                            let domain = if let Event::ContextWithState { state_hash, .. } = from {
                                state_hash
                            } else {
                                0
                            };
                            
                            mem.apply_update(domain, from_hash, to_hash, delta);
                        }
                    }
                }

                recent_events.push((current_event, current_time));
            };

            while let Ok(message) = rx.recv() {
                match message {
                    PlasticityMessage::Single(event, time) => { 
                        process_event(event, time, &mut recent_events); 
                    }
                    PlasticityMessage::Batch(events) => {
                        if events.is_empty() { continue; }
                        let now = Instant::now();
                        let len = events.len();
                        let step = if len > 1 { WINDOW_S / (len as f64) } else { 0.0 };

                        for (idx, event) in events.into_iter().enumerate() {
                            let offset = (len - 1 - idx) as f64 * step;
                            let event_time = if offset > 0.0 { now - Duration::from_secs_f64(offset) } else { now };
                            process_event(event, event_time, &mut recent_events);
                        }
                    }
                    PlasticityMessage::Sync(reply_tx) => { 
                        let _ = reply_tx.send(()); 
                    }
                }
            }
        });

        Self { sender: tx, memory }
    }

    pub fn observe(&self, event: Event) {
        let _ = self.sender.send(PlasticityMessage::Single(event, Instant::now()));
    }

    pub fn observe_batch(&self, events: Vec<Event>) {
        let _ = self.sender.send(PlasticityMessage::Batch(events));
    }

    pub fn sync(&self) {
        let (tx, rx) = mpsc::channel();
        let _ = self.sender.send(PlasticityMessage::Sync(tx));
        let _ = rx.recv();
    }

    pub fn best_next_event(&self, from: Event) -> Option<(u64, f64)> {
        let mem = self.memory.read().ok()?;
        let from_hash = hash_event(&from);
        let bucket = mem.get_bucket(from_hash)?;

        let mut total_weight: f32 = 0.0;
        let mut best_target = 0u64;
        let mut best_weight = -1.0f32;

        for syn in &bucket.synapses {
            if syn.target_hash != 0 {
                total_weight += syn.weight.max(0.0);
                if syn.weight > best_weight {
                    best_weight = syn.weight;
                    best_target = syn.target_hash;
                }
            }
        }

        if best_target != 0 {
            let confidence = if total_weight > 0.0 {
                (best_weight / total_weight).clamp(0.0, 1.0) as f64
            } else {
                0.0
            };
            Some((best_target, confidence))
        } else {
            None
        }
    }

    pub fn get_op_distribution<A: Copy + IntoOpcode>(
        &self, 
        context_event: Event, 
        allowed_ops: &[A]
    ) -> Vec<(A, f32)> {
        let context_hash = hash_event(&context_event);
        
        let mem = match self.memory.read() {
            Ok(guard) => guard,
            Err(_) => {
                return allowed_ops
                    .iter()
                    .map(|&op| (op, 1.0 / allowed_ops.len() as f32))
                    .collect();
            }
        };

        let mut priors = Vec::with_capacity(allowed_ops.len());
        let mut sum: f32 = 0.0;
        let temperature: f32 = 1.5;

        if let Some(bucket) = mem.get_bucket(context_hash) {
            for &op in allowed_ops {
                let target_op_id = op.into_opcode();
                let target_hash = hash_event(&Event::Opcode { 
                    opcode: target_op_id, 
                    stack_depth: 0 
                });
                
                let mut weight: f32 = 0.02;
                
                for syn in &bucket.synapses {
                    if syn.target_hash == target_hash {
                        weight = syn.weight;
                        break;
                    }
                }
                
                let w = if weight < 0.0 {
                    0.001
                } else {
                    weight.max(0.02)
                };
                
                let score = w.powf(temperature);
                priors.push((op, score));
                sum += score;
            }
        } else {
            let uniform = 1.0 / allowed_ops.len() as f32;
            for &op in allowed_ops { 
                priors.push((op, uniform)); 
            }
            return priors;
        }

        if sum > 0.0 {
            for (_, score) in &mut priors { 
                *score /= sum; 
            }
        } else {
            let uniform = 1.0 / allowed_ops.len() as f32;
            for (_, score) in &mut priors { 
                *score = uniform; 
            }
        }

        priors
    }
}
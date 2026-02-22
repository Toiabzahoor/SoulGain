use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter};
use std::path::Path;
use std::sync::{Arc, RwLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};

// 🌟 NEW: The Universal Action Trait!
// Any module (Math, TicTacToe, etc.) can implement this to talk to the brain.
pub trait IntoOpcode {
    fn into_opcode(self) -> i64;
}
// 🌟 NEW: The Universal Action Trait!
// Any module (Math, TicTacToe, etc.) can implement this to talk to the brain.


// 🌟 Tell the brain how to read basic numbers (like Tic-Tac-Toe spots!)
impl IntoOpcode for usize {
    fn into_opcode(self) -> i64 {
        self as i64
    }
}
// --- CONSTANTS ---
const A_PLUS: f64 = 0.1;
const A_MINUS: f64 = 0.12;
const TAU: f64 = 0.020;
const WINDOW_S: f64 = 0.1;
const NORMALIZATION_CAP: f64 = 5.0;
const REWARD_BOOST: f64 = 0.5;
const SURPRISE_LOW: f64 = 0.2;
const SURPRISE_HIGH: f64 = 0.8;

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

#[derive(Clone, Debug)]
pub struct PersistentMemory {
    pub weights: HashMap<Event, HashMap<Event, f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WeightEntry {
    from: Event,
    to: Event,
    weight: f64,
}

impl PersistentMemory {
    pub fn new() -> Self {
        Self { weights: HashMap::new() }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;
        let entries: Vec<WeightEntry> = self.weights.iter().flat_map(|(from, outgoing)| {
            outgoing.iter().map(|(to, weight)| WeightEntry {
                from: *from, to: *to, weight: *weight,
            })
        }).collect();
        serde_json::to_writer_pretty(BufWriter::new(file), &entries)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let entries: Vec<WeightEntry> = match serde_json::from_reader(BufReader::new(file)) {
            Ok(entries) => entries,
            Err(_) => return Ok(Self::new()),
        };
        let mut weights: HashMap<Event, HashMap<Event, f64>> = HashMap::new();
        for entry in entries {
            weights.entry(entry.from).or_insert_with(HashMap::new).insert(entry.to, entry.weight);
        }
        Ok(Self { weights })
    }
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
        let memory = Arc::new(RwLock::new(PersistentMemory::new()));
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

                let mut updates: Vec<(Event, Event, f64)> = Vec::new();
                let mut normalize_sources: HashSet<Event> = HashSet::new();

                let (ltp_multiplier, ltd_multiplier) = if surprise_gate <= SURPRISE_LOW {
                    (0.5, 2.0)
                } else if surprise_gate >= SURPRISE_HIGH {
                    (100.0, 0.1)
                } else {
                    (1.0, 1.0)
                };

                for (past_event, past_time) in recent_events.iter() {
                    let delta_t = current_time.duration_since(*past_time).as_secs_f64();
                    if delta_t <= 0.0 || delta_t >= WINDOW_S { continue; }

                    match current_event {
                        Event::Reward(intensity) => {
                            let scale = intensity as f64 / 100.0;
                            if scale > 0.0 {
                                let reward_change = (REWARD_BOOST * scale) * (-delta_t / TAU).exp();
                                updates.push((*past_event, current_event, reward_change));
                                normalize_sources.insert(*past_event);
                            }
                            continue;
                        }
                        Event::Error(_) => {
                            let penalty = -REWARD_BOOST * (-delta_t / TAU).exp();
                            updates.push((*past_event, current_event, penalty));
                            normalize_sources.insert(*past_event);
                            continue;
                        }
                        _ => {}
                    }

                    let ltp_change = A_PLUS * ltp_multiplier * (-delta_t / TAU).exp();
                    updates.push((*past_event, current_event, ltp_change));

                    let ltd_change = A_MINUS * ltd_multiplier * (-delta_t / TAU).exp();
                    updates.push((current_event, *past_event, -ltd_change));

                    normalize_sources.insert(*past_event);
                }

                if !updates.is_empty() {
                    if let Ok(mut mem) = mem_clone.write() {
                        for (from, to, delta) in updates {
                            let weight = mem.weights.entry(from).or_insert_with(HashMap::new).entry(to).or_insert(0.0);
                            *weight += delta;
                        }

                        for past_event in normalize_sources {
                            let mut sum = 0.0;
                            if let Some(outgoing) = mem.weights.get(&past_event) {
                                sum = outgoing.values().sum();
                            }

                            if sum > NORMALIZATION_CAP {
                                let factor = NORMALIZATION_CAP / sum;
                                if let Some(outgoing) = mem.weights.get_mut(&past_event) {
                                    for w in outgoing.values_mut() { *w *= factor; }
                                }
                            }
                        }
                    }
                }

                recent_events.push((current_event, current_time));
            };

            while let Ok(message) = rx.recv() {
                match message {
                    PlasticityMessage::Single(event, time) => { process_event(event, time, &mut recent_events); }
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
                    PlasticityMessage::Sync(reply_tx) => { let _ = reply_tx.send(()); }
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

    // 🌟 Restored for the English Chat Engine!
    pub fn best_next_event(&self, from: Event) -> Option<(Event, f64)> {
        let mem = self.memory.read().ok()?;
        let outgoing = mem.weights.get(&from)?;
        let mut total_weight = 0.0;
        let mut best: Option<(Event, f64)> = None;

        let filtered: Box<dyn Iterator<Item = (Event, f64)> + '_> = match from {
            Event::ContextWithState { state_hash, .. } => {
                Box::new(outgoing.iter().filter_map(move |(dst, weight)| {
                    if let Event::ContextWithState {
                        state_hash: dst_hash,
                        ..
                    } = dst
                    {
                        if *dst_hash == state_hash {
                            Some((*dst, *weight))
                        } else {
                            None
                        }
                    } else {
                        Some((*dst, *weight)) 
                    }
                }))
            }
            _ => Box::new(outgoing.iter().map(|(dst, weight)| (*dst, *weight))),
        };

        for (dst, weight) in filtered {
            total_weight += weight;
            let is_better = best.as_ref().map_or(true, |(_, w)| weight > *w);
            if is_better {
                best = Some((dst, weight));
            }
        }

        best.map(|(dst, weight)| {
            let confidence = if total_weight > 0.0 {
                (weight / total_weight).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (dst, confidence)
        })
    }
    // 🌟 NEW: This is now perfectly generic. It doesn't care what `<A>` is!
    pub fn get_op_distribution<A: Copy + IntoOpcode>(&self, context_event: Event, allowed_ops: &[A]) -> Vec<(A, f32)> {
        let mem = match self.memory.read() {
            Ok(guard) => guard,
            Err(_) => {
                return allowed_ops.iter().map(|&op| (op, 1.0 / allowed_ops.len() as f32)).collect();
            }
        };

        let mut priors = Vec::with_capacity(allowed_ops.len());
        let mut sum = 0.0;
        let temperature = 2.0;

        if let Some(outgoing) = mem.weights.get(&context_event) {
            for &op in allowed_ops {
                let target_op_id = op.into_opcode();
                let mut max_weight = 0.05;

                for (evt, w) in outgoing {
                    if let Event::Opcode { opcode, .. } = evt {
                        if *opcode == target_op_id {
                            if *w > max_weight { max_weight = *w; }
                        }
                    }
                }

                let score = max_weight.powf(temperature) as f32;
                priors.push((op, score));
                sum += score;
            }
        } else {
            let uniform = 1.0 / allowed_ops.len() as f32;
            for &op in allowed_ops { priors.push((op, uniform)); }
            return priors;
        }

        if sum > 0.0 {
            for (_, score) in &mut priors { *score /= sum; }
        } else {
            let uniform = 1.0 / allowed_ops.len() as f32;
            for (_, score) in &mut priors { *score = uniform; }
        }

        priors
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mem = self.memory.read().map_err(|_| io::Error::new(io::ErrorKind::Other, "lock poisoned"))?;
        mem.save_to_file(path)
    }

    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let loaded = PersistentMemory::load_from_file(path)?;
        let mut mem = self.memory.write().map_err(|_| io::Error::new(io::ErrorKind::Other, "lock poisoned"))?;
        *mem = loaded;
        Ok(())
    }
}
use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::collections::HashMap;
use crate::types::UVal;

// ─────────────────────────────────────────────────────────────────────────────
// 1. THE AGI BRAIN ARENA (1M Buckets × 16 Synapses Each = 16M Fixed Synapses)
// ─────────────────────────────────────────────────────────────────────────────
pub const ARENA_SIZE: usize = 1_000_000; 
pub const SYNAPSES_PER_NODE: usize = 16; 
pub const DEFAULT_MEMORY_PATH: &str = "src/memory/memory.bin";

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Synapse {
    pub target_hash: u64,
    pub weight: f32,      
    pub utility: f32,     
}

impl Default for Synapse {
    fn default() -> Self {
        Self { target_hash: 0, weight: 0.0, utility: 0.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NodeBucket {
    pub context_hash: u64,
    pub domain_tag: u64,
    pub synapses: [Synapse; SYNAPSES_PER_NODE],
}

impl Default for NodeBucket {
    fn default() -> Self {
        Self { 
            context_hash: 0, 
            domain_tag: 0, 
            synapses: [Synapse::default(); SYNAPSES_PER_NODE] 
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. PERSISTENT MEMORY (Memory-Mapped Zero-Copy Arena)
// ─────────────────────────────────────────────────────────────────────────────
pub struct PersistentMemory {
    _file: File,
    mmap: MmapMut,
    arena_size: usize,
}

impl PersistentMemory {
    /// Initialize memory-mapped arena
    pub fn new() -> io::Result<Self> {
        let path = Path::new(DEFAULT_MEMORY_PATH);
        
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open or create file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        // Calculate required size
        let needed_size = (ARENA_SIZE * std::mem::size_of::<NodeBucket>()) as u64;
        
        // Ensure file is exactly the right size
        let current_size = file.metadata().map(|m| m.len()).unwrap_or(0);
        if current_size != needed_size {
            file.set_len(needed_size)?;
        }

        // Memory-map the file with mutable access
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        println!("💾 Brain arena initialized: {} synapses mapped to disk", ARENA_SIZE);

        Ok(Self {
            _file: file,
            mmap,
            arena_size: ARENA_SIZE,
        })
    }

    /// Get read-only view of arena as slice
    #[inline]
    fn arena(&self) -> &[NodeBucket] {
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr() as *const NodeBucket,
                self.arena_size
            )
        }
    }

    /// Get mutable view of arena as slice
    #[inline]
    fn arena_mut(&mut self) -> &mut [NodeBucket] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.mmap.as_mut_ptr() as *mut NodeBucket,
                self.arena_size
            )
        }
    }

    #[inline(always)]
    pub fn get_bucket_index(context_hash: u64) -> usize {
        (context_hash % (ARENA_SIZE as u64)) as usize
    }

    /// Safe bucket lookup with linear probing collision detection
    pub fn get_bucket(&self, context_hash: u64) -> Option<&NodeBucket> {
        let arena = self.arena();
        let mut idx = Self::get_bucket_index(context_hash);
        let max_probes = 100;
        let mut probes = 0;

        loop {
            if probes >= max_probes { return None; }

            let bucket = &arena[idx];
            
            if bucket.context_hash == context_hash {
                return Some(bucket);
            }
            
            if bucket.context_hash == 0 {
                return None;
            }
            
            idx = (idx + 1) % ARENA_SIZE;
            probes += 1;
        }
    }

    /// Helper: Update synapses within a bucket
    fn update_synapses_in_bucket(bucket: &mut NodeBucket, target_hash: u64, delta: f32) {
        // 1. Try to find existing synapse
        for syn in bucket.synapses.iter_mut() {
            if syn.target_hash == target_hash {
                syn.weight += delta;
                if delta > 0.0 { syn.utility += 0.01; }
                return;
            }
        }

        // 2. Synapse doesn't exist - find weakest to evict
        let mut weakest_idx = 0;
        let mut lowest_score = f32::MAX;

        for (i, syn) in bucket.synapses.iter().enumerate() {
            if syn.target_hash == 0 {
                weakest_idx = i;
                break;
            }
            let score = syn.weight * syn.utility;
            if score < lowest_score {
                lowest_score = score;
                weakest_idx = i;
            }
        }

        // Insert new synapse (evicts weakest)
        // 🌟 Preserve negative weights (trauma) - only clamp positive weights up
        let weight = if delta < 0.0 {
            delta  // Trauma: keep the negative value
        } else {
            delta.max(0.01)  // Positive: floor at 0.01 for exploration
        };
        
        bucket.synapses[weakest_idx] = Synapse {
            target_hash,
            weight,
            utility: 0.1,
        };
    }

    /// Apply weight update with linear probing
    pub fn apply_update(&mut self, domain_tag: u64, context_hash: u64, target_hash: u64, delta: f32) {
        let arena = self.arena_mut();
        let mut idx = Self::get_bucket_index(context_hash);
        let max_probes = 100;
        let mut probes = 0;

        loop {
            if probes >= max_probes { return; }

            let bucket = &mut arena[idx];
            
            if bucket.context_hash == context_hash {
                Self::update_synapses_in_bucket(bucket, target_hash, delta);
                return;
            }
            
            if bucket.context_hash == 0 {
                bucket.context_hash = context_hash;
                bucket.domain_tag = domain_tag;
                bucket.synapses = [Synapse::default(); SYNAPSES_PER_NODE];
                Self::update_synapses_in_bucket(bucket, target_hash, delta);
                return;
            }
            
            idx = (idx + 1) % ARENA_SIZE;
            probes += 1;
        }
    }

    /// Flush mmap to disk
    pub fn flush(&mut self) -> io::Result<()> {
        self.mmap.flush()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. THE VM RAM (Execution Memory for Op::Load and Op::Store)
// ─────────────────────────────────────────────────────────────────────────────
const PRECISION_SCALE: f64 = 10_000_000_000.0;

#[derive(Debug, Clone)]
pub struct MemorySystem {
    storage: HashMap<i64, UVal>,
}

impl MemorySystem {
    pub fn new() -> Self {
        Self { storage: HashMap::new() }
    }

    #[inline]
    fn quantize(addr: f64) -> Option<i64> {
        if !addr.is_finite() { return None; }
        Some((addr * PRECISION_SCALE).round() as i64)
    }

    pub fn read(&self, addr: f64) -> Option<UVal> {
        let key = Self::quantize(addr)?;
        self.storage.get(&key).cloned()
    }

    pub fn write(&mut self, addr: f64, val: UVal) -> bool {
        let key = match Self::quantize(addr) {
            Some(k) => k,
            None => return false,
        };
        self.storage.insert(key, val);
        true
    }
}
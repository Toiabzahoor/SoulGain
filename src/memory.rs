use std::collections::HashMap;
use crate::types::UVal; // Import UVal so we can store complex types

/// Precision for floating point addresses (1e10).
const PRECISION_SCALE: f64 = 10_000_000_000.0;

#[derive(Debug, Clone)]
pub struct MemorySystem {
    // UPDATED: Now stores UVal instead of just f64
    storage: HashMap<i64, UVal>,
}

impl MemorySystem {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }

    #[inline]
    fn quantize(addr: f64) -> Option<i64> {
        if !addr.is_finite() {
            return None;
        }
        Some((addr * PRECISION_SCALE).round() as i64)
    }

    // UPDATED: Returns a UVal (cloned safely via Arc)
    pub fn read(&self, addr: f64) -> Option<UVal> {
        let key = Self::quantize(addr)?;
        self.storage.get(&key).cloned()
    }

    // UPDATED: Accepts any UVal to write to memory
    pub fn write(&mut self, addr: f64, val: UVal) -> bool {
        let key = match Self::quantize(addr) {
            Some(k) => k,
            None => return false,
        };
        self.storage.insert(key, val);
        true
    }
}
use serde::{Deserialize, Serialize};

pub mod skills; // Add this line
pub mod value;

pub use skills::SkillLibrary; // Add this line
pub use value::UVal;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct InternalState {
    pub entropy: f64,
    pub momentum: f64,
    pub coherence: f64,
    pub plasticity: f64,
}

impl InternalState {
    pub fn hash(&self) -> u64 {
        fn quantize(value: f64) -> u8 {
            (value.clamp(0.0, 1.0) * 255.0).round() as u8
        }

        let entropy = quantize(self.entropy) as u64;
        let momentum = quantize(self.momentum) as u64;
        let coherence = quantize(self.coherence) as u64;
        let plasticity = quantize(self.plasticity) as u64;

        entropy | (momentum << 8) | (coherence << 16) | (plasticity << 24)
    }
}

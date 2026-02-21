use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SkillLibrary {
    pub macros: HashMap<i64, Vec<f64>>,
}

impl SkillLibrary {
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
        }
    }

    pub fn define_skill(&mut self, id: i64, mut program: Vec<f64>) {
        // Strip HALT so skills can be piped together
        if program.last() == Some(&8.0) {
            program.pop();
        }
        self.macros.insert(id, program);
    }

    pub fn get_skill(&self, id: i64) -> Option<&Vec<f64>> {
        self.macros.get(&id)
    }
}
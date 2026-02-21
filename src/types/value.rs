use std::collections::HashMap;
use std::sync::Arc;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum UVal {
    Nil,
    Bool(bool),
    Number(f64),
    String(Arc<String>),
    Object(Arc<HashMap<String, UVal>>),
}

impl UVal {
    /// Helper to convert our types into a "truthy" boolean for logic ops
    pub fn is_truthy(&self) -> bool {
        match self {
            UVal::Nil => false,
            UVal::Bool(b) => *b,
            UVal::Number(n) => *n != 0.0 && !n.is_nan(),
            UVal::String(s) => !s.is_empty(),
            UVal::Object(_) => true, // Objects are always truthy
        }
    }
}

impl fmt::Display for UVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UVal::Nil => write!(f, "nil"),
            UVal::Bool(b) => write!(f, "{}", b),
            UVal::Number(n) => write!(f, "{}", n),
            UVal::String(s) => write!(f, "\"{}\"", s),
            UVal::Object(_) => write!(f, "[Object]"),
        }
    }
}
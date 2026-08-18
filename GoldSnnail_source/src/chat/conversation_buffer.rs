//! Conversation Buffer — Ring buffer for chat context
//!
//! Stores the last N conversation turns to provide context
//! for the SNN-LLM bridge.

use std::collections::VecDeque;

/// A single turn in a conversation.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    /// Role: "user", "assistant", "system"
    pub role: String,
    /// The text content
    pub text: String,
    /// Decoded tokens
    pub tokens: Vec<String>,
    /// Timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// Optional reward signal
    pub reward: Option<f64>,
}

impl ConversationTurn {
    pub fn new_user(text: String) -> Self {
        Self {
            role: "user".to_string(),
            text,
            tokens: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            reward: None,
        }
    }

    pub fn new_assistant(text: String, tokens: Vec<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            text,
            tokens,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            reward: None,
        }
    }
}

/// Ring buffer for conversation context.
#[derive(Debug, Clone)]
pub struct ConversationBuffer {
    /// Maximum number of turns to keep
    capacity: usize,
    /// The actual buffer
    buffer: VecDeque<ConversationTurn>,
}

impl ConversationBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    /// Add a new turn to the buffer.
    pub fn push(&mut self, turn: ConversationTurn) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(turn);
    }

    /// Get all turns as a vector of references.
    pub fn turns(&self) -> Vec<&ConversationTurn> {
        self.buffer.iter().collect()
    }

    /// Get the last N turns.
    pub fn last_n(&self, n: usize) -> Vec<&ConversationTurn> {
        let start = self.buffer.len().saturating_sub(n);
        self.buffer.iter().skip(start).collect()
    }

    /// Get the last user turn.
    pub fn last_user_turn(&self) -> Option<&ConversationTurn> {
        self.buffer.iter().rev().find(|t| t.role == "user")
    }

    /// Get the last assistant turn.
    pub fn last_assistant_turn(&self) -> Option<&ConversationTurn> {
        self.buffer.iter().rev().find(|t| t.role == "assistant")
    }

    /// Number of turns in buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Export all turns to JSON string.
    pub fn to_json(&self) -> String {
        let mut json = String::from("[\n");
        let turns_vec: Vec<_> = self.buffer.iter().collect();
        for (i, turn) in turns_vec.iter().enumerate() {
            json.push_str(&format!(
                "  {{\"role\": \"{}\", \"text\": \"{}\", \"tokens\": [{}], \"timestamp\": {}}}",
                turn.role,
                escape_json(&turn.text),
                turn.tokens.iter().map(|t| format!("\"{}\"", escape_json(t))).collect::<Vec<_>>().join(", "),
                turn.timestamp
            ));
            if i < turns_vec.len() - 1 {
                json.push_str(",\n");
            } else {
                json.push('\n');
            }
        }
        json.push_str("]\n");
        json
    }

    /// Import turns from JSON string.
    pub fn from_json(json: &str, capacity: usize) -> Result<Self, String> {
        let mut buffer = VecDeque::new();
        let mut current_role = String::new();
        let mut current_text = String::new();
        let mut current_tokens: Vec<String> = Vec::new();
        let mut current_timestamp: u64 = 0;

        for line in json.lines() {
            let trimmed = line.trim();
            
            let mut role = String::new();
            let mut text = String::new();
            let mut timestamp: u64 = 0;
            
            if trimmed.contains("\"role\"") {
                role = extract_json_string_value(trimmed, "role");
            }
            if trimmed.contains("\"text\"") {
                text = extract_json_string_value(trimmed, "text");
            }
            if trimmed.contains("\"timestamp\"") {
                if let Some(start) = trimmed.find(':') {
                    let num_str = trimmed[start + 1..].trim().trim_end_matches(',').trim_end_matches('}').to_string();
                    timestamp = num_str.parse().unwrap_or(0);
                }
            }
            
            if trimmed.contains('}') && !role.is_empty() {
                buffer.push_back(ConversationTurn {
                    role,
                    text,
                    tokens: Vec::new(),
                    timestamp,
                    reward: None,
                });
            }
        }

        let mut buf = Self::new(capacity);
        for turn in buffer {
            buf.push(turn);
        }
        Ok(buf)
    }
}

fn extract_json_string_value(line: &str, key: &str) -> String {
    let search = format!("\"{}\"", key);
    if let Some(pos) = line.find(&search) {
        let after_key = &line[pos + search.len()..];
        if let Some(colon_pos) = after_key.find(':') {
            let after_colon = &after_key[colon_pos + 1..].trim();
            if after_colon.starts_with('"') {
                let mut result = String::new();
                let mut escaped = false;
                for ch in after_colon[1..].chars() {
                    if escaped {
                        result.push(ch);
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == '"' {
                        break;
                    } else {
                        result.push(ch);
                    }
                }
                return result;
            }
        }
    }
    String::new()
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"', "\\\"")
     .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_overflow() {
        let mut buf = ConversationBuffer::new(3);
        for i in 0..5 {
            buf.push(ConversationTurn::new_user(format!("msg {}", i)));
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.turns()[0].text, "msg 2");
    }

    #[test]
    fn last_user_turn() {
        let mut buf = ConversationBuffer::new(10);
        buf.push(ConversationTurn::new_user("hello".to_string()));
        buf.push(ConversationTurn::new_assistant("hi".to_string(), vec!["hi".to_string()]));
        buf.push(ConversationTurn::new_user("how are you?".to_string()));
        
        let last_user = buf.last_user_turn().unwrap();
        assert_eq!(last_user.text, "how are you?");
    }

    #[test]
    fn json_roundtrip() {
        let mut buf = ConversationBuffer::new(10);
        buf.push(ConversationTurn::new_user("test \"quote\"".to_string()));
        
        let json = buf.to_json();
        let buf2 = ConversationBuffer::from_json(&json, 10).unwrap();
        assert_eq!(buf2.len(), 1);
        assert_eq!(buf2.turns()[0].text, "test \"quote\"");
    }

    #[test]
    fn json_roundtrip_multiple_turns() {
        let mut buf = ConversationBuffer::new(10);
        buf.push(ConversationTurn::new_user("Hello there".to_string()));
        buf.push(ConversationTurn::new_assistant("Hi!".to_string(), vec!["hi".to_string()]));
        buf.push(ConversationTurn::new_user("How are you?".to_string()));
        buf.push(ConversationTurn::new_assistant("I am good.".to_string(), vec!["i".to_string(), "am".to_string(), "good".to_string()]));
        buf.push(ConversationTurn::new_user("That is great.".to_string()));

        let json = buf.to_json();
        let imported = ConversationBuffer::from_json(&json, 10).unwrap();

        assert_eq!(imported.len(), 5);
        assert_eq!(imported.turns()[0].role, "user");
        assert_eq!(imported.turns()[0].text, "Hello there");
        assert_eq!(imported.turns()[1].role, "assistant");
        assert_eq!(imported.turns()[1].text, "Hi!");
        assert_eq!(imported.turns()[2].role, "user");
        assert_eq!(imported.turns()[2].text, "How are you?");
        assert_eq!(imported.turns()[3].role, "assistant");
        assert_eq!(imported.turns()[3].text, "I am good.");
        assert_eq!(imported.turns()[4].role, "user");
        assert_eq!(imported.turns()[4].text, "That is great.");
    }

    #[test]
    fn last_n_returns_recent() {
        let mut buf = ConversationBuffer::new(20);
        for i in 0..10 {
            buf.push(ConversationTurn::new_user(format!("msg {}", i)));
        }

        let last_3 = buf.last_n(3);
        assert_eq!(last_3.len(), 3);
        assert_eq!(last_3[0].text, "msg 7");
        assert_eq!(last_3[1].text, "msg 8");
        assert_eq!(last_3[2].text, "msg 9");
    }

    #[test]
    fn clear_empties_buffer() {
        let mut buf = ConversationBuffer::new(10);
        buf.push(ConversationTurn::new_user("one".to_string()));
        buf.push(ConversationTurn::new_assistant("two".to_string(), vec![]));
        buf.push(ConversationTurn::new_user("three".to_string()));

        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());

        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }
}

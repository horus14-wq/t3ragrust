//! Model identification: which HF repo + ONNX export variant + tokenizer.

/// Filename candidates inside an `onnx-community`-style HF export, in
/// preference order (uint8/fp16 per the locked decision; fp32 last resort).
pub const ONNX_FILE_CANDIDATES: [&str; 5] = [
    "onnx/model_uint8.onnx",
    "onnx/model_int8.onnx",
    "onnx/model_fp16.onnx",
    "onnx/model_q4f16.onnx",
    "onnx/model.onnx",
];

/// Everything needed to fetch and run one embedding model.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    /// HF repo id, e.g. `onnx-community/Qwen3-Embedding-0.6B-ONNX`.
    pub repo_id: String,
    /// Preferred ONNX file candidates (see [`ONNX_FILE_CANDIDATES`]).
    pub onnx_files: Vec<String>,
    /// Tokenizer file inside the repo (usually `tokenizer.json` at root).
    pub tokenizer_file: String,
    /// Expected embedding dimension (verified against model output).
    pub dims: usize,
    /// Instruction prefix prepended to queries (empty = none).
    pub instruction: String,
}

impl ModelSpec {
    /// Locked default: Qwen3-Embedding-0.6B, MTEB ~70.7, 32K context,
    /// CPU-friendly, ~1024 dims.
    pub fn qwen3_embedding_0_6b() -> Self {
        Self {
            repo_id: "onnx-community/Qwen3-Embedding-0.6B-ONNX".into(),
            onnx_files: ONNX_FILE_CANDIDATES.iter().map(|s| s.to_string()).collect(),
            tokenizer_file: "tokenizer.json".into(),
            dims: 1024,
            instruction: super::DEFAULT_QUERY_INSTRUCTION.into(),
        }
    }

    /// Generic spec for a HF repo assumed to follow the same layout
    /// (onnx/*.onnx + tokenizer.json). Dims discovered at runtime.
    pub fn for_hf_repo(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            onnx_files: ONNX_FILE_CANDIDATES.iter().map(|s| s.to_string()).collect(),
            tokenizer_file: "tokenizer.json".into(),
            dims: 0,
            instruction: super::DEFAULT_QUERY_INSTRUCTION.into(),
        }
    }
}

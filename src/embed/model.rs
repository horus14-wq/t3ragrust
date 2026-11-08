//! Model identification: which HF repo + ONNX export variant + tokenizer,
//! plus the per-model registry (t3-css.1.2): instruction prefixes and input
//! signatures differ per family — the previous `for_hf_repo` hardcoded the
//! Qwen3 instruction for *every* repo (silently wrong on BGE/e5/jina) and
//! unconditionally fed `position_ids` (hard crash on exports that don't
//! declare it). Registry lookups give each known repo the right behavior;
//! unknown repos get the honest generic defaults: no instruction, declared-
//! inputs-only feeding, dims discovered at runtime.

/// Filename candidates inside an `onnx-community`-style HF export, in
/// preference order (uint8/int8 preferred; fp16 last — a known ORT
/// per-thread memory blowup; fp32 exports carry external `_data` siblings).
pub const ONNX_FILE_CANDIDATES: [&str; 5] = [
    "onnx/model_uint8.onnx",
    "onnx/model_int8.onnx",
    "onnx/model_fp16.onnx",
    "onnx/model_q4f16.onnx",
    "onnx/model.onnx",
];

/// Root-level fp32 export candidates (e.g. `Qwen3-Embedding-4B-ONNX` ships
/// `model.onnx` + `model.onnx_data` at the repo ROOT, no `onnx/` dir).
pub const ONNX_ROOT_FILE_CANDIDATES: [&str; 1] = ["model.onnx"];

/// One registry row: known repo → retrieval behavior (public for `t3rag models`).
pub struct RegistryEntry {
    pub repo: &'static str,
    /// Query-side instruction prefix ("" when the model isn't instruct-tuned).
    pub instruction: &'static str,
    /// Passage-side prefix (e5 wants `passage: `; most want nothing).
    pub passage_instruction: &'static str,
    pub dims: usize,
    pub onnx_files: &'static [&'static str],
}

/// Known embedding models — each verified against its HF repo layout.
/// Unknown repos fall through to `for_hf_repo`'s generic spec.
pub const REGISTRY: &[RegistryEntry] = &[
    RegistryEntry {
        repo: "onnx-community/Qwen3-Embedding-0.6B-ONNX",
        instruction: super::DEFAULT_QUERY_INSTRUCTION,
        passage_instruction: "",
        dims: 1024,
        onnx_files: &ONNX_FILE_CANDIDATES,
    },
    // Bigger Qwen3 embedders ship fp32 external-data exports only (root
    // `model.onnx` + ~15GB `model.onnx_data`) — no quantized variants
    // exist anywhere. Workable via the sibling fetch (t3-css.1.1) but
    // heavy on CPU; `--max` therefore does NOT point here.
    RegistryEntry {
        repo: "onnx-community/Qwen3-Embedding-4B-ONNX",
        instruction: super::DEFAULT_QUERY_INSTRUCTION,
        passage_instruction: "",
        dims: 2560,
        onnx_files: &ONNX_ROOT_FILE_CANDIDATES,
    },
    RegistryEntry {
        repo: "onnx-community/Qwen3-Embedding-8B-ONNX",
        instruction: super::DEFAULT_QUERY_INSTRUCTION,
        passage_instruction: "",
        dims: 4096,
        onnx_files: &ONNX_ROOT_FILE_CANDIDATES,
    },
    // bge-m3: multilingual ~568M embedder; query prefix per model card.
    RegistryEntry {
        repo: "onnx-community/bge-m3-ONNX",
        instruction: "Represent this sentence for searching relevant passages: ",
        passage_instruction: "",
        dims: 1024,
        onnx_files: &ONNX_FILE_CANDIDATES,
    },
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
    /// Expected embedding dimension (verified against model output; 0 = unknown).
    pub dims: usize,
    /// Instruction prefix prepended to queries (empty = none).
    pub instruction: String,
    /// Instruction prefix prepended to passages (empty = none).
    pub passage_instruction: String,
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
            passage_instruction: String::new(),
        }
    }

    /// Resolve a repo through the registry when known; otherwise a generic
    /// spec with NO instruction (honest default — a foreign prefix silently
    /// degrades retrieval) and dims discovered at runtime.
    pub fn for_hf_repo(repo_id: &str) -> Self {
        if let Some(e) = REGISTRY.iter().find(|e| e.repo == repo_id) {
            return Self {
                repo_id: repo_id.to_string(),
                onnx_files: e.onnx_files.iter().map(|s| s.to_string()).collect(),
                tokenizer_file: "tokenizer.json".into(),
                dims: e.dims,
                instruction: e.instruction.to_string(),
                passage_instruction: e.passage_instruction.to_string(),
            };
        }
        Self {
            repo_id: repo_id.to_string(),
            onnx_files: ONNX_FILE_CANDIDATES.iter().map(|s| s.to_string()).collect(),
            tokenizer_file: "tokenizer.json".into(),
            dims: 0,
            instruction: String::new(),
            passage_instruction: String::new(),
        }
    }
}

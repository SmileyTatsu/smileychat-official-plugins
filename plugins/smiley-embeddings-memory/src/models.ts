/**
 * Local embedding model catalog and the text prefixes each model expects.
 *
 * Facts come from measured runs (see DESIGN.md §1) and the Hugging Face tree API;
 * sha256 values are the `lfs.oid` of each file and were cross-checked against the
 * downloaded files.
 */

export type PrefixProfile = "qwen3" | "gemma" | "none";

export type ModelProfileId = "default" | "strong";

export type LocalModelSpec = {
    id: ModelProfileId;
    label: string;
    description: string;
    repo: string;
    file: string;
    bytes: number;
    sha256: string;
    dim: number;
    pooling: "mean" | "last";
    prefixProfile: PrefixProfile;
    license: string;
    recommendedCtx: number;
};

export const LOCAL_MODELS: Record<ModelProfileId, LocalModelSpec> = {
    default: {
        id: "default",
        label: "EmbeddingGemma 2 (default, light)",
        description:
            "176 MB, runs on CPU at a usable speed. The safe pick for weak machines and phones.",
        repo: "unsloth/embeddinggemma-2-GGUF",
        file: "embeddinggemma-2-UD-Q4_K_XL.gguf",
        bytes: 175673856,
        sha256: "ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493",
        dim: 768,
        pooling: "mean",
        prefixProfile: "gemma",
        license: "apache-2.0",
        recommendedCtx: 4096,
    },
    strong: {
        id: "strong",
        label: "Qwen3 Embedding 4B (strong)",
        description:
            "2.5 GB, needs a capable GPU. Better retrieval; Q8_0 costs 1.7 GB more and does not retrieve better here.",
        repo: "Qwen/Qwen3-Embedding-4B-GGUF",
        file: "Qwen3-Embedding-4B-Q4_K_M.gguf",
        bytes: 2496703776,
        sha256: "2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b",
        dim: 2560,
        pooling: "last",
        prefixProfile: "qwen3",
        license: "apache-2.0",
        recommendedCtx: 4096,
    },
};

export const DEFAULT_PROFILE: ModelProfileId = "default";

export const DEFAULT_QWEN3_TASK =
    "Given a chat message, retrieve relevant past events from the same conversation";

/**
 * Query text for a profile. Qwen3 asks the instruction and the query to be separated
 * by a real newline with no space after `Query:`; Gemma uses its task prefix.
 *
 * Do NOT append `<|endoftext|>` for Qwen3: llama-server already adds the EOS token, and
 * a literal suffix produces a second one (cosine 0.990 against the correct vector).
 */
export function formatQuery(
    text: string,
    profile: PrefixProfile,
    task: string = DEFAULT_QWEN3_TASK,
): string {
    if (profile === "qwen3") return `Instruct: ${task}\nQuery:${text}`;
    if (profile === "gemma") return `task: search result | query: ${text}`;

    return text;
}

/** Document text for a profile: only Gemma wants a prefix on stored chunks. */
export function formatDocument(text: string, profile: PrefixProfile): string {
    if (profile === "gemma") return `title: none | text: ${text}`;

    return text;
}

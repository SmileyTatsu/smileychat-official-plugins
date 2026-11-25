import { describe, expect, test } from "bun:test";

import {
    DEFAULT_PROFILE,
    DEFAULT_QWEN3_TASK,
    LOCAL_MODELS,
    formatDocument,
    formatQuery,
} from "./models";

describe("model catalog", () => {
    test("has both profiles with matching ids", () => {
        expect(Object.keys(LOCAL_MODELS).sort()).toEqual(["default", "strong"]);
        expect(LOCAL_MODELS.default.id).toBe("default");
        expect(LOCAL_MODELS.strong.id).toBe("strong");
        expect(DEFAULT_PROFILE).toBe("default");
    });

    test("pins the measured file sizes and dimensions", () => {
        expect(LOCAL_MODELS.default.bytes).toBe(175673856);
        expect(LOCAL_MODELS.default.dim).toBe(768);
        expect(LOCAL_MODELS.default.pooling).toBe("mean");
        expect(LOCAL_MODELS.default.prefixProfile).toBe("gemma");
        expect(LOCAL_MODELS.default.repo).toBe("unsloth/embeddinggemma-2-GGUF");
        expect(LOCAL_MODELS.default.file).toBe("embeddinggemma-2-UD-Q4_K_XL.gguf");
        expect(LOCAL_MODELS.default.label).toBe("EmbeddingGemma 2 (default, light)");

        expect(LOCAL_MODELS.strong.bytes).toBe(2496703776);
        expect(LOCAL_MODELS.strong.dim).toBe(2560);
        expect(LOCAL_MODELS.strong.pooling).toBe("last");
        expect(LOCAL_MODELS.strong.prefixProfile).toBe("qwen3");
        expect(LOCAL_MODELS.strong.repo).toBe("Qwen/Qwen3-Embedding-4B-GGUF");
        expect(LOCAL_MODELS.strong.file).toBe("Qwen3-Embedding-4B-Q4_K_M.gguf");
        expect(LOCAL_MODELS.strong.label).toBe("Qwen3 Embedding 4B (strong)");
    });

    test("carries well-formed sha256 digests", () => {
        for (const spec of Object.values(LOCAL_MODELS)) {
            expect(spec.sha256).toMatch(/^[0-9a-f]{64}$/);
            expect(spec.license).toBe("apache-2.0");
            expect(spec.recommendedCtx).toBe(4096);
            expect(spec.file.endsWith(".gguf")).toBe(true);
            expect(spec.repo.split("/").length).toBe(2);
            expect(spec.description.length).toBeGreaterThan(20);
        }
        expect(LOCAL_MODELS.default.sha256).toBe(
            "ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493",
        );
        expect(LOCAL_MODELS.strong.sha256).toBe(
            "2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b",
        );
    });
});

describe("prefixes", () => {
    test("formats the qwen3 query with a real newline and no space after Query:", () => {
        expect(formatQuery("hello", "qwen3")).toBe(
            "Instruct: Given a chat message, retrieve relevant past events from the same conversation\nQuery:hello",
        );
        expect(DEFAULT_QWEN3_TASK).toBe(
            "Given a chat message, retrieve relevant past events from the same conversation",
        );
    });

    test("uses a custom qwen3 task when given one", () => {
        expect(formatQuery("hello", "qwen3", "find the cat")).toBe(
            "Instruct: find the cat\nQuery:hello",
        );
    });

    test("formats the gemma query and document prefixes", () => {
        expect(formatQuery("hello", "gemma")).toBe("task: search result | query: hello");
        expect(formatDocument("hello", "gemma")).toBe("title: none | text: hello");
    });

    test("leaves text unchanged for the none profile", () => {
        expect(formatQuery("hello", "none")).toBe("hello");
        expect(formatDocument("hello", "none")).toBe("hello");
    });

    test("never appends an EOS token to qwen3 documents", () => {
        // llama-server adds EOS itself; a literal suffix would add a second one.
        const document = formatDocument("hello", "qwen3");

        expect(document).toBe("hello");
        expect(document).not.toContain("<|endoftext|>");
        expect(formatQuery("hello", "qwen3")).not.toContain("<|endoftext|>");
    });

    test("keeps whitespace and unicode of the input text", () => {
        expect(formatQuery("  hi\nthere 😀 ", "gemma")).toBe(
            "task: search result | query:   hi\nthere 😀 ",
        );
    });
});

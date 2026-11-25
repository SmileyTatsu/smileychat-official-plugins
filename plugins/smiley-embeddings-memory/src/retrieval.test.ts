import { describe, expect, test } from "bun:test";

import type { ActiveModel, Embedder } from "./embedder";
import { EmbeddingError } from "./embedder";
import type { ChatIndex, IndexModelInfo } from "./index-store";
import { retrieveMemories } from "./retrieval";
import { normalizeSettings } from "./settings";
import { normalize } from "./vectors";

function unit(...values: number[]): Float32Array {
    return normalize(new Float32Array(values));
}

function makeIndex(
    chunks: { messageId: string; ord: number; text: string; vector: number[] }[],
    model: Partial<IndexModelInfo> = {},
): ChatIndex {
    return {
        chatId: "chat-1",
        model: { id: "local:test/model.gguf", source: "local", dim: 3, prefixProfile: "none", ...model },
        chunks: chunks.map((chunk) => ({
            id: `${chunk.messageId}:${chunk.ord}`,
            messageId: chunk.messageId,
            ord: chunk.ord,
            text: chunk.text,
            vector: unit(...chunk.vector),
        })),
        messages: {},
        updatedAt: "2026-01-01T00:00:00.000Z",
    };
}

function fakeEmbedder(
    queryVectors: Float32Array[],
    options: { model?: Partial<ActiveModel>; error?: Error } = {},
) {
    const state = { queries: [] as string[] };

    const embedder: Embedder = {
        model: { id: "local:test/model.gguf", source: "local", prefixProfile: "none", dim: 3, ...options.model },
        async embedDocuments() {
            return [];
        },
        async embedQueries(texts: string[]) {
            state.queries = texts;

            if (options.error) {
                throw options.error;
            }

            return texts.map((_text, index) => queryVectors[Math.min(index, queryVectors.length - 1)]);
        },
    };

    return { embedder, state };
}

const charTokens = (text: string) => text.length;

function settingsWith(overrides: Record<string, unknown> = {}) {
    return normalizeSettings({
        enabled: true,
        retain: 0,
        topK: 3,
        threshold: 0.3,
        queryMessages: 1,
        template: "M\n{{memories}}",
        maxMemoryTokens: 600,
        ...overrides,
    });
}

function messagesOf(...texts: string[]) {
    return texts.map((text, index) => ({ id: `m${index + 1}`, text }));
}

describe("retrieveMemories", () => {
    test("injects the best memories in chronological order", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] },
            { messageId: "m2", ord: 0, text: "beta", vector: [0.6, 0.8, 0] },
            { messageId: "m3", ord: 0, text: "gamma", vector: [0.8, 0.6, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three", "four", "five"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBeUndefined();
        expect(result.memories.map((memory) => memory.messageId)).toEqual(["m1", "m2", "m3"]);
        expect(result.memories.map((memory) => memory.score.toFixed(2))).toEqual(["1.00", "0.60", "0.80"]);
        // Distance from the newest message, not the array index.
        expect(result.memories.map((memory) => memory.messagesAgo)).toEqual([4, 3, 2]);
        expect(result.injectedText).toBe("M\n- alpha\n- beta\n- gamma");
        expect(result.tokens).toBe(result.injectedText.length);
        expect(result.candidates).toBe(3);
        expect(result.belowThreshold).toBe(0);
        expect(result.droppedByTokenCap).toBe(0);
        expect(result.queries).toEqual(["five"]);
        expect(result.line).toBe(`match: 3 memories (scores 1.00, 0.80, 0.60) -> ${result.injectedText.length} tokens`);
    });

    test("keeps only the top K scores", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] },
            { messageId: "m2", ord: 0, text: "beta", vector: [0.6, 0.8, 0] },
            { messageId: "m3", ord: 0, text: "gamma", vector: [0.8, 0.6, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three", "four", "five"),
            settings: settingsWith({ topK: 2 }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.memories.map((memory) => memory.messageId)).toEqual(["m1", "m3"]);
        expect(result.candidates).toBe(3);
    });

    test("skips the recent window before scoring and counts it", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 0, text: "old", vector: [1, 0, 0] },
            { messageId: "m2", ord: 0, text: "old but weak", vector: [0.1, 0, 0.99] },
            { messageId: "m5", ord: 0, text: "newest", vector: [1, 0, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three", "four", "five"),
            settings: settingsWith({ retain: 2 }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.skippedRecent).toBe(1);
        expect(result.candidates).toBe(2);
        expect(result.belowThreshold).toBe(1);
        expect(result.memories.map((memory) => memory.messageId)).toEqual(["m1"]);
    });

    test("reports all-inside-recent-window when nothing is eligible", async () => {
        const index = makeIndex([
            { messageId: "m4", ord: 0, text: "four", vector: [1, 0, 0] },
            { messageId: "m5", ord: 0, text: "five", vector: [1, 0, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three", "four", "five"),
            settings: settingsWith({ retain: 5 }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBe("all-in-recent-window");
        expect(result.skippedRecent).toBe(2);
        expect(result.memories).toEqual([]);
        expect(result.line).toBe("no match: all inside recent window");
    });

    test("reports the best score when everything is below the threshold", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 0, text: "alpha", vector: [0.1, 0.99, 0] },
            { messageId: "m2", ord: 0, text: "beta", vector: [0.2, 0.97, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBe("below-threshold");
        expect(result.memories).toEqual([]);
        expect(result.injectedText).toBe("");
        expect(result.belowThreshold).toBe(2);
        expect(result.line).toBe("no match: all below threshold (best 0.20)");
    });

    test("uses the model profile's calibrated threshold when the setting is auto", async () => {
        // 0.65 sits between Qwen3's 0.59 cut-off and Gemma's 0.76: unrelated text
        // scores that high for Gemma, so only one of the two may inject it.
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [0.65, 0.759934, 0] }]);
        const query = unit(1, 0, 0);

        const qwen = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith({ threshold: null }),
            embedder: fakeEmbedder([query], { model: { prefixProfile: "qwen3" } }).embedder,
            estimateTokens: charTokens,
        });

        expect(qwen.memories.length).toBe(1);

        const gemma = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith({ threshold: null }),
            embedder: fakeEmbedder([query], { model: { prefixProfile: "gemma" } }).embedder,
            estimateTokens: charTokens,
        });

        expect(gemma.reason).toBe("below-threshold");
        expect(gemma.detail).toContain("below 0.76");
    });

    test("an explicit threshold overrides the model's default", async () => {
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [0.65, 0.759934, 0] }]);
        const query = unit(1, 0, 0);

        const strict = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith({ threshold: 0.9 }),
            embedder: fakeEmbedder([query], { model: { prefixProfile: "qwen3" } }).embedder,
            estimateTokens: charTokens,
        });

        expect(strict.reason).toBe("below-threshold");

        const loose = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith({ threshold: 0.1 }),
            embedder: fakeEmbedder([query], { model: { prefixProfile: "gemma" } }).embedder,
            estimateTokens: charTokens,
        });

        expect(loose.memories.length).toBe(1);
    });

    test("merges adjacent chunks of one message and keeps distant ones apart", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 0, text: "first part", vector: [1, 0, 0] },
            { messageId: "m1", ord: 1, text: "second part", vector: [0.9, 0.1, 0] },
            { messageId: "m2", ord: 2, text: "far away", vector: [0.85, 0, 0.1] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.memories.length).toBe(2);
        expect(result.memories[0]).toMatchObject({
            chunkId: "m1:0",
            messageId: "m1",
            ord: 0,
            text: "first part second part",
            score: 1,
        });
        expect(result.memories[1].chunkId).toBe("m2:2");
        expect(result.injectedText).toBe("M\n- first part second part\n- far away");
    });

    test("orders memories by chat position, not by where the index appended them", async () => {
        // m1 was edited after m2 was indexed, so reconcile re-appended it at the
        // end of the index; the prompt must still read oldest message first.
        const index = makeIndex([
            { messageId: "m2", ord: 0, text: "second", vector: [1, 0, 0] },
            { messageId: "m1", ord: 0, text: "first", vector: [0.95, 0.1, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("first", "second", "third"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        // Scored by relevance, injected in chat order.
        expect(result.memories.map((memory) => memory.messageId)).toEqual(["m1", "m2"]);
        expect(result.injectedText).toBe("M\n- first\n- second");
    });

    test("merges consecutive ords that arrive shuffled in the index", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 1, text: "second part", vector: [0.9, 0.1, 0] },
            { messageId: "m1", ord: 0, text: "first part", vector: [1, 0, 0] },
            { messageId: "m1", ord: 3, text: "far tail", vector: [0.85, 0.1, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.memories.map((memory) => memory.ord)).toEqual([0, 3]);
        expect(result.memories[0].text).toBe("first part second part");
        expect(result.injectedText).toBe("M\n- first part second part\n- far tail");
    });

    test("keeps newlines in a memory but indents its continuation lines", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 0, text: "line one\nline two", vector: [1, 0, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.injectedText).toBe("M\n- line one\n  line two");
    });

    test("drops the lowest score first when the token cap is exceeded", async () => {
        const index = makeIndex([
            { messageId: "m1", ord: 0, text: "a".repeat(20), vector: [1, 0, 0] },
            { messageId: "m2", ord: 0, text: "b".repeat(20), vector: [0.5, 0.8660254, 0] },
            { messageId: "m3", ord: 0, text: "c".repeat(20), vector: [0.8, 0.6, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three"),
            // Three memories render to 68 chars, two to 47.
            settings: settingsWith({ maxMemoryTokens: 50 }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.droppedByTokenCap).toBe(1);
        expect(result.memories.map((memory) => memory.messageId)).toEqual(["m1", "m3"]);
        expect(result.tokens).toBe(result.injectedText.length);
        expect(result.tokens).toBeLessThanOrEqual(50);
    });

    test("reports token-cap when not even one memory fits", async () => {
        const longText = "x".repeat(120);
        const index = makeIndex([{ messageId: "m1", ord: 0, text: longText, vector: [1, 0, 0] }]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith({ maxMemoryTokens: 50 }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBe("token-cap");
        expect(result.memories).toEqual([]);
        expect(result.injectedText).toBe("");
        expect(result.tokens).toBe(0);
        expect(result.droppedByTokenCap).toBe(1);
        expect(result.line).toContain("no match: token cap exceeded");
    });

    test("reports a model mismatch with the index's identity in the detail", async () => {
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)], { model: { id: "api:host/other" } });

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBe("model-mismatch");
        expect(result.detail).toContain("index built with local:test/model.gguf");
        expect(result.detail).toContain("api:host/other");
        expect(result.line).toBe(`no match: ${result.detail}`);
    });

    test("detects a dimension mismatch the manifest cannot see", async () => {
        // The manifest check passes because the embedder's dimension is unknown.
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]);
        const { embedder } = fakeEmbedder([unit(1, 0)], { model: { dim: undefined } });

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBe("model-mismatch");
        expect(result.detail).toContain("2d vectors");
        expect(result.detail).toContain("3d");
    });

    test("reports an empty index for a missing or chunkless index", async () => {
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const missing = await retrieveMemories({
            index: null,
            messages: messagesOf("one"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(missing.reason).toBe("empty-index");
        expect(missing.line).toBe("no match: empty index");

        const chunkless = await retrieveMemories({
            index: makeIndex([]),
            messages: messagesOf("one"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(chunkless.reason).toBe("empty-index");
    });

    test("reports disabled before doing any work", async () => {
        const { embedder, state } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index: makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]),
            messages: messagesOf("one", "two"),
            settings: settingsWith({ enabled: false }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBe("disabled");
        expect(state.queries).toEqual([]);
        expect(result.line).toBe("no match: disabled");
    });

    test("uses the draft as the only query in a dry run", async () => {
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]);
        const { embedder, state } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
            draft: "  where is the sword?  ",
        });

        expect(state.queries).toEqual(["where is the sword?"]);
        expect(result.queries).toEqual(["where is the sword?"]);
        expect(result.memories.length).toBe(1);
    });

    test("reports no-queries when the chat has no text", async () => {
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("   ", "\n\n"),
            settings: settingsWith({ queryMessages: 2 }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.reason).toBe("no-queries");
        expect(result.line).toBe("no match: no queries");
    });

    test("queries only the last N non-empty messages, truncated", async () => {
        const long = "x".repeat(5000);
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]);
        const { embedder, state } = fakeEmbedder([unit(1, 0, 0)]);

        await retrieveMemories({
            index,
            messages: [{ id: "m1", text: "old" }, { id: "m2", text: long }, { id: "m3", text: "" }],
            settings: settingsWith({ queryMessages: 2 }),
            embedder,
            estimateTokens: charTokens,
        });

        expect(state.queries.length).toBe(2);
        expect(state.queries[0]).toBe("old");
        expect(state.queries[1].length).toBe(1500);
    });

    test("marks memories of messages that left the chat with -1", async () => {
        const index = makeIndex([
            { messageId: "gone", ord: 0, text: "deleted", vector: [1, 0, 0] },
            { messageId: "m3", ord: 0, text: "present", vector: [0.9, 0.1, 0] },
        ]);
        const { embedder } = fakeEmbedder([unit(1, 0, 0)]);

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two", "three"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
        });

        expect(result.memories.map((memory) => memory.messagesAgo)).toEqual([-1, 0]);
    });

    test("never throws when the embedder fails", async () => {
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]);

        const plain = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder: fakeEmbedder([], { error: new Error("boom") }).embedder,
            estimateTokens: charTokens,
        });

        expect(plain.reason).toBe("embedding-error");
        expect(plain.detail).toBe("boom");
        expect(plain.line).toBe("no match: embedding error: boom");

        const embedded = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder: fakeEmbedder([], { error: new EmbeddingError("unreachable", "Could not reach.", "Start it.") })
                .embedder,
            estimateTokens: charTokens,
        });

        expect(embedded.reason).toBe("embedding-error");
        expect(embedded.detail).toBe("Could not reach. Start it.");
    });

    test("reports an already aborted signal as an embedding error", async () => {
        const index = makeIndex([{ messageId: "m1", ord: 0, text: "alpha", vector: [1, 0, 0] }]);
        const { embedder, state } = fakeEmbedder([unit(1, 0, 0)]);
        const controller = new AbortController();
        controller.abort();

        const result = await retrieveMemories({
            index,
            messages: messagesOf("one", "two"),
            settings: settingsWith(),
            embedder,
            estimateTokens: charTokens,
            signal: controller.signal,
        });

        expect(result.reason).toBe("embedding-error");
        expect(result.detail).toBe("aborted");
        expect(state.queries).toEqual([]);
    });
});

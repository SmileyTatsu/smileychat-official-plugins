import { describe, expect, test } from "bun:test";

import { buildInjection, runWithTimeout } from "./injection";
import { emptyRetrievalResult } from "./retrieval";
import type { RetrievalResult } from "./retrieval";
import { normalizeSettings } from "./settings";

function resultWith(memories: number, injectedText = "M\n- one"): RetrievalResult {
    return {
        ...emptyRetrievalResult("below-threshold", "unused"),
        reason: undefined,
        memories: Array.from({ length: memories }, (_value, index) => ({
            chunkId: `m1:${index}`,
            messageId: "m1",
            ord: index,
            score: 0.9,
            text: `memory ${index}`,
            messagesAgo: 3,
        })),
        injectedText,
        tokens: 12,
        line: "match: 1 memory (scores 0.90) -> 12 tokens",
    };
}

describe("buildInjection", () => {
    test("returns null when there is nothing to inject", () => {
        expect(buildInjection(resultWith(0), normalizeSettings({}))).toBeNull();
        expect(buildInjection(resultWith(1, ""), normalizeSettings({}))).toBeNull();
    });

    test("shapes the injection the app expects", () => {
        const settings = normalizeSettings({ depth: 4 });

        const injection = buildInjection(resultWith(2), settings);

        expect(injection).toEqual({
            id: "smiley-embeddings-memory",
            source: "plugin",
            role: "system",
            content: "M\n- one",
            anchor: "at-depth",
            depth: 4,
            order: 0,
            priority: 10,
            tokenBudgetBehavior: "counted",
            metadata: { memories: 2 },
        });
    });

    test("uses the configured depth", () => {
        const injection = buildInjection(resultWith(1), normalizeSettings({ depth: 0 }));

        expect(injection?.depth).toBe(0);
    });
});

describe("runWithTimeout", () => {
    test("passes the task's result through", async () => {
        const expected = resultWith(1);

        const result = await runWithTimeout(async () => expected, 1000);

        expect(result).toBe(expected);
    });

    test("resolves with a timeout result and aborts the signal at the deadline", async () => {
        let aborted = false;

        const result = await runWithTimeout(async (signal) => {
            signal.addEventListener("abort", () => {
                aborted = true;
            });

            // Shorter than the test timeout: only the deadline can end this.
            await new Promise((resolve) => setTimeout(resolve, 200));

            return resultWith(1);
        }, 20);

        expect(result.reason).toBe("timeout");
        expect(result.memories).toEqual([]);
        expect(result.injectedText).toBe("");
        expect(result.tokens).toBe(0);
        expect(result.durationMs).toBe(0);
        expect(result.line).toBe("no match: timeout after 20 ms");
        expect(aborted).toBe(true);
    });

    test("never rejects when the task throws", async () => {
        const result = await runWithTimeout(async () => {
            throw new Error("retrieval exploded");
        }, 1000);

        expect(result.reason).toBe("embedding-error");
        expect(result.detail).toBe("retrieval exploded");
        expect(result.line).toBe("no match: embedding error: retrieval exploded");
    });
});

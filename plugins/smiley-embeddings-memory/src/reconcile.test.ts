import { describe, expect, test } from "bun:test";

import { hashText } from "./chunking";
import type { ChatIndex } from "./index-store";
import { createIndex } from "./index-store";
import type { ChatMessageLike } from "./reconcile";
import { planReconcile } from "./reconcile";

const model = {
    id: "embeddinggemma-test",
    source: "local" as const,
    dim: 4,
    prefixProfile: "gemma" as const,
};

/** An index whose entries are exactly the given texts, as a previous run left it. */
function indexWith(texts: Record<string, string>): ChatIndex {
    const index = createIndex("chat", model);

    for (const [messageId, text] of Object.entries(texts)) {
        index.messages[messageId] = { hash: hashText(text), chunks: 1 };
    }

    return index;
}

function message(id: string, text: string, streaming = false): ChatMessageLike {
    return { id, text, ...(streaming ? { streaming } : {}) };
}

describe("planReconcile", () => {
    test("a chat with no index embeds everything as new", () => {
        const plan = planReconcile(null, [message("m1", "Hello"), message("m2", "Hi there")]);

        expect(plan.toEmbed.map((entry) => entry.message.id)).toEqual(["m1", "m2"]);
        expect(plan.toEmbed.map((entry) => entry.reason)).toEqual(["new", "new"]);
        expect(plan.toEmbed[0].hash).toBe(hashText("Hello"));
        expect(plan.toDrop).toEqual([]);
        expect(plan.unchanged).toBe(0);
    });

    test("leaves matching hashes alone and embeds only new messages", () => {
        const plan = planReconcile(indexWith({ m1: "Hello" }), [
            message("m1", "Hello"),
            message("m2", "Something new"),
        ]);

        expect(plan.unchanged).toBe(1);
        expect(plan.toEmbed).toHaveLength(1);
        expect(plan.toEmbed[0].message.id).toBe("m2");
        expect(plan.toEmbed[0].reason).toBe("new");
        expect(plan.toDrop).toEqual([]);
    });

    test("an edited message is dropped and re-embedded in the same plan", () => {
        const plan = planReconcile(indexWith({ m1: "Hello" }), [message("m1", "Hello, edited")]);

        expect(plan.toDrop).toEqual(["m1"]);
        expect(plan.toEmbed).toHaveLength(1);
        expect(plan.toEmbed[0].reason).toBe("edited");
        expect(plan.toEmbed[0].hash).toBe(hashText("Hello, edited"));
        expect(plan.unchanged).toBe(0);
    });

    test("a swipe change looks like an edit: same id, different text", () => {
        // Swipes are not visible here by design: the caller passes the active
        // swipe text, so a swipe whose content differs is just a hash mismatch.
        const plan = planReconcile(indexWith({ m1: "First swipe" }), [message("m1", "Second swipe")]);

        expect(plan.toDrop).toEqual(["m1"]);
        expect(plan.toEmbed.map((entry) => entry.reason)).toEqual(["edited"]);
    });

    test("drops entries for messages that left the chat", () => {
        const plan = planReconcile(indexWith({ m1: "Kept", m2: "Deleted", m3: "Regenerated" }), [
            message("m1", "Kept"),
            message("m4", "Its replacement"),
        ]);

        expect(plan.toDrop.sort()).toEqual(["m2", "m3"]);
        expect(plan.unchanged).toBe(1);
        expect(plan.toEmbed.map((entry) => entry.message.id)).toEqual(["m4"]);
    });

    test("skips streaming messages and never drops them", () => {
        // The stored hash belongs to the finished text of a previous generation,
        // which is still the best memory available until the new one completes.
        const plan = planReconcile(indexWith({ m1: "Done text" }), [
            message("m1", "Still arriv", true),
            message("m2", "Also streaming", true),
        ]);

        expect(plan.skippedStreaming).toBe(2);
        expect(plan.toDrop).toEqual([]);
        expect(plan.toEmbed).toEqual([]);
        expect(plan.unchanged).toBe(0);
        expect(plan.emptyMessages).toBe(0);
    });

    test("counts empty messages without embedding or dropping them", () => {
        const plan = planReconcile(indexWith({}), [message("m1", "")]);

        expect(plan.emptyMessages).toBe(1);
        expect(plan.toEmbed).toEqual([]);
        expect(plan.toDrop).toEqual([]);
    });

    test("an edit down to nothing drops the old chunks but embeds nothing", () => {
        const plan = planReconcile(indexWith({ m1: "Was here" }), [message("m1", "")]);

        expect(plan.emptyMessages).toBe(1);
        expect(plan.toDrop).toEqual(["m1"]);
        expect(plan.toEmbed).toEqual([]);
    });

    test("whitespace-only text counts as empty", () => {
        // Cleaned text strips markers, and a message left with just spacing has
        // nothing to retrieve; embedding it would store a meaningless vector.
        const plan = planReconcile(indexWith({ m1: "Was here" }), [message("m1", " \n\t ")]);

        expect(plan.emptyMessages).toBe(1);
        expect(plan.toDrop).toEqual(["m1"]);
    });

    test("handles a mixed chat in one pass", () => {
        const plan = planReconcile(
            indexWith({ keep: "same", edit: "before", gone: "vanished", empty: "had text" }),
            [
                message("keep", "same"),
                message("edit", "after"),
                message("empty", ""),
                message("new", "brand new"),
                message("stream", "partial", true),
            ],
        );

        expect(plan.unchanged).toBe(1);
        expect(plan.skippedStreaming).toBe(1);
        expect(plan.emptyMessages).toBe(1);
        expect(plan.toDrop.sort()).toEqual(["edit", "empty", "gone"]);
        expect(plan.toEmbed.map((entry) => entry.message.id).sort()).toEqual(["edit", "new"]);
        expect(plan.toEmbed.every((entry) => entry.hash === hashText(entry.message.text))).toBe(true);
    });

    test("an index entry with zero chunks but a matching hash is unchanged", () => {
        const index = indexWith({ m1: "emitted nothing" });

        index.messages.m1.chunks = 0;

        const plan = planReconcile(index, [message("m1", "emitted nothing")]);

        expect(plan.unchanged).toBe(1);
        expect(plan.toEmbed).toEqual([]);
        expect(plan.toDrop).toEqual([]);
    });
});

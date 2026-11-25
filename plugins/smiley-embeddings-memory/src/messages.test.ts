import { describe, expect, test } from "bun:test";

import type { Message, MessageSwipe } from "../types/smileychat-plugin-api";

import { activeSwipeContent, chatMessagesForIndex, isIndexableMessage, isStreamingMessage, retrievalMessages } from "./messages";

function swipe(content: string, extra: Record<string, unknown> = {}): MessageSwipe {
    // The spread comes first so an extra field the published typings do not
    // know yet (a streaming flag) can be simulated without widening the type.
    return { ...extra, id: `swipe-${content}`, content, createdAt: "2026-01-01T00:00:00.000Z" };
}

function message(content: string, overrides: Partial<Message> = {}, swipes?: MessageSwipe[]): Message {
    return {
        id: "m1",
        author: "Aria",
        role: "character",
        createdAt: "2026-01-01T00:00:00.000Z",
        activeSwipeIndex: 0,
        swipes: swipes ?? [swipe(content)],
        ...overrides,
    };
}

describe("activeSwipeContent", () => {
    test("reads the visible swipe, not the first one", () => {
        const target = message("", { activeSwipeIndex: 1 }, [swipe("first"), swipe("second")]);

        expect(activeSwipeContent(target)).toBe("second");
    });

    test("is an empty string when the swipe index points nowhere", () => {
        // A message whose swipes were cleared, which the app allows for a
        // freshly created placeholder.
        expect(activeSwipeContent(message("x", { activeSwipeIndex: 3 }))).toBe("");
    });
});

describe("isStreamingMessage", () => {
    test("is false for a finished message", () => {
        expect(isStreamingMessage(message("done"))).toBe(false);
    });

    test("reads a flag on the message when a future build adds one", () => {
        const target = message("typing") as Message & { streaming?: boolean };
        target.streaming = true;

        expect(isStreamingMessage(target)).toBe(true);
    });

    test("reads a flag on the active swipe", () => {
        const target = message("", {}, [swipe("typing", { streaming: true })]);

        expect(isStreamingMessage(target)).toBe(true);
    });
});

describe("isIndexableMessage", () => {
    test("keeps user and character messages", () => {
        expect(isIndexableMessage(message("hi"))).toBe(true);
        expect(isIndexableMessage(message("hi", { role: "user" }))).toBe(true);
    });

    test("drops system messages", () => {
        expect(isIndexableMessage(message("note", { role: "system" }))).toBe(false);
    });

    test("drops messages the user kept out of the prompt or marked as system display", () => {
        expect(isIndexableMessage(message("hidden", { metadata: { includeInPrompt: false } }))).toBe(false);
        expect(isIndexableMessage(message("hidden", { metadata: { promptRole: "none" } }))).toBe(false);
        expect(isIndexableMessage(message("hidden", { metadata: { displayRole: "system" } }))).toBe(false);
    });
});

describe("chatMessagesForIndex", () => {
    test("uses the active swipe and cleans bubbles and image markers", () => {
        const target = message(
            "",
            { activeSwipeIndex: 1 },
            [swipe("ignored"), swipe('<msg>Hey!</msg><msg delay="1s">Look {{photo[0]}}</msg>')],
        );

        expect(chatMessagesForIndex([target])).toEqual([
            { id: "m1", text: "Hey!\n\nLook" },
        ]);
    });

    test("keeps chat order and skips system messages", () => {
        const messages = [
            message("First", { id: "m1", role: "user" }),
            message("Narrator note", { id: "m2", role: "system" }),
            message("Answer", { id: "m3" }),
        ];

        expect(chatMessagesForIndex(messages)).toEqual([
            { id: "m1", text: "First" },
            { id: "m3", text: "Answer" },
        ]);
    });

    test("marks streaming messages and leaves the flag off otherwise", () => {
        const typing = message("typing") as Message & { streaming?: boolean };
        typing.streaming = true;

        expect(chatMessagesForIndex([typing, message("done", { id: "m2" })])).toEqual([
            { id: "m1", text: "typing", streaming: true },
            { id: "m2", text: "done" },
        ]);
    });

    test("keeps a message whose text cleaned down to nothing so its chunks can be dropped", () => {
        expect(chatMessagesForIndex([message("{{photo[0]}}")])).toEqual([{ id: "m1", text: "" }]);
    });

    test("accepts the app's own content reader", () => {
        const content = () => "from api.messages.content";

        expect(chatMessagesForIndex([message("ignored")], { content })).toEqual([
            { id: "m1", text: "from api.messages.content" },
        ]);
    });
});

describe("retrievalMessages", () => {
    test("has exactly the id and text the retrieval reads", () => {
        const [first] = retrievalMessages([message("<msg>Hi</msg>")]);

        expect(first).toEqual({ id: "m1", text: "Hi" });
        expect(Object.keys(first)).toEqual(["id", "text"]);
    });
});

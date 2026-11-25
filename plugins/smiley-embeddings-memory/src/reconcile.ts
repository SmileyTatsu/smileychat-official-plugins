import { hashText } from "./chunking";
import type { ChatIndex } from "./index-store";

/** A chat message reduced to what the planner needs: cleaned text already. */
export type ChatMessageLike = {
    id: string;
    text: string;
    streaming?: boolean;
};

export type ReconcilePlan = {
    toEmbed: { message: ChatMessageLike; hash: string; reason: "new" | "edited" }[];
    toDrop: string[];
    unchanged: number;
    skippedStreaming: number;
    emptyMessages: number;
};

/**
 * Diff of a chat against its index.
 *
 * Pure on purpose: the indexer calls it from a chat switch, a debounce after
 * message updates and at the start of every retrieval, and none of those places
 * should touch storage to find out what is stale.
 */
export function planReconcile(index: ChatIndex | null, messages: ChatMessageLike[]): ReconcilePlan {
    const plan: ReconcilePlan = {
        toEmbed: [],
        toDrop: [],
        unchanged: 0,
        skippedStreaming: 0,
        emptyMessages: 0,
    };

    const entries = index?.messages ?? {};
    const present = new Set<string>();

    for (const message of messages) {
        present.add(message.id);

        if (message.streaming) {
            // The text is still arriving; embedding it would store a hash that
            // never matches the finished message. Its old chunks stay usable.
            plan.skippedStreaming += 1;
            continue;
        }

        const hasEntry = Object.prototype.hasOwnProperty.call(entries, message.id);

        if (message.text.trim().length === 0) {
            plan.emptyMessages += 1;

            if (hasEntry) {
                // Edited down to nothing: the stored chunks describe text that
                // no longer exists, so they are dropped rather than kept stale.
                plan.toDrop.push(message.id);
            }

            continue;
        }

        const hash = hashText(message.text);

        if (!hasEntry) {
            plan.toEmbed.push({ message, hash, reason: "new" });
            continue;
        }

        if (entries[message.id].hash === hash) {
            plan.unchanged += 1;
            continue;
        }

        // Edit, swipe change or regeneration: same message id, different text,
        // so the old chunks go and the new text is embedded in the same run.
        plan.toDrop.push(message.id);
        plan.toEmbed.push({ message, hash, reason: "edited" });
    }

    for (const messageId of Object.keys(entries)) {
        // Absent from the chat at all (deleted, or replaced by a regeneration).
        // Streaming and emptied messages were seen above, so they never land here.
        if (!present.has(messageId)) {
            plan.toDrop.push(messageId);
        }
    }

    return plan;
}

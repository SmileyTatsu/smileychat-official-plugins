/**
 * Turns the app's chat messages into the flat `{ id, text, streaming }` list the
 * reconciler and the injector work on.
 *
 * Pure on purpose: the indexer and the retrieval path both need exactly the
 * same view of the conversation (the visible swipe, cleaned), and a divergence
 * between them would make every reconcile re-embed messages that never changed.
 */

import type { Message } from "../types/smileychat-plugin-api";

import { cleanMessageText, hashText } from "./chunking";
import type { ChatMessageLike } from "./reconcile";

export type MessageReaders = {
    /** Visible text of a message; defaults to the active swipe's content. */
    content?: (message: Message) => string;
    /** Streaming probe; defaults to `isStreamingMessage`. */
    streaming?: (message: Message) => boolean;
};

/**
 * Text of the swipe the user is looking at.
 *
 * `api.messages.content` does the same thing, but this module is called from
 * pure helpers and from the indexer's injected `getChat`, neither of which has
 * an API handle.
 */
export function activeSwipeContent(message: Message): string {
    return message.swipes[message.activeSwipeIndex]?.content ?? "";
}

/**
 * Whether a message is still being generated.
 *
 * The published snapshot has no streaming marker: `MessageSwipe.status` is
 * `"error"` only. The flag is read if a future app build adds it, and
 * `index.tsx` additionally ignores snapshot updates that do not change the
 * message list (a streaming turn only rewrites the last swipe's text).
 */
export function isStreamingMessage(message: Message): boolean {
    const onMessage = (message as { streaming?: unknown }).streaming;

    if (onMessage === true) {
        return true;
    }

    const swipe = message.swipes[message.activeSwipeIndex] as { streaming?: unknown } | undefined;

    return swipe?.streaming === true;
}

/**
 * Messages worth remembering: what the model actually saw.
 *
 * System notes and tool traffic are not part of the story, and a message the
 * user excluded from the prompt (`includeInPrompt: false`, `promptRole: none`)
 * must not come back as an injected memory either.
 */
export function isIndexableMessage(message: Message): boolean {
    if (message.role !== "user" && message.role !== "character") {
        return false;
    }

    const metadata = message.metadata;

    if (!metadata) {
        return true;
    }

    if (metadata.includeInPrompt === false || metadata.promptRole === "none") {
        return false;
    }

    return metadata.displayRole !== "system";
}

/** Cleaned text per indexable message, in chat order. */
export function chatMessagesForIndex(messages: Message[], readers: MessageReaders = {}): ChatMessageLike[] {
    const content = readers.content ?? activeSwipeContent;
    const streaming = readers.streaming ?? isStreamingMessage;
    const out: ChatMessageLike[] = [];

    for (const message of messages) {
        if (!isIndexableMessage(message)) {
            continue;
        }

        const text = cleanMessageText(content(message));

        // The flag is only set when true so the common entry stays two fields,
        // which keeps hash comparisons in the reconciler honest.
        out.push(streaming(message) ? { id: message.id, text, streaming: true } : { id: message.id, text });
    }

    return out;
}

/**
 * The shape `retrieveMemories` and the recent-window logic read.
 *
 * `streaming` is carried through because retrieval must not treat a turn that is
 * still generating as an edit: its text changes every token while the index
 * holds the previous, complete version of the same message.
 */
export function retrievalMessages(
    messages: Message[],
    readers: MessageReaders = {},
): { id: string; text: string; streaming?: boolean }[] {
    return chatMessagesForIndex(messages, readers).map((message) =>
        message.streaming
            ? { id: message.id, text: message.text, streaming: true }
            : { id: message.id, text: message.text },
    );
}

/** How many trailing messages the scheduling fingerprint looks at. */
const FINGERPRINT_WINDOW = 3;

/**
 * Identity of the *visible* conversation for the state subscription.
 *
 * Message ids alone are not enough: generating or switching a swipe appends to
 * `swipes` and moves `activeSwipeIndex` while the id stays put, so an ids-only
 * key never noticed the text the index should now hold. The active content is
 * hashed as well, which a streaming turn changes on every token; the indexer's
 * debounce is what turns that into one run after the text settles.
 *
 * Only the tail is fingerprinted: an older message is edited through the app's
 * update middleware, which schedules a run on its own.
 */
export function messagesFingerprint(messages: Message[], window = FINGERPRINT_WINDOW): string {
    const start = Math.max(0, messages.length - Math.max(1, window));
    const parts: string[] = [];

    for (let i = start; i < messages.length; i += 1) {
        const message = messages[i];

        parts.push(
            `${message.id}:${message.activeSwipeIndex}:${message.swipes.length}:${hashText(activeSwipeContent(message))}`,
        );
    }

    return parts.join("|");
}

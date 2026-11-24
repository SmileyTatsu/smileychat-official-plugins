import type { Message } from "../types/smileychat-plugin-api";

/**
 * Text of the visible swipe.
 *
 * Local on purpose: the pipeline helpers that need it are pure functions with no
 * API handle, and this plugin's tests exercise them directly.
 */
export function getMessageContent(message: Message) {
    return message.swipes[message.activeSwipeIndex]?.content ?? "";
}

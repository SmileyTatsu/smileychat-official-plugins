import type { Message } from "../types/smileychat-plugin-api";

/**
 * Text of the visible swipe.
 *
 * Kept local instead of going through `api.messages.content` because the helpers
 * in this plugin that need it (eligibility and transcript formatting) are pure
 * functions without an API handle, and threading one through them only to read
 * two fields would make them harder to test.
 */
export function getMessageContent(message: Message) {
    return message.swipes[message.activeSwipeIndex]?.content ?? "";
}

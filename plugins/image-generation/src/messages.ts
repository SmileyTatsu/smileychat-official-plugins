import type { Message } from "../types/smileychat-plugin-api";

/** Text of the visible swipe. */
export function getMessageContent(message: Message) {
    return message.swipes[message.activeSwipeIndex]?.content ?? "";
}

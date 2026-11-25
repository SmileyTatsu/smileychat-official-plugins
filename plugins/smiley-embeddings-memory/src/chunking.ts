/**
 * Text chunking, message cleanup and content hashing for the memory index.
 *
 * Everything here is pure and browser-safe: no Node APIs, no dependencies.
 */

export type ChunkOptions = { size: number; overlap: number };

export type TextChunk = { text: string; start: number };

const fallbackSize = 800;

// A separator this early in the window would leave a useless fragment behind, so we
// would rather cut at the next-best separator (or hard-cut) than honour it.
const minCutRatio = 0.25;

function isWhitespace(char: string) {
    return (
        char === " " ||
        char === "\n" ||
        char === "\r" ||
        char === "\t" ||
        char === "\f" ||
        char === "\v"
    );
}

function normalizeSize(size: number) {
    if (!Number.isFinite(size)) return fallbackSize;
    return Math.max(1, Math.floor(size));
}

function normalizeOverlap(overlap: number, size: number) {
    if (!Number.isFinite(overlap)) return 0;
    return Math.min(Math.max(0, Math.floor(overlap)), size - 1);
}

/**
 * Nudges a cut left so an emoji (surrogate pair) is never split across two chunks.
 * Cuts at index < 2 are left alone: shrinking them would make no progress.
 */
function alignToCodePoint(text: string, index: number) {
    if (index < 2 || index >= text.length) return index;
    const before = text.charCodeAt(index - 1);
    const after = text.charCodeAt(index);
    const splitsPair =
        before >= 0xd800 && before <= 0xdbff && after >= 0xdc00 && after <= 0xdfff;

    return splitsPair ? index - 1 : index;
}

/** Returns the exclusive end offset of the chunk that starts at `cursor`. */
function findCut(text: string, cursor: number, size: number) {
    if (cursor + size >= text.length) return text.length;

    const window = text.slice(cursor, cursor + size);
    const minCut = Math.max(1, Math.floor(size * minCutRatio));
    let cut = 0;

    // Paragraph breaks keep whole thoughts together, line breaks keep whole lines,
    // spaces keep whole words; a hard cut is the last resort.
    const paragraph = window.lastIndexOf("\n\n");
    const line = window.lastIndexOf("\n");
    const space = window.lastIndexOf(" ");

    if (paragraph >= minCut) {
        cut = paragraph + 2;
    } else if (line >= minCut) {
        cut = line + 1;
    } else if (space >= minCut) {
        cut = space + 1;
    }

    const cutAt = cursor + (cut > 0 ? cut : size);
    // A size of 1 cannot honour both the bound and the pair, and progress wins:
    // without it a lone emoji at the boundary would loop forever.
    return Math.max(alignToCodePoint(text, cutAt), cursor + 1);
}

/**
 * Where the next chunk starts. `overlap` characters of the previous chunk are
 * repeated, snapped forward to the next word start so a word is never cut in half;
 * if no word boundary fits, the overlap is dropped so the loop always progresses.
 */
function findNextStart(text: string, cursor: number, end: number, overlap: number) {
    if (overlap <= 0) return end;

    let next = end - overlap;
    if (next <= cursor) return end;

    while (next < end && !isWhitespace(text[next - 1])) next += 1;
    while (next < end && isWhitespace(text[next])) next += 1;

    return next > cursor ? next : end;
}

function trimmedStart(raw: string) {
    let index = 0;
    while (index < raw.length && isWhitespace(raw[index])) index += 1;
    return index;
}

/**
 * Splits `text` into chunks of at most `size` characters with `overlap` characters
 * repeated at the start of the next chunk. `start` is the offset of the chunk's
 * first character in the input.
 */
export function chunkText(text: string, opts: ChunkOptions): TextChunk[] {
    const size = normalizeSize(opts.size);
    const overlap = normalizeOverlap(opts.overlap, size);

    if (!text) return [];

    if (text.length <= size) {
        const trimmed = text.trim();
        return trimmed ? [{ text: trimmed, start: trimmedStart(text) }] : [];
    }

    const chunks: TextChunk[] = [];
    let cursor = 0;

    while (cursor < text.length) {
        const end = findCut(text, cursor, size);
        const raw = text.slice(cursor, end);
        const trimmed = raw.trim();

        if (trimmed) {
            chunks.push({ text: trimmed, start: cursor + trimmedStart(raw) });
        }

        if (end >= text.length) break;
        // `end` is always > cursor, so the loop terminates even on unbroken strings.
        cursor = findNextStart(text, cursor, end, overlap);
    }

    return chunks;
}

const messageTagPattern = /<msg\b[^>]*>([\s\S]*?)<\/msg\s*>/gi;
const openingTagPattern = /<msg\b[^>]*>/gi;
const closingTagPattern = /<\/msg\s*>/gi;
const photoMarkerPattern = /\{\{\s*photo(?:\s*\[\s*\d+\s*\]|\s*:\s*id=[^}]*)?\s*\}\}/gi;

/**
 * Turns a raw chat message into plain indexable text: `<msg>` bubbles become their
 * content separated by a blank line, image markers are dropped. Tolerant by design:
 * half-streamed or malformed tags are stripped rather than thrown on.
 */
export function cleanMessageText(raw: string): string {
    if (!raw) return "";

    const withBubbles = raw.replace(
        messageTagPattern,
        (_match, content: string) => `\n\n${content}\n\n`,
    );

    return withBubbles
        .replace(openingTagPattern, "")
        .replace(closingTagPattern, "")
        .replace(photoMarkerPattern, "")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
}

/**
 * cyrb53: fast, stable, non-cryptographic 53-bit hash used only to detect changed
 * messages (edits and swipes), so collisions are not a security concern.
 */
export function hashText(text: string): string {
    let h1 = 0xdeadbeef;
    let h2 = 0x41c6ce57;

    for (let i = 0; i < text.length; i += 1) {
        const code = text.charCodeAt(i);
        h1 = Math.imul(h1 ^ code, 2654435761);
        h2 = Math.imul(h2 ^ code, 1597334677);
    }

    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);

    return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16).padStart(14, "0");
}

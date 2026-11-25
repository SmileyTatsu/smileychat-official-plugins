/**
 * Retrieval: pick the old chat fragments most similar to what is being said now.
 *
 * Pure apart from the embedder it is handed: no storage, no plugin API, no
 * logger. That keeps the whole decision path testable with hand-built vectors,
 * and lets the caller decide how to log or display the result line.
 */

import type { Embedder } from "./embedder";
import { EmbeddingError } from "./embedder";
import { modelMatches } from "./index-store";
import type { ChatIndex, IndexedChunk } from "./index-store";
import type { PluginSettings } from "./settings";
import { effectiveThreshold, MEMORIES_PLACEHOLDER } from "./settings";
import { rankVectors } from "./vectors";

export type NoMatchReason =
    | "disabled"
    | "empty-index"
    | "model-mismatch"
    | "no-queries"
    | "all-in-recent-window"
    | "below-threshold"
    | "token-cap"
    | "timeout"
    | "embedding-error";

export type Memory = {
    chunkId: string;
    messageId: string;
    ord: number;
    score: number;
    text: string;
    /** Distance from the newest message (0 = newest); -1 when it left the chat. */
    messagesAgo: number;
};

export type RetrievalInput = {
    index: ChatIndex | null;
    /** Clean, chronological chat text. */
    messages: { id: string; text: string }[];
    settings: PluginSettings;
    embedder: Embedder;
    estimateTokens: (text: string) => number;
    signal?: AbortSignal;
    /** Dry run: use this one text as the query instead of the last messages. */
    draft?: string;
};

export type RetrievalResult = {
    memories: Memory[];
    injectedText: string;
    tokens: number;
    reason?: NoMatchReason;
    detail?: string;
    candidates: number;
    skippedRecent: number;
    belowThreshold: number;
    best: number;
    droppedByTokenCap: number;
    queries: string[];
    durationMs: number;
    /** The log line, e.g. `match: 3 memories (scores 0.71, 0.58, 0.44) -> 212 tokens`. */
    line: string;
};

/** A chunk longer than this is not a useful query and only slows the embedder. */
const MAX_QUERY_CHARS = 1500;

type Working = Omit<RetrievalResult, "durationMs" | "line">;

/** A result with nothing injected; also what the timeout wrapper returns. */
export function emptyRetrievalResult(
    reason: NoMatchReason,
    line: string,
    detail?: string,
): RetrievalResult {
    return {
        memories: [],
        injectedText: "",
        tokens: 0,
        reason,
        detail,
        candidates: 0,
        skippedRecent: 0,
        belowThreshold: 0,
        best: 0,
        droppedByTokenCap: 0,
        queries: [],
        durationMs: 0,
        line,
    };
}

/**
 * Never throws: the prompt injector that awaits this cannot afford an exception
 * (the app aborts the whole prompt build), so every failure becomes a reason.
 */
export async function retrieveMemories(input: RetrievalInput): Promise<RetrievalResult> {
    const started = performance.now();
    const { settings, index, messages, embedder, estimateTokens } = input;

    const working: Working = {
        memories: [],
        injectedText: "",
        tokens: 0,
        candidates: 0,
        skippedRecent: 0,
        belowThreshold: 0,
        best: 0,
        droppedByTokenCap: 0,
        queries: [],
    };

    const finish = (line: string, reason?: NoMatchReason, detail?: string): RetrievalResult => ({
        ...working,
        reason,
        detail,
        durationMs: Math.round(performance.now() - started),
        line,
    });

    if (!settings.enabled) {
        return finish("no match: disabled", "disabled");
    }

    if (!index || index.chunks.length === 0) {
        return finish("no match: empty index", "empty-index");
    }

    const match = modelMatches(index, {
        id: embedder.model.id,
        // An unknown dimension (custom model, manual file) is only comparable
        // against the index's own, which is the best guess available.
        dim: embedder.model.dim ?? index.model.dim,
    });

    if (!match.ok) {
        return finish(`no match: ${match.reason}`, "model-mismatch", match.reason);
    }

    if (input.signal?.aborted) {
        return finish("no match: embedding error: aborted", "embedding-error", "aborted");
    }

    const queries = buildQueries(messages, settings.queryMessages, input.draft);
    working.queries = queries;

    if (queries.length === 0) {
        return finish("no match: no queries", "no-queries");
    }

    let queryVectors: Float32Array[];

    try {
        queryVectors = await embedder.embedQueries(queries, input.signal);
    } catch (error) {
        const detail = describeError(error);

        return finish(`no match: embedding error: ${detail}`, "embedding-error", detail);
    }

    const wrongDim = queryVectors.find((vector) => vector.length !== index.model.dim);

    if (wrongDim) {
        // The index was built with another model than the one answering now; the
        // manifest check cannot see it when the embedder has no known dimension.
        const detail = `the embedder returned ${wrongDim.length}d vectors but the index is ${index.model.dim}d`;
        return finish(`no match: ${detail}`, "model-mismatch", detail);
    }

    const recent = recentMessageIds(messages, settings.retain);
    const { scores, skipped, best } = scoreChunks(queryVectors, index.chunks, (chunk) =>
        recent.has(index.chunks[chunk].messageId),
    );
    // The embedder knows which prefix profile (and therefore which model family)
    // is answering, which is what decides the sensible cut-off.
    //
    // The cut-off is absolute on purpose: a relative band ("keep only what is
    // within X of the best match") cannot reject an unrelated turn, because the
    // best chunk always survives it. Measured on three chats, the false matches
    // come from the two-message window itself (an unrelated question inherits
    // its neighbour's score), which the cut-off cannot repair either; see the
    // calibration note on DEFAULT_THRESHOLDS.
    const threshold = effectiveThreshold(settings, embedder.model.prefixProfile);

    working.skippedRecent = skipped;
    working.candidates = scores.size;
    working.best = best;

    if (scores.size === 0) {
        // Every chunk belongs to a message that is still in the recent window.
        return finish("no match: all inside recent window", "all-in-recent-window");
    }

    const above = [...scores.entries()]
        // Model-aware: the profile's calibrated cut-off when the user left it on auto.
        .filter(([, score]) => score >= threshold)
        .sort((a, b) => b[1] - a[1]);

    working.belowThreshold = scores.size - above.length;

    if (above.length === 0) {
        // Clamped at 0 because a near-orthogonal cosine can round to `-0.00`,
        // which reads like a bug in the viewer.
        const bestText = formatScore(Math.max(0, best));

        return finish(
            `no match: all below threshold (best ${bestText})`,
            "below-threshold",
            `best score ${bestText} is below ${threshold}`,
        );
    }

    const selected = above.slice(0, settings.topK).map(([chunk, score]) => ({ chunk, score }));
    const merged = mergeChunks(index, selected, messages);

    // Template first, then drop the weakest memories until it fits: the template's
    // own wording costs tokens the caller cannot see from the cap alone.
    let kept = merged;
    let injectedText = renderTemplate(settings.template, kept);
    let tokens = estimateTokens(injectedText);
    let dropped = 0;

    while (tokens > settings.maxMemoryTokens && kept.length > 0) {
        kept = dropLowestScore(kept);
        dropped += 1;
        injectedText = renderTemplate(settings.template, kept);
        tokens = estimateTokens(injectedText);
    }

    working.droppedByTokenCap = dropped;

    if (kept.length === 0) {
        // Even one memory does not fit; injecting the template alone would tell
        // the model about memories it cannot see.
        working.memories = [];
        working.injectedText = "";
        working.tokens = 0;
        const detail = `one memory does not fit within ${settings.maxMemoryTokens} tokens`;

        return finish(`no match: token cap exceeded (${detail})`, "token-cap", detail);
    }

    working.memories = kept;
    working.injectedText = injectedText;
    working.tokens = tokens;

    const scoresText = kept
        .map((memory) => memory.score)
        .sort((a, b) => b - a)
        .map(formatScore)
        .join(", ");
    const count = kept.length === 1 ? "1 memory" : `${kept.length} memories`;

    return finish(`match: ${count} (scores ${scoresText}) -> ${tokens} tokens`);
}

/** Last `count` non-empty texts, oldest first so the batched request is stable. */
function buildQueries(
    messages: { id: string; text: string }[],
    count: number,
    draft: string | undefined,
): string[] {
    if (draft !== undefined) {
        const text = draft.trim().slice(0, MAX_QUERY_CHARS);
        return text ? [text] : [];
    }

    const queries: string[] = [];

    for (let i = messages.length - 1; i >= 0 && queries.length < count; i -= 1) {
        const text = messages[i].text.trim().slice(0, MAX_QUERY_CHARS);

        if (text) {
            queries.push(text);
        }
    }

    return queries.reverse();
}

function recentMessageIds(messages: { id: string; text: string }[], retain: number): Set<string> {
    const ids = new Set<string>();

    if (retain <= 0) {
        return ids;
    }

    for (let i = Math.max(0, messages.length - retain); i < messages.length; i += 1) {
        ids.add(messages[i].id);
    }

    return ids;
}

/**
 * Cosine of every chunk against every query, keeping the best match per chunk.
 *
 * `rankVectors` scores one query at a time, so it is called per query with no
 * threshold and the maximum is merged here; the recent window is excluded
 * through its `skip` so excluded chunks are never even scored, which keeps
 * `best` meaningful (it describes what could actually be injected).
 */
function scoreChunks(
    queryVectors: Float32Array[],
    chunks: IndexedChunk[],
    skip: (index: number) => boolean,
): { scores: Map<number, number>; skipped: number; best: number } {
    const vectors = chunks.map((chunk) => chunk.vector);
    const scores = new Map<number, number>();
    let skipped = 0;
    let best = -1;

    for (const query of queryVectors) {
        const ranked = rankVectors(query, vectors, { threshold: -Infinity, topK: vectors.length, skip });

        skipped = ranked.skipped;

        if (ranked.best > best) {
            best = ranked.best;
        }

        for (const entry of ranked.ranked) {
            const previous = scores.get(entry.index);

            if (previous === undefined || entry.score > previous) {
                scores.set(entry.index, entry.score);
            }
        }
    }

    return { scores, skipped, best };
}

/**
 * Turns the top-K chunks into prompt memories.
 *
 * Adjacency is decided per message by `ord`, never by index position:
 * `appendMessageChunks` re-appends an edited message at the end of the index, so
 * index order stops matching chat order after the first edit. The prompt order
 * therefore comes from `messages` itself; a memory whose message has left the
 * chat has no position, so it goes first and keeps its relative index order.
 */
function mergeChunks(
    index: ChatIndex,
    selected: { chunk: number; score: number }[],
    messages: { id: string; text: string }[],
): Memory[] {
    const ages = messageAges(messages);
    const positions = messagePositions(messages);
    const byMessage = new Map<string, { chunk: number; score: number }[]>();

    // Chunks of one message may arrive in any index order; grouping first lets
    // the merge work on `ord`, which is the real adjacency.
    for (const entry of selected) {
        const messageId = index.chunks[entry.chunk].messageId;
        const group = byMessage.get(messageId);

        if (group) {
            group.push(entry);
        } else {
            byMessage.set(messageId, [entry]);
        }
    }

    const merged: { memory: Memory; firstChunk: number }[] = [];

    for (const [messageId, entries] of byMessage) {
        const sorted = [...entries].sort((a, b) => index.chunks[a.chunk].ord - index.chunks[b.chunk].ord);
        let memory: Memory | null = null;
        let firstChunk = 0;
        let lastOrd = -1;

        const flush = () => {
            if (memory) {
                merged.push({ memory, firstChunk });
            }
        };

        for (const entry of sorted) {
            const chunk = index.chunks[entry.chunk];

            if (memory && chunk.ord === lastOrd + 1) {
                memory.text = `${memory.text} ${chunk.text}`;
                memory.score = Math.max(memory.score, entry.score);
                lastOrd = chunk.ord;
                continue;
            }

            flush();

            memory = {
                chunkId: chunk.id,
                messageId,
                ord: chunk.ord,
                score: entry.score,
                text: chunk.text,
                messagesAgo: ages.get(messageId) ?? -1,
            };
            firstChunk = entry.chunk;
            lastOrd = chunk.ord;
        }

        flush();
    }

    merged.sort((a, b) => {
        const leftKnown = a.memory.messagesAgo !== -1;
        const rightKnown = b.memory.messagesAgo !== -1;

        if (leftKnown !== rightKnown) {
            return leftKnown ? 1 : -1;
        }

        if (!leftKnown) {
            return a.firstChunk - b.firstChunk;
        }

        return (positions.get(a.memory.messageId) ?? 0) - (positions.get(b.memory.messageId) ?? 0);
    });

    return merged.map((entry) => entry.memory);
}

/** Position counted from the newest message; absent messages stay -1. */
function messageAges(messages: { id: string; text: string }[]): Map<string, number> {
    const ages = new Map<string, number>();

    for (let i = 0; i < messages.length; i += 1) {
        ages.set(messages[i].id, messages.length - 1 - i);
    }

    return ages;
}

/** Chat position, oldest first; the prompt order source. */
function messagePositions(messages: { id: string; text: string }[]): Map<string, number> {
    const positions = new Map<string, number>();

    for (let i = 0; i < messages.length; i += 1) {
        positions.set(messages[i].id, i);
    }

    return positions;
}

function renderTemplate(template: string, memories: Memory[]): string {
    const block = memories
        .map((memory) => `- ${memory.text.replace(/\r?\n/g, "\n  ")}`)
        .join("\n");

    // split/join so a template that repeats the marker is filled everywhere.
    return template.split(MEMORIES_PLACEHOLDER).join(block);
}

/** Removes the weakest memory; ties drop the later one, keeping the older story. */
function dropLowestScore(memories: Memory[]): Memory[] {
    let worst = 0;

    for (let i = 1; i < memories.length; i += 1) {
        if (memories[i].score < memories[worst].score) {
            worst = i;
        }
    }

    return memories.filter((_memory, i) => i !== worst);
}

function describeError(error: unknown): string {
    if (error instanceof EmbeddingError) {
        // An abort is not actionable, and the hint would only add noise to the log.
        if (error.code === "aborted") {
            return "aborted";
        }

        return error.hint ? `${error.message} ${error.hint}` : error.message;
    }

    return error instanceof Error ? error.message : String(error);
}

function formatScore(score: number): string {
    return score.toFixed(2);
}

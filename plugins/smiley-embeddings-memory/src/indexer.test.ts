import { describe, expect, test } from "bun:test";

import { hashText } from "./chunking";
import type { Embedder } from "./embedder";
import { EmbeddingError } from "./embedder";
import { createIndexer } from "./indexer";
import type { Indexer, IndexerChat } from "./indexer";
import { indexKey, loadIndex } from "./index-store";
import type { StorageLike } from "./index-store";
import type { ChatMessageLike } from "./reconcile";
import { DEFAULT_SETTINGS } from "./settings";
import type { PluginSettings } from "./settings";
import { createTrace } from "./trace";

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

async function waitFor(predicate: () => boolean, timeoutMs = 1000): Promise<void> {
    const deadline = Date.now() + timeoutMs;

    while (!predicate()) {
        if (Date.now() > deadline) {
            throw new Error("timed out waiting for the indexer");
        }

        await sleep(2);
    }
}

/** JSON round trip through strings: the store only ever writes JSON anyway. */
function memoryStorage() {
    const data = new Map<string, string>();
    const reads: string[] = [];

    const storage: StorageLike & { data: Map<string, string>; reads: string[] } = {
        data,
        reads,
        async getJson<T>(key: string, fallback: T): Promise<T> {
            reads.push(key);
            const raw = data.get(key);

            return raw === undefined ? fallback : (JSON.parse(raw) as T);
        },
        async setJson<T>(key: string, value: unknown): Promise<void> {
            data.set(key, JSON.stringify(value));
        },
        async remove(key: string): Promise<void> {
            data.delete(key);
        },
    };

    return storage;
}

function vectorFor(text: string, dim: number): Float32Array {
    const vector = new Float32Array(dim);
    let seed = 7;

    for (let i = 0; i < text.length; i += 1) {
        seed = (seed * 31 + text.charCodeAt(i)) % 99_991;
    }

    for (let i = 0; i < dim; i += 1) {
        vector[i] = ((seed + i * 7) % 13) + 1;
    }

    return vector;
}

function msg(id: string, text: string, streaming = false): ChatMessageLike {
    return streaming ? { id, text, streaming: true } : { id, text };
}

function chat(messages: ChatMessageLike[], id = "chat-1"): IndexerChat {
    return { id, messages };
}

function manyMessages(count: number): ChatMessageLike[] {
    return Array.from({ length: count }, (_unused, index) => msg(`m${index + 1}`, `message ${index + 1}`));
}

type Emb = {
    calls: string[][];
    id: string;
    dim: number;
    /** Overrides the vector size without changing the model's reported dim. */
    vectorDim: number;
    /** Block inside the n-th `embedDocuments` call until `blocked` is called. */
    hold: number;
    blocked: (() => void) | null;
    error: Error | null;
};

function harness(overrides: Partial<PluginSettings> = {}) {
    const storage = memoryStorage();
    const trace = createTrace(500);
    let settings: PluginSettings = {
        ...DEFAULT_SETTINGS,
        source: "local",
        autoIndexBacklog: true,
        chunkSize: 400,
        chunkOverlap: 0,
        ...overrides,
    };
    let chatState: IndexerChat | null = null;

    const emb: Emb = {
        calls: [],
        id: "fake-model",
        dim: 4,
        vectorDim: 0,
        hold: 0,
        blocked: null,
        error: null,
    };

    const embedder: Embedder = {
        get model() {
            return {
                id: emb.id,
                source: settings.source,
                prefixProfile: "none" as const,
                dim: emb.dim,
            };
        },
        async embedDocuments(texts, signal) {
            emb.calls.push([...texts]);

            if (emb.hold > 0 && emb.calls.length === emb.hold) {
                await new Promise<void>((resolve) => {
                    emb.blocked = resolve;
                });
            }

            if (signal?.aborted) {
                throw new EmbeddingError("aborted", "The embedding request was cancelled.", "Nothing to do.");
            }

            if (emb.error) {
                throw emb.error;
            }

            return texts.map((text) => vectorFor(text, emb.vectorDim || emb.dim));
        },
        async embedQueries(texts) {
            return texts.map((text) => vectorFor(text, emb.vectorDim || emb.dim));
        },
    };

    const indexer: Indexer = createIndexer({
        storage,
        getSettings: () => settings,
        getEmbedder: () => embedder,
        getChat: () => chatState,
        trace,
        scheduleMs: 10,
    });

    return {
        indexer,
        storage,
        trace,
        emb,
        setSettings(patch: Partial<PluginSettings>) {
            settings = { ...settings, ...patch };
        },
        setChat(next: IndexerChat | null) {
            chatState = next;
        },
    };
}

describe("createIndexer", () => {
    test("indexes a chat, persists it and logs the run", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));

        await h.indexer.indexAll();

        // One request covers both messages: batching is across messages, not per message.
        expect(h.emb.calls).toEqual([["Hello there", "Second message"]]);

        const index = await loadIndex(h.storage, "chat-1");

        expect(index?.chunks.map((chunk) => chunk.text)).toEqual(["Hello there", "Second message"]);
        expect(index?.messages.m1).toEqual({ hash: hashText("Hello there"), chunks: 1 });

        const status = await h.indexer.status("fake-model");

        expect(status).toEqual({
            chatId: "chat-1",
            state: "up-to-date",
            chunks: 2,
            messagesIndexed: 2,
            messagesTotal: 2,
            messagesPending: 0,
            indexModelId: "fake-model",
            activeModelId: "fake-model",
        });

        expect(h.trace.logs().some((entry) => /^index: 2 chunks from 2 messages in \d+ ms$/.test(entry.message))).toBe(true);
    });

    test("a new message is the only thing embedded on the next run", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));
        await h.indexer.reconcileNow();

        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message"), msg("m3", "Third")]));
        await h.indexer.reconcileNow();

        expect(h.emb.calls).toEqual([["Hello there", "Second message"], ["Third"]]);
        expect((await loadIndex(h.storage, "chat-1"))?.chunks).toHaveLength(3);
    });

    test("an edited message is dropped and embedded again", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));
        await h.indexer.reconcileNow();

        h.setChat(chat([msg("m1", "Hello again"), msg("m2", "Second message")]));
        await h.indexer.reconcileNow();

        expect(h.emb.calls[1]).toEqual(["Hello again"]);

        const index = await loadIndex(h.storage, "chat-1");

        // The edited message is appended after the drop, so its chunk moves to
        // the end of the index; ids stay stable.
        expect(index?.chunks.map((chunk) => chunk.id).sort()).toEqual(["m1:0", "m2:0"]);
        expect(index?.chunks.find((chunk) => chunk.messageId === "m1")?.text).toBe("Hello again");
        expect(index?.messages.m1.hash).toBe(hashText("Hello again"));
    });

    test("a deleted message is dropped without re-embedding the rest", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));
        await h.indexer.reconcileNow();

        h.setChat(chat([msg("m1", "Hello there")]));
        await h.indexer.reconcileNow();

        expect(h.emb.calls).toHaveLength(1);

        const index = await loadIndex(h.storage, "chat-1");

        expect(index?.chunks.map((chunk) => chunk.id)).toEqual(["m1:0"]);
        expect(index?.messages.m2).toBeUndefined();
        expect(h.trace.logs().some((entry) => entry.message === "drop: 1 messages")).toBe(true);
    });

    test("a streaming message is skipped until it finishes", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "typing...", true)]));
        await h.indexer.reconcileNow();

        expect(h.emb.calls).toEqual([["Hello there"]]);

        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "typing... done")]));
        await h.indexer.reconcileNow();

        expect(h.emb.calls).toEqual([["Hello there"], ["typing... done"]]);
    });

    test("cancel keeps the messages that finished and the next run resumes", async () => {
        const h = harness();
        h.setChat(chat(manyMessages(20)));
        h.emb.hold = 2;

        const run = h.indexer.indexAll();

        await waitFor(() => h.emb.blocked !== null);

        const during = h.indexer.progress();

        expect(during.running).toBe(true);
        expect(during.total).toBe(20);
        expect(during.done).toBe(16);

        h.indexer.cancel();
        h.emb.blocked?.();
        await run;

        expect((await loadIndex(h.storage, "chat-1"))?.chunks).toHaveLength(16);
        expect(h.trace.logs().some((entry) => entry.message === "index: cancelled after 16/20 messages")).toBe(true);

        const cancelled = await h.indexer.status("fake-model");

        expect(cancelled.state).toBe("stale");
        expect(cancelled.messagesPending).toBe(4);

        h.emb.hold = 0;
        h.emb.blocked = null;
        await h.indexer.indexAll();

        const finished = await h.indexer.status("fake-model");

        expect(finished.state).toBe("up-to-date");
        expect(finished.chunks).toBe(20);
    });

    test("a model change is reported as a mismatch and re-index rebuilds", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));
        await h.indexer.indexAll();

        h.emb.id = "other-model";

        const mismatch = await h.indexer.status("other-model");

        expect(mismatch.state).toBe("model-mismatch");
        expect(mismatch.detail).toContain("index built with fake-model");

        // Nothing was embedded with the new model: the two spaces are never mixed.
        expect(h.emb.calls).toHaveLength(1);

        await h.indexer.reindex();

        const rebuilt = await h.indexer.status("other-model");

        expect(rebuilt.state).toBe("up-to-date");
        expect(rebuilt.indexModelId).toBe("other-model");
        expect(rebuilt.chunks).toBe(2);
    });

    test("a dimension surprise stops with a model mismatch and does not retry", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there")]));
        await h.indexer.indexAll();

        h.emb.vectorDim = 5;
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));
        await h.indexer.reconcileNow();

        expect(h.emb.calls).toHaveLength(2);

        const status = await h.indexer.status("fake-model");

        expect(status.state).toBe("model-mismatch");
        expect(status.lastError).toContain("5d vectors but the index is 4d");

        // Asking again must not hammer the endpoint.
        await h.indexer.status("fake-model");
        await h.indexer.status("fake-model");
        expect(h.emb.calls).toHaveLength(2);
    });

    test("an embedding error is reported once and the next trigger retries", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));
        h.emb.error = new EmbeddingError(
            "server",
            "The embedding endpoint failed with HTTP 500.",
            "Try again; if it keeps failing, check the endpoint's logs.",
        );

        await h.indexer.reconcileNow();

        const failed = await h.indexer.status("fake-model");

        expect(failed.state).toBe("error");
        expect(failed.lastError).toContain("The embedding endpoint failed with HTTP 500.");
        expect(failed.lastError).toContain("check the endpoint's logs");
        expect(h.emb.calls).toHaveLength(1);
        expect(h.trace.logs().some((entry) => entry.level === "error" && entry.scope === "index")).toBe(true);

        h.emb.error = null;
        await h.indexer.reconcileNow();

        const recovered = await h.indexer.status("fake-model");

        expect(recovered.state).toBe("up-to-date");
        expect(h.emb.calls).toHaveLength(2);
    });

    test("switching chats drops the cached index and reloads it from storage", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")], "chat-1"));
        await h.indexer.indexAll();

        h.setChat(chat([msg("n1", "Other chat")], "chat-2"));
        await h.indexer.reconcileNow();

        expect((await h.indexer.status("fake-model")).chunks).toBe(1);

        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")], "chat-1"));

        const readsBefore = h.storage.reads.filter((key) => key === indexKey("chat-1")).length;
        const status = await h.indexer.status("fake-model");
        const readsAfter = h.storage.reads.filter((key) => key === indexKey("chat-1")).length;

        expect(status.chunks).toBe(2);
        expect(readsAfter).toBeGreaterThan(readsBefore);
        // The chat-1 index was never rebuilt on the way back.
        expect(h.emb.calls).toHaveLength(2);
    });

    test("purge deletes the index and the chat counts as not indexed again", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there")]));
        await h.indexer.indexAll();

        await h.indexer.purge();

        expect(await loadIndex(h.storage, "chat-1")).toBeNull();

        const status = await h.indexer.status("fake-model");

        expect(status.state).toBe("stale");
        expect(status.chunks).toBe(0);
        expect(status.messagesIndexed).toBe(0);
    });

    test("an API source needs consent before anything is sent", async () => {
        const h = harness({ source: "api", apiNoticeAccepted: false });
        h.setChat(chat([msg("m1", "Hello there")]));

        await h.indexer.reconcileNow();
        h.indexer.schedule();
        await sleep(40);

        expect(h.emb.calls).toEqual([]);
        expect((await h.indexer.status("fake-model")).state).toBe("needs-consent");
        await expect(h.indexer.indexAll()).rejects.toThrow(/accept the notice/i);

        h.setSettings({ apiNoticeAccepted: true });
        await h.indexer.indexAll();

        expect(h.emb.calls).toHaveLength(1);
        expect((await h.indexer.status("fake-model")).state).toBe("up-to-date");
    });

    test("an API backlog above 20 waits for the user's Index all", async () => {
        const h = harness({ source: "api", apiNoticeAccepted: true });
        h.setChat(chat(manyMessages(25)));

        h.indexer.schedule();
        await sleep(60);

        expect(h.emb.calls).toEqual([]);

        const waiting = await h.indexer.status("fake-model");

        expect(waiting.state).toBe("stale");
        expect(waiting.messagesPending).toBe(25);
        expect(waiting.detail).toContain("25 messages waiting");

        await h.indexer.indexAll();

        // 25 messages only cost two requests (16 + 9).
        expect(h.emb.calls).toHaveLength(2);

        h.setChat(chat(manyMessages(26)));
        h.indexer.schedule();
        await sleep(60);

        const status = await h.indexer.status("fake-model");

        expect(status.state).toBe("up-to-date");
        expect(status.chunks).toBe(26);
        expect(h.emb.calls).toHaveLength(3);
    });

    test("the schedule debounce coalesces a burst into one run", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));

        for (let i = 0; i < 5; i += 1) {
            h.indexer.schedule();
        }

        await sleep(60);

        expect(h.emb.calls).toHaveLength(1);
        expect((await h.indexer.status("fake-model")).state).toBe("up-to-date");

        h.indexer.dispose();
    });

    test("two concurrent runs do the work once", async () => {
        const h = harness();
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));

        await Promise.all([h.indexer.indexAll(), h.indexer.indexAll()]);

        expect(h.emb.calls).toHaveLength(1);
        expect((await loadIndex(h.storage, "chat-1"))?.chunks).toHaveLength(2);
    });

    test("with auto-indexing off, a new chat waits for Index all", async () => {
        const h = harness({ autoIndexBacklog: false });
        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message")]));

        await h.indexer.reconcileNow();

        expect(h.emb.calls).toEqual([]);

        const waiting = await h.indexer.status("fake-model");

        expect(waiting.state).toBe("stale");
        expect(waiting.messagesPending).toBe(2);

        await h.indexer.indexAll();

        h.setChat(chat([msg("m1", "Hello there"), msg("m2", "Second message"), msg("m3", "Third")]));
        await h.indexer.reconcileNow();

        // Once the index exists, new messages are indexed automatically again.
        expect(h.emb.calls).toHaveLength(2);
        expect((await h.indexer.status("fake-model")).state).toBe("up-to-date");
    });

    test("a corrupt index is reported as an error instead of crashing the run", async () => {
        const h = harness();
        h.storage.data.set(indexKey("chat-1"), JSON.stringify({ version: 1, chatId: "chat-1", model: {} }));
        h.setChat(chat([msg("m1", "Hello there")]));

        await h.indexer.reconcileNow();

        const status = await h.indexer.status("fake-model");

        expect(status.state).toBe("error");
        expect(status.lastError).toContain("corrupt");
        expect(h.emb.calls).toEqual([]);
    });

    test("progress is idle and empty when nothing runs", () => {
        const h = harness();

        expect(h.indexer.progress()).toEqual({
            running: false,
            cancelling: false,
            chatId: null,
            done: 0,
            total: 0,
            chunks: 0,
        });
    });
});

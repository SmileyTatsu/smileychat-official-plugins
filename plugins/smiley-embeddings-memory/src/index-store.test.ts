import { describe, expect, test } from "bun:test";

import type { StorageLike } from "./index-store";
import {
    REGISTRY_KEY,
    SHARD_SIZE,
    appendMessageChunks,
    createIndex,
    indexKey,
    loadIndex,
    modelMatches,
    purgeIndex,
    removeMessages,
    saveIndex,
    shardKey,
    sweepOrphans,
} from "./index-store";
import { encodeVector } from "./vectors";

/**
 * Storage stand-in that behaves like the app's: JSON in, JSON out.
 *
 * `getJson` clones, so a test cannot pass by mutating the object the store kept
 * and the call log is what lets the append/remove tests assert the *exact* set
 * of keys written, which is the whole point of sharding.
 */
class MemoryStorage implements StorageLike {
    readonly data = new Map<string, unknown>();
    readonly calls: { op: "get" | "set" | "remove"; key: string }[] = [];

    async getJson<T>(key: string, fallback: T): Promise<T> {
        this.calls.push({ op: "get", key });

        if (!this.data.has(key)) {
            return fallback;
        }

        return JSON.parse(JSON.stringify(this.data.get(key))) as T;
    }

    async setJson(key: string, value: unknown): Promise<void> {
        this.calls.push({ op: "set", key });
        this.data.set(key, JSON.parse(JSON.stringify(value)));
    }

    async remove(key: string): Promise<void> {
        this.calls.push({ op: "remove", key });
        this.data.delete(key);
    }

    keysWritten(): string[] {
        return this.calls.filter((call) => call.op === "set").map((call) => call.key);
    }

    keysRemoved(): string[] {
        return this.calls.filter((call) => call.op === "remove").map((call) => call.key);
    }

    reset(): void {
        this.calls.length = 0;
    }
}

const model = {
    id: "embeddinggemma-test",
    source: "local" as const,
    dim: 4,
    prefixProfile: "gemma" as const,
};

function vector(dim: number, seed: number): Float32Array {
    const values = new Float32Array(dim);

    for (let i = 0; i < dim; i++) {
        values[i] = Math.sin(seed + i * 0.5) * (1 + i / 3);
    }

    return values;
}

function makeChunks(count: number, seed = 0): { text: string; vector: Float32Array }[] {
    return Array.from({ length: count }, (_value, i) => ({
        text: `chunk ${seed + i}`,
        vector: vector(model.dim, seed + i),
    }));
}

function manifestOf(storage: MemoryStorage, chatId: string) {
    return storage.data.get(indexKey(chatId)) as {
        version: number;
        chatId: string;
        shards: number[];
        counts: number[];
        messages: Record<string, { hash: string; chunks: number }>;
        updatedAt: string;
    };
}

describe("index store", () => {
    test("round-trips vectors bit-exactly through base64 shards", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-1", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(2));
        await appendMessageChunks(storage, index, "m2", "h2", makeChunks(3, 10));
        await saveIndex(storage, index);

        const loaded = await loadIndex(storage, "chat-1");

        expect(loaded).not.toBeNull();
        expect(loaded?.chatId).toBe("chat-1");
        expect(loaded?.model).toEqual(model);
        expect(loaded?.updatedAt).toBe(index.updatedAt);
        expect(loaded?.chunks.length).toBe(5);
        expect(loaded?.messages).toEqual({
            m1: { hash: "h1", chunks: 2 },
            m2: { hash: "h2", chunks: 3 },
        });

        for (let i = 0; i < 5; i++) {
            expect(loaded?.chunks[i].id).toBe(index.chunks[i].id);
            expect(loaded?.chunks[i].text).toBe(index.chunks[i].text);
            // Exact float equality: the point of base64 float32 is that the
            // bytes survive, so a rounded comparison would hide real damage.
            expect(Array.from(loaded?.chunks[i].vector ?? [])).toEqual(
                Array.from(index.chunks[i].vector),
            );
        }
    });

    test("uses stable chunk ids from the message id and ordinal", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-ids", model);

        await appendMessageChunks(storage, index, "msg-9", "h", makeChunks(3, 5));

        expect(index.chunks.map((chunk) => chunk.id)).toEqual(["msg-9:0", "msg-9:1", "msg-9:2"]);
        expect(index.chunks.map((chunk) => chunk.ord)).toEqual([0, 1, 2]);
    });

    test("splits chunks into shards of at most SHARD_SIZE (or the given size)", async () => {
        expect(SHARD_SIZE).toBe(2000);

        const storage = new MemoryStorage();
        const index = createIndex("chat-2", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(7));
        await saveIndex(storage, index, 3);

        const manifest = manifestOf(storage, "chat-2");

        expect(manifest.version).toBe(1);
        expect(manifest.shards).toEqual([0, 1, 2]);
        expect(manifest.counts).toEqual([3, 3, 1]);
        expect(storage.data.has(shardKey("chat-2", 2))).toBe(true);
        expect(storage.data.has(shardKey("chat-2", 3))).toBe(false);

        const loaded = await loadIndex(storage, "chat-2");

        expect(loaded?.chunks.map((chunk) => chunk.id)).toEqual(
            index.chunks.map((chunk) => chunk.id),
        );
    });

    test("append writes only the tail shard and the manifest", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-3", model);

        // Two shards, the second one half full: 4 chunks with shardSize 3.
        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(4), 3);
        storage.reset();

        await appendMessageChunks(storage, index, "m2", "h2", makeChunks(1, 20), 3);

        expect(storage.keysWritten()).toEqual([shardKey("chat-3", 1), indexKey("chat-3")]);
        expect(storage.keysRemoved()).toEqual([]);

        const loaded = await loadIndex(storage, "chat-3");

        expect(loaded?.chunks.length).toBe(5);
        expect(loaded?.messages.m2).toEqual({ hash: "h2", chunks: 1 });
    });

    test("append that overflows the tail opens the next shard only", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-3b", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(4), 3);
        storage.reset();

        await appendMessageChunks(storage, index, "m2", "h2", makeChunks(3, 20), 3);

        expect(storage.keysWritten()).toEqual([
            shardKey("chat-3b", 1),
            shardKey("chat-3b", 2),
            indexKey("chat-3b"),
        ]);
        expect(manifestOf(storage, "chat-3b").counts).toEqual([3, 3, 1]);
    });

    test("append registers the chat once", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-reg", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(2));
        storage.reset();
        await appendMessageChunks(storage, index, "m2", "h2", makeChunks(2, 9));

        expect(storage.keysWritten()).not.toContain(REGISTRY_KEY);
        expect(storage.data.get(REGISTRY_KEY)).toEqual({ chatIds: ["chat-reg"] });
    });

    test("records a message with no chunks without writing a shard", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-4", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(2), 3);
        storage.reset();
        await appendMessageChunks(storage, index, "empty", "h-empty", [], 3);

        // Only the manifest: an empty message must not create an empty shard,
        // but it still needs its entry so reconcile stops retrying it.
        expect(storage.keysWritten()).toEqual([indexKey("chat-4")]);
        expect(manifestOf(storage, "chat-4").counts).toEqual([2]);
        expect(index.messages.empty).toEqual({ hash: "h-empty", chunks: 0 });

        const loaded = await loadIndex(storage, "chat-4");

        expect(loaded?.messages.empty).toEqual({ hash: "h-empty", chunks: 0 });
        expect(loaded?.chunks.length).toBe(2);
    });

    test("removeMessages repacks the shards after the first removed chunk", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-5", model);

        for (const id of ["m1", "m2", "m3", "m4"]) {
            await appendMessageChunks(storage, index, id, `h-${id}`, makeChunks(3), 3);
        }

        // 12 chunks -> shards 0..3 of 3 each.
        expect(manifestOf(storage, "chat-5").counts).toEqual([3, 3, 3, 3]);
        storage.reset();

        await removeMessages(storage, index, ["m1", "m2"], 3);

        // Six chunks left -> two shards. Only those two are rewritten, and the
        // two keys past the new end are deleted.
        expect(storage.keysWritten()).toEqual([shardKey("chat-5", 0), shardKey("chat-5", 1), indexKey("chat-5")]);
        expect(storage.keysRemoved()).toEqual([shardKey("chat-5", 2), shardKey("chat-5", 3)]);
        expect(manifestOf(storage, "chat-5").counts).toEqual([3, 3]);
        expect(Object.keys(manifestOf(storage, "chat-5").messages)).toEqual(["m3", "m4"]);

        const loaded = await loadIndex(storage, "chat-5");

        expect(loaded?.chunks.map((chunk) => chunk.id)).toEqual(
            ["m3:0", "m3:1", "m3:2", "m4:0", "m4:1", "m4:2"],
        );
        expect(loaded?.messages).toEqual({ m3: { hash: "h-m3", chunks: 3 }, m4: { hash: "h-m4", chunks: 3 } });
    });

    test("removeMessages drops an entry whose message had no chunks", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-5b", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(2), 3);
        await appendMessageChunks(storage, index, "empty", "h-empty", [], 3);
        storage.reset();

        await removeMessages(storage, index, ["empty"], 3);

        expect(storage.keysWritten()).toEqual([indexKey("chat-5b")]);
        expect(index.messages).toEqual({ m1: { hash: "h1", chunks: 2 } });
    });

    test("removeMessages with nothing to remove writes nothing", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-5c", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(2), 3);
        storage.reset();

        await removeMessages(storage, index, ["not-there"], 3);

        expect(storage.calls).toEqual([]);
    });

    test("purge leaves only the other chats and a registry without the purged id", async () => {
        const storage = new MemoryStorage();
        const alpha = createIndex("alpha", model);
        const beta = createIndex("beta", model);

        await appendMessageChunks(storage, alpha, "m1", "h", makeChunks(2));
        await appendMessageChunks(storage, beta, "m1", "h", makeChunks(2));

        expect(storage.data.get(REGISTRY_KEY)).toEqual({ chatIds: ["alpha", "beta"] });

        await purgeIndex(storage, "alpha");

        expect(await loadIndex(storage, "alpha")).toBeNull();
        expect(storage.data.get(REGISTRY_KEY)).toEqual({ chatIds: ["beta"] });
        expect([...storage.data.keys()].sort()).toEqual(
            [indexKey("beta"), shardKey("beta", 0), REGISTRY_KEY].sort(),
        );
        expect(await loadIndex(storage, "beta")).not.toBeNull();
    });

    test("purge of a chat without an index still cleans the registry", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("kept", model);

        await appendMessageChunks(storage, index, "m1", "h", makeChunks(2));
        await storage.setJson(REGISTRY_KEY, { chatIds: ["kept", "ghost"] });

        await purgeIndex(storage, "ghost");

        expect(storage.data.get(REGISTRY_KEY)).toEqual({ chatIds: ["kept"] });
        expect(await loadIndex(storage, "kept")).not.toBeNull();
    });

    test("sweepOrphans removes only indexes for chats that no longer exist", async () => {
        const storage = new MemoryStorage();
        const kept = createIndex("kept", model);
        const gone = createIndex("gone", model);

        await appendMessageChunks(storage, kept, "m1", "h", makeChunks(2));
        await appendMessageChunks(storage, gone, "m1", "h", makeChunks(2));

        const removed = await sweepOrphans(storage, new Set(["kept"]));

        expect(removed).toEqual(["gone"]);
        expect(await loadIndex(storage, "gone")).toBeNull();
        expect(await loadIndex(storage, "kept")).not.toBeNull();
        expect(storage.data.get(REGISTRY_KEY)).toEqual({ chatIds: ["kept"] });
        expect([...storage.data.keys()].sort()).toEqual(
            [indexKey("kept"), shardKey("kept", 0), REGISTRY_KEY].sort(),
        );
    });

    test("sweepOrphans with an empty chat list removes nothing", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("only", model);

        await appendMessageChunks(storage, index, "m1", "h", makeChunks(2));

        const removed = await sweepOrphans(storage, new Set());

        // A failed listChats() returns empty; wiping every index then would
        // throw away embeddings the user paid for.
        expect(removed).toEqual([]);
        expect(await loadIndex(storage, "only")).not.toBeNull();
        expect(storage.data.get(REGISTRY_KEY)).toEqual({ chatIds: ["only"] });
    });

    test("loadIndex returns null when there is no manifest", async () => {
        const storage = new MemoryStorage();

        expect(await loadIndex(storage, "never-indexed")).toBeNull();
        expect(storage.keysWritten()).toEqual([]);
    });

    test("loadIndex throws naming the key of a missing shard", async () => {
        const storage = new MemoryStorage();
        const index = createIndex("chat-6", model);

        await appendMessageChunks(storage, index, "m1", "h1", makeChunks(4), 3);
        storage.data.delete(shardKey("chat-6", 1));

        await expect(loadIndex(storage, "chat-6")).rejects.toThrow(
            new RegExp(`index shard index-chat-6-1 is missing or corrupt`),
        );
    });

    test("loadIndex throws when a vector does not match the manifest dimension", async () => {
        const storage = new MemoryStorage();
        const chatId = "wrong-dim";

        await storage.setJson(indexKey(chatId), {
            version: 1,
            chatId,
            model,
            shards: [0],
            counts: [1],
            messages: { m1: { hash: "h", chunks: 1 } },
            updatedAt: new Date().toISOString(),
        });
        await storage.setJson(shardKey(chatId, 0), {
            chunks: [
                {
                    id: "m1:0",
                    messageId: "m1",
                    ord: 0,
                    text: "text",
                    // Three floats against a four-dim manifest: the classic
                    // "user switched to another embedding model" damage.
                    vec: encodeVector(new Float32Array([1, 2, 3])),
                },
            ],
        });

        await expect(loadIndex(storage, chatId)).rejects.toThrow(/is not 4 floats/);
    });

    test("modelMatches compares both id and dimension", () => {
        const index = createIndex("chat-7", model);

        expect(modelMatches(index, { id: model.id, dim: model.dim })).toEqual({ ok: true });

        const otherModel = modelMatches(index, { id: "qwen3-4b", dim: model.dim });

        expect(otherModel.ok).toBe(false);

        if (!otherModel.ok) {
            expect(otherModel.reason).toBe(
                "index built with embeddinggemma-test (4d), active model is qwen3-4b (4d)",
            );
        }

        const otherDim = modelMatches(index, { id: model.id, dim: 768 });

        expect(otherDim.ok).toBe(false);

        if (!otherDim.ok) {
            expect(otherDim.reason).toBe(
                "index built with embeddinggemma-test (4d), active model is embeddinggemma-test (768d)",
            );
        }
    });

    test("keys concatenate the chat id verbatim, so sanitised ids would collide", async () => {
        expect(indexKey("chat id with space")).toBe("index-chat id with space");
        expect(shardKey("c", 2)).toBe("index-c-2");
        expect(REGISTRY_KEY).toBe("index-registry");

        const storage = new MemoryStorage();
        const index = createIndex("a/b", model);

        await appendMessageChunks(storage, index, "m1", "h", makeChunks(1));

        expect(storage.data.has("index-a/b")).toBe(true);

        // The server sanitises chat ids to [A-Za-z0-9._-], so "a/b" and "a_b"
        // reach storage as the same id and share one index. This store does not
        // re-sanitise (the app owns id generation, and rewriting here would make
        // a chat unfindable); the collision is documented, not hidden.
        const sanitise = (id: string) => id.replace(/[^A-Za-z0-9._-]/g, "_");

        expect(sanitise("a/b")).toBe(sanitise("a_b"));
    });

    test("round-trips 5000 chunks at 768 dimensions", async () => {
        const storage = new MemoryStorage();
        const largeModel = { ...model, id: "qwen3-4b", dim: 768 };
        const index = createIndex("big", largeModel);
        const chunks = Array.from({ length: 5000 }, (_value, i) => ({
            text: `big chunk ${i}`,
            vector: vector(768, i),
        }));

        const saveStart = performance.now();

        await appendMessageChunks(storage, index, "m1", "h1", chunks);
        await saveIndex(storage, index);

        const saveMs = performance.now() - saveStart;
        const loadStart = performance.now();
        const loaded = await loadIndex(storage, "big");
        const loadMs = performance.now() - loadStart;

        // Printed, not asserted: this machine's numbers only bound the shape.
        console.log(
            `[index-store] 5000 x 768d: save ${saveMs.toFixed(0)} ms, load ${loadMs.toFixed(0)} ms, ${manifestOf(storage, "big").counts.length} shards`,
        );

        expect(loaded?.chunks.length).toBe(5000);
        expect(Array.from(loaded?.chunks[4999].vector ?? [])).toEqual(Array.from(chunks[4999].vector));
    });
});

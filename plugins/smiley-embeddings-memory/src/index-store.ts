import { decodeVector, encodeVector } from "./vectors";
import type { PrefixProfile } from "./models";

/**
 * The subset of `api.storage` the index needs.
 *
 * Typed structurally instead of through the plugin API so the store is testable
 * with a Map and so nothing else in this plugin depends on the app package.
 */
export type StorageLike = {
    getJson<T>(key: string, fallback: T): Promise<T>;
    setJson<T>(key: string, value: unknown): Promise<void>;
    remove(key: string): Promise<void>;
};

export type IndexModelInfo = {
    id: string;
    source: "local" | "api";
    dim: number;
    prefixProfile: PrefixProfile;
};

export type IndexedChunk = {
    id: string;
    messageId: string;
    ord: number;
    text: string;
    vector: Float32Array;
};

export type MessageEntry = {
    hash: string;
    chunks: number;
};

export type ChatIndex = {
    chatId: string;
    model: IndexModelInfo;
    chunks: IndexedChunk[];
    messages: Record<string, MessageEntry>;
    updatedAt: string;
};

/** Chunks per shard. 2000 x 2560d stays around 27 MB of base64 per write. */
export const SHARD_SIZE = 2000;

/**
 * Storage keys are built by plain concatenation from the chat id as given.
 *
 * The server sanitises chat ids to `[A-Za-z0-9._-]`, so re-sanitising here would
 * only hide a mismatch. The consequence, documented by a test: two distinct ids
 * that differ *only* in characters outside that set (say `a/b` and `a_b`) would
 * be sanitised to the same key by the server and therefore share one index.
 */
export function indexKey(chatId: string): string {
    return `index-${chatId}`;
}

export function shardKey(chatId: string, n: number): string {
    return `index-${chatId}-${n}`;
}

export const REGISTRY_KEY = "index-registry";

type StoredChunk = {
    id: string;
    messageId: string;
    ord: number;
    text: string;
    /** base64 little-endian float32, so the shard stays JSON. */
    vec: string;
};

type StoredShard = {
    chunks: StoredChunk[];
};

type StoredManifest = {
    version: 1;
    chatId: string;
    model: IndexModelInfo;
    shards: number[];
    counts: number[];
    messages: Record<string, MessageEntry>;
    updatedAt: string;
};

type StoredRegistry = {
    chatIds: string[];
};

export function createIndex(chatId: string, model: IndexModelInfo): ChatIndex {
    return {
        chatId,
        model: { ...model },
        chunks: [],
        messages: {},
        updatedAt: nowIso(),
    };
}

export async function loadIndex(storage: StorageLike, chatId: string): Promise<ChatIndex | null> {
    const manifestKey = indexKey(chatId);
    const manifest = await storage.getJson<StoredManifest | null>(manifestKey, null);

    if (manifest === null || manifest === undefined) {
        return null;
    }

    if (!isManifest(manifest)) {
        throw new Error(`index manifest ${manifestKey} is corrupt; rebuild the index`);
    }

    const dim = manifest.model.dim;
    const shardNumbers = Array.isArray(manifest.shards) ? manifest.shards : [];

    // Shards are independent keys, so load them together: a rebuilt 20k-chunk
    // index is ten round trips instead of ten serial ones.
    const loaded = await Promise.all(
        shardNumbers.map(async (n) => {
            const key = shardKey(chatId, n);
            const shard = await storage.getJson<StoredShard | null>(key, null);

            if (!shard || !Array.isArray(shard.chunks)) {
                throw new Error(`index shard ${key} is missing or corrupt; rebuild the index`);
            }

            return { key, chunk: shard.chunks };
        }),
    );

    const chunks: IndexedChunk[] = [];

    for (const { key, chunk: storedChunks } of loaded) {
        for (const stored of storedChunks) {
            if (!stored || typeof stored.vec !== "string") {
                throw new Error(`index shard ${key} has a malformed chunk; rebuild the index`);
            }

            let vector: Float32Array;

            try {
                vector = decodeVector(stored.vec, dim);
            } catch {
                // Vectors of another dimension mean the manifest and the shard
                // disagree; mixing them would silently compare unrelated spaces.
                throw new Error(`index shard ${key} has a vector that is not ${dim} floats; rebuild the index`);
            }

            chunks.push({
                id: stored.id,
                messageId: stored.messageId,
                ord: stored.ord,
                text: stored.text,
                vector,
            });
        }
    }

    return {
        chatId,
        model: { ...manifest.model },
        chunks,
        messages: { ...(manifest.messages ?? {}) },
        updatedAt: manifest.updatedAt ?? "",
    };
}

export async function saveIndex(
    storage: StorageLike,
    index: ChatIndex,
    shardSize = SHARD_SIZE,
): Promise<void> {
    const previous = await storage.getJson<StoredManifest | null>(indexKey(index.chatId), null);
    const counts = shardCounts(index.chunks.length, shardSize);

    // Shards first, manifest last: a crash before the manifest leaves the old
    // manifest pointing at shards that are still valid, never at half a chat.
    for (let n = 0; n < counts.length; n++) {
        await storage.setJson(shardKey(index.chatId, n), packShard(index.chunks, n, shardSize));
    }

    index.updatedAt = nowIso();
    await storage.setJson(indexKey(index.chatId), buildManifest(index, counts));

    // Only now is nothing pointing at the surplus shards.
    await removeShardsFrom(storage, index.chatId, counts.length, previous);
    await registerChat(storage, index.chatId);
}

export async function appendMessageChunks(
    storage: StorageLike,
    index: ChatIndex,
    messageId: string,
    hash: string,
    chunks: { text: string; vector: Float32Array }[],
    shardSize = SHARD_SIZE,
): Promise<void> {
    const firstNew = index.chunks.length;

    for (let ord = 0; ord < chunks.length; ord++) {
        index.chunks.push({
            // Ordinals are per message, so the id is stable across reloads and
            // does not shift when earlier messages are dropped.
            id: `${messageId}:${ord}`,
            messageId,
            ord,
            text: chunks[ord].text,
            vector: chunks[ord].vector,
        });
    }

    // A message with zero chunks still needs an entry, otherwise reconcile would
    // retry embedding it on every run.
    index.messages[messageId] = { hash, chunks: chunks.length };

    const counts = shardCounts(index.chunks.length, shardSize);

    // Shards are always written packed in chunk order, so the tail can be
    // recomputed from the flat array instead of reading the old shard back.
    const firstTouched = Math.floor(firstNew / shardSize);

    for (let n = firstTouched; n < counts.length && chunks.length > 0; n++) {
        await storage.setJson(shardKey(index.chatId, n), packShard(index.chunks, n, shardSize));
    }

    index.updatedAt = nowIso();
    await storage.setJson(indexKey(index.chatId), buildManifest(index, counts));
    await registerChat(storage, index.chatId);
}

export async function removeMessages(
    storage: StorageLike,
    index: ChatIndex,
    messageIds: string[],
    shardSize = SHARD_SIZE,
): Promise<void> {
    const doomed = new Set(messageIds);
    const beforeCount = index.chunks.length;

    let firstRemovedAt = -1;

    for (let i = 0; i < index.chunks.length; i++) {
        if (doomed.has(index.chunks[i].messageId)) {
            firstRemovedAt = i;
            break;
        }
    }

    const entryIds = Object.keys(index.messages).filter((id) => doomed.has(id));

    if (firstRemovedAt === -1 && entryIds.length === 0) {
        return;
    }

    const kept = index.chunks.filter((chunk) => !doomed.has(chunk.messageId));
    index.chunks.splice(0, index.chunks.length, ...kept);

    for (const id of entryIds) {
        delete index.messages[id];
    }

    const beforeShards = Math.ceil(beforeCount / shardSize);
    const counts = shardCounts(index.chunks.length, shardSize);

    // Everything from the shard that held the first removed chunk shifts, so
    // those are the only shards that need a rewrite; earlier ones are untouched.
    const firstTouched = firstRemovedAt === -1 ? counts.length : Math.floor(firstRemovedAt / shardSize);

    for (let n = firstTouched; n < counts.length; n++) {
        await storage.setJson(shardKey(index.chatId, n), packShard(index.chunks, n, shardSize));
    }

    index.updatedAt = nowIso();
    await storage.setJson(indexKey(index.chatId), buildManifest(index, counts));

    for (let n = counts.length; n < beforeShards; n++) {
        await storage.remove(shardKey(index.chatId, n));
    }
}

export async function purgeIndex(storage: StorageLike, chatId: string): Promise<void> {
    await purgeChat(storage, chatId);
    await unregisterChat(storage, chatId);
}

export async function sweepOrphans(storage: StorageLike, existingChatIds: Set<string>): Promise<string[]> {
    // An empty set is "the listing failed", not "every chat was deleted":
    // removing every index on a failed listing would destroy user data that
    // cannot be rebuilt without paying for embeddings again.
    if (existingChatIds.size === 0) {
        return [];
    }

    const registry = await readRegistry(storage);
    const orphans = registry.chatIds.filter((chatId) => !existingChatIds.has(chatId));

    if (orphans.length === 0) {
        return [];
    }

    for (const chatId of orphans) {
        await purgeChat(storage, chatId);
    }

    // One registry write for the whole sweep instead of one per chat.
    await storage.setJson<StoredRegistry>(REGISTRY_KEY, {
        chatIds: registry.chatIds.filter((chatId) => existingChatIds.has(chatId)),
    });

    return orphans;
}

export function modelMatches(
    index: ChatIndex,
    active: { id: string; dim: number },
): { ok: true } | { ok: false; reason: string } {
    if (index.model.id === active.id && index.model.dim === active.dim) {
        return { ok: true };
    }

    return {
        ok: false,
        reason: `index built with ${index.model.id} (${index.model.dim}d), active model is ${active.id} (${active.dim}d)`,
    };
}

function nowIso(): string {
    return new Date().toISOString();
}

function shardCounts(total: number, shardSize: number): number[] {
    const counts: number[] = [];

    for (let n = 0; n < total; n += shardSize) {
        counts.push(Math.min(shardSize, total - n));
    }

    return counts;
}

function packShard(chunks: IndexedChunk[], n: number, shardSize: number): StoredShard {
    const start = n * shardSize;

    return {
        chunks: chunks.slice(start, start + shardSize).map((chunk) => ({
            id: chunk.id,
            messageId: chunk.messageId,
            ord: chunk.ord,
            text: chunk.text,
            vec: encodeVector(chunk.vector),
        })),
    };
}

function buildManifest(index: ChatIndex, counts: number[]): StoredManifest {
    return {
        version: 1,
        chatId: index.chatId,
        model: { ...index.model },
        shards: counts.map((_count, n) => n),
        counts,
        messages: { ...index.messages },
        updatedAt: index.updatedAt,
    };
}

/**
 * Removes every shard key a previous manifest listed from `firstKept` up, plus
 * one probe past the highest one it knew about. The probe covers the shard an
 * interrupted append wrote before its manifest landed.
 */
async function removeShardsFrom(
    storage: StorageLike,
    chatId: string,
    firstKept: number,
    previous: StoredManifest | null,
): Promise<void> {
    const known = previous?.shards?.filter((n) => Number.isInteger(n)) ?? [];
    const highest = known.length > 0 ? Math.max(...known) : firstKept - 1;

    for (let n = firstKept; n <= highest + 1; n++) {
        await storage.remove(shardKey(chatId, n));
    }
}

/**
 * Deletes the manifest and shards of one chat. The registry is left alone so a
 * sweep can batch that write.
 */
async function purgeChat(storage: StorageLike, chatId: string): Promise<void> {
    const manifest = await storage.getJson<StoredManifest | null>(indexKey(chatId), null);
    const known = manifest && Array.isArray(manifest.shards) ? manifest.shards : [];
    const highest = known.reduce((max, n) => (Number.isInteger(n) && n > max ? n : max), -1);

    await storage.remove(indexKey(chatId));

    for (let n = 0; n <= highest + 1; n++) {
        await storage.remove(shardKey(chatId, n));
    }
}

async function readRegistry(storage: StorageLike): Promise<StoredRegistry> {
    const raw = await storage.getJson<StoredRegistry | null>(REGISTRY_KEY, null);
    const chatIds = Array.isArray(raw?.chatIds) ? raw.chatIds.filter((id) => typeof id === "string") : [];

    return { chatIds: [...new Set(chatIds)] };
}

async function registerChat(storage: StorageLike, chatId: string): Promise<boolean> {
    const registry = await readRegistry(storage);

    if (registry.chatIds.includes(chatId)) {
        return false;
    }

    await storage.setJson<StoredRegistry>(REGISTRY_KEY, { chatIds: [...registry.chatIds, chatId] });

    return true;
}

async function unregisterChat(storage: StorageLike, chatId: string): Promise<void> {
    const registry = await readRegistry(storage);

    if (!registry.chatIds.includes(chatId)) {
        return;
    }

    await storage.setJson<StoredRegistry>(REGISTRY_KEY, {
        chatIds: registry.chatIds.filter((id) => id !== chatId),
    });
}

function isModelInfo(value: unknown): value is IndexModelInfo {
    if (!value || typeof value !== "object") {
        return false;
    }

    const model = value as Partial<IndexModelInfo>;

    return (
        typeof model.id === "string" &&
        (model.source === "local" || model.source === "api") &&
        typeof model.dim === "number" &&
        model.dim > 0 &&
        Number.isInteger(model.dim)
    );
}

function isManifest(value: unknown): value is StoredManifest {
    if (!value || typeof value !== "object") {
        return false;
    }

    const manifest = value as Partial<StoredManifest>;

    return manifest.version === 1 && typeof manifest.chatId === "string" && isModelInfo(manifest.model);
}

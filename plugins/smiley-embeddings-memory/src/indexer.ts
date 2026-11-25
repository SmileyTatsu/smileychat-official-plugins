/**
 * The background indexer: keeps one chat's embedding index in step with its
 * messages.
 *
 * Everything the app owns arrives through small injected functions, so the whole
 * state machine (debounce, batching, cancellation, error state) is testable
 * without a browser, a server or a chat store.
 *
 * Two invariants drive the design:
 * - One run per chat at a time. A second request joins the run in flight instead
 *   of restarting it, and the per-chat promise chain keeps a queued run from
 *   overlapping the previous one.
 * - Progress is persisted per message (`appendMessageChunks`), so a cancel, a
 *   crash or a reload keeps every message that already finished.
 */

import { chunkText } from "./chunking";
import type { Embedder } from "./embedder";
import { EmbeddingError } from "./embedder";
import type { IndexState, IndexStatus, IndexerProgress } from "./controller";
import type { ChatIndex, StorageLike } from "./index-store";
import {
    appendMessageChunks,
    createIndex,
    loadIndex,
    modelMatches,
    purgeIndex,
    removeMessages,
} from "./index-store";
import { planReconcile } from "./reconcile";
import type { ChatMessageLike, ReconcilePlan } from "./reconcile";
import type { PluginSettings } from "./settings";
import type { Trace } from "./trace";

/** The active chat reduced to what the indexer reads. */
export type IndexerChat = {
    id: string;
    /** Cleaned text per indexable message, in chat order (see messages.ts). */
    messages: ChatMessageLike[];
};

export type IndexerDeps = {
    storage: StorageLike;
    getSettings(): PluginSettings;
    /**
     * The embedder for the current settings. Read at the start of every run, so
     * a settings change is picked up by the next trigger without a restart.
     */
    getEmbedder(): Embedder;
    getChat(): IndexerChat | null;
    trace: Trace;
    /** Injected so tests get deterministic durations. */
    now?(): number;
    /** Debounce for `schedule()`; 1500 ms keeps a burst of edits to one run. */
    scheduleMs?: number;
};

export type Indexer = {
    /** Plan against the loaded index and embed/drop what changed. */
    reconcileNow(): Promise<void>;
    /** Index every message of the open chat, ignoring the auto-index policy. */
    indexAll(): Promise<void>;
    /** Purge, then index everything again with the active model. */
    reindex(): Promise<void>;
    purge(): Promise<void>;
    /** Debounced trigger used by the app subscriptions. */
    schedule(): void;
    cancel(): void;
    status(activeModelId: string): Promise<IndexStatus>;
    progress(): IndexerProgress;
    /**
     * Read-only index for `chatId`: the cache for the active chat, a plain load
     * for any other. Never writes, so the injector can call it per turn.
     */
    readIndex(chatId: string): Promise<ChatIndex | null>;
    onChange(listener: () => void): () => void;
    dispose(): void;
};

type RunKind = "reconcile" | "all";

/**
 * The embedder already batches at 16; capping here as well is what lets one
 * request cover several short messages while a single long message is split
 * across batches and appended once, when complete.
 */
const EMBED_BATCH = 16;

/** Above this, an API backlog waits for the user's own "Index all". */
const API_AUTO_LIMIT = 20;

const DEFAULT_SCHEDULE_MS = 1500;

type Failure = { kind: "error" | "model-mismatch"; message: string };

type RunState = {
    chatId: string;
    controller: AbortController;
    cancelling: boolean;
    done: number;
    total: number;
    chunks: number;
    startedAt: string;
};

type Inflight = { chatId: string; promise: Promise<void> };

/** A vector whose size disagrees with the index: never mixed, never averaged. */
class DimensionMismatch extends Error {
    constructor(expected: number, actual: number) {
        super(
            `The embedder returned ${actual}d vectors but the index is ${expected}d. The model behind this id changed; press Re-index to rebuild it.`,
        );
        this.name = "DimensionMismatch";
    }
}

export function createIndexer(deps: IndexerDeps): Indexer {
    const trace = deps.trace;
    const scheduleMs = deps.scheduleMs ?? DEFAULT_SCHEDULE_MS;
    const now = deps.now ?? (() => Date.now());

    const listeners = new Set<() => void>();
    const chains = new Map<string, Promise<void>>();

    /** Only the active chat is decoded in memory; another chat is loaded on demand. */
    let cached: { chatId: string; modelId: string; index: ChatIndex } | null = null;
    let running: RunState | null = null;
    let inflight: Inflight | null = null;
    let failure: Failure | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let disposed = false;

    function notify(): void {
        for (const listener of [...listeners]) {
            try {
                listener();
            } catch {
                // A broken panel must not break indexing.
            }
        }
    }

    function fail(kind: Failure["kind"], message: string): void {
        failure = { kind, message };
        trace.log("error", "index", message, { scope: "index" });
        notify();
    }

    function clearFailure(): void {
        failure = null;
    }

    /**
     * The cache is keyed by the model that was active when it was loaded: a
     * model change must not reuse vectors from the old space, and the index on
     * disk stays untouched so status can report the mismatch.
     */
    function dropCacheFor(chatId?: string): void {
        if (!chatId || cached?.chatId === chatId) {
            cached = null;
        }
    }

    async function readIndex(chatId: string): Promise<ChatIndex | null> {
        const modelId = deps.getEmbedder().model.id;

        if (cached && cached.chatId === chatId && cached.modelId === modelId) {
            return cached.index;
        }

        const index = await loadIndex(deps.storage, chatId);
        const activeChatId = deps.getChat()?.id ?? null;

        // Loading another chat must not evict the active one: the prompt
        // injector asks for its own chat id, which can differ from the one the
        // workspace is showing while a generation is prepared.
        if (index && activeChatId === chatId) {
            cached = { chatId, modelId, index };
        }

        return index;
    }

    /**
     * Auto-index policy for the debounced path.
     *
     * Local embeddings cost nothing, so a backlog is indexed as soon as the user
     * allowed it. API embeddings cost money, so a large backlog waits for an
     * explicit "Index all" and the status shows `stale` in the meantime.
     */
    function autoAllowed(settings: PluginSettings, plan: ReconcilePlan, index: ChatIndex | null): boolean {
        if (settings.source === "local") {
            return settings.autoIndexBacklog || index !== null;
        }

        return plan.toEmbed.length <= API_AUTO_LIMIT;
    }

    function requireManualRun(settings: PluginSettings): void {
        if (!settings.enabled) {
            throw new Error("Memory is turned off. Enable it in the plugin settings before indexing.");
        }

        if (settings.source === "api" && !settings.apiNoticeAccepted) {
            throw new Error(
                "Before indexing with an API embedding source, accept the notice in the plugin settings: your chat text is sent to that endpoint.",
            );
        }
    }

    function enqueue(kind: RunKind, chatId: string): Promise<void> {
        if (disposed) {
            return Promise.resolve();
        }

        // A second request for the same chat joins the run in flight: starting a
        // parallel one would embed the same messages twice.
        if (inflight && inflight.chatId === chatId) {
            return inflight.promise;
        }

        const entry: Inflight = { chatId, promise: Promise.resolve() };
        const previous = chains.get(chatId) ?? Promise.resolve();
        const task = previous.then(() => execute(kind, chatId));

        entry.promise = task;
        inflight = entry;
        chains.set(chatId, task);

        void task
            .catch((error: unknown) => {
                // `execute` already catches its own failures; this is the last
                // guard for a fire-and-forget caller.
                trace.log("error", "index", `index: unexpected failure: ${describeError(error)}`);
            })
            .finally(() => {
                if (inflight === entry) {
                    inflight = null;
                }

                if (chains.get(chatId) === task) {
                    chains.delete(chatId);
                }
            });

        return task;
    }

    async function execute(kind: RunKind, chatId: string): Promise<void> {
        if (disposed) {
            return;
        }

        const settings = deps.getSettings();
        const embedder = deps.getEmbedder();
        const model = embedder.model;
        const chat = deps.getChat();

        // The open chat changed while this run was queued: the new chat has its
        // own trigger, and indexing the old one would write an index nobody sees.
        if (!chat || chat.id !== chatId) {
            return;
        }

        let index: ChatIndex | null;

        try {
            index = await readIndex(chatId);
        } catch (error) {
            fail("error", describeError(error));
            return;
        }

        const plan = planReconcile(index, chat.messages);

        if (kind === "reconcile" && !autoAllowed(settings, plan, index)) {
            trace.log(
                "debug",
                "index",
                `index: waiting, ${plan.toEmbed.length} messages pending and auto-indexing does not cover them`,
            );
            notify();
            return;
        }

        if (index) {
            const match = modelMatches(index, { id: model.id, dim: model.dim ?? index.model.dim });

            if (!match.ok) {
                // Never mix vectors of two models; the UI offers Re-index.
                trace.log("warn", "index", `index: skipped, ${match.reason}`);
                notify();
                return;
            }
        } else if (model.dim) {
            // Catalog models know their dimension up front. Only the manifest
            // write is deferred to the first append, so a cancelled run leaves
            // no empty index behind.
            index = createIndex(chatId, {
                id: model.id,
                source: model.source,
                dim: model.dim,
                prefixProfile: model.prefixProfile,
            });
            cached = { chatId, modelId: model.id, index };
        }

        if (index && plan.toDrop.length > 0) {
            try {
                await removeMessages(deps.storage, index, plan.toDrop);
            } catch (error) {
                fail("error", describeError(error));
                return;
            }

            trace.log("debug", "index", `drop: ${plan.toDrop.length} messages`, { ids: plan.toDrop });
        }

        if (plan.toEmbed.length === 0) {
            trace.log("debug", "index", `index: up to date (${index?.chunks.length ?? 0} chunks)`);
            notify();
            return;
        }

        const startedAt = now();
        const run: RunState = {
            chatId,
            controller: new AbortController(),
            cancelling: false,
            done: 0,
            total: plan.toEmbed.length,
            chunks: index?.chunks.length ?? 0,
            startedAt: new Date(startedAt).toISOString(),
        };

        running = run;
        // A run clears the previous failure: this is the "next trigger retries"
        // half of "do not retry in a loop".
        clearFailure();
        notify();

        let added = 0;

        try {
            added = await embedPlan(plan, index, chatId, embedder, settings, run);
        } catch (error) {
            if (isAbort(error)) {
                // Cancelling is a normal outcome: the persisted messages stay,
                // and the next trigger picks up the rest.
                trace.log("info", "index", `index: cancelled after ${run.done}/${run.total} messages`, {
                    chunks: added,
                });
            } else if (error instanceof DimensionMismatch) {
                fail("model-mismatch", error.message);
            } else {
                fail("error", describeIndexError(error));
            }
        } finally {
            running = null;
            notify();
        }

        if (added > 0) {
            trace.log(
                "info",
                "index",
                `index: ${added} chunks from ${run.done} messages in ${now() - startedAt} ms`,
                { chatId, model: model.id },
            );
        }
    }

    /**
     * Embeds the planned messages in batches of at most `EMBED_BATCH` texts and
     * appends each message as soon as all of its vectors arrived.
     */
    async function embedPlan(
        plan: ReconcilePlan,
        index: ChatIndex | null,
        chatId: string,
        embedder: Embedder,
        settings: PluginSettings,
        run: RunState,
    ): Promise<number> {
        const chunkOptions = { size: settings.chunkSize, overlap: settings.chunkOverlap };

        type Prepared = { message: ChatMessageLike; hash: string; texts: string[] };

        const prepared: Prepared[] = plan.toEmbed.map((entry) => ({
            message: entry.message,
            hash: entry.hash,
            texts: chunkText(entry.message.text, chunkOptions).map((chunk) => chunk.text),
        }));

        // A model outside the catalog has no known dimension, so the index is
        // created from the first vector instead of guessing one.
        let target: ChatIndex | null = index;
        let added = 0;

        const pendingVectors = new Map<Prepared, Float32Array[]>();

        const appendPrepared = async (item: Prepared, vectors: Float32Array[]): Promise<void> => {
            if (vectors.length > 0) {
                if (target && vectors[0].length !== target.model.dim) {
                    throw new DimensionMismatch(target.model.dim, vectors[0].length);
                }

                if (!target) {
                    const model = embedder.model;

                    target = createIndex(chatId, {
                        id: model.id,
                        source: model.source,
                        dim: vectors[0].length,
                        prefixProfile: model.prefixProfile,
                    });
                    cached = { chatId, modelId: model.id, index: target };
                }
            }

            if (!target) {
                // Nothing to write: only reachable for a blank message, which
                // reconcile never asks for. Counted as done so progress moves.
                run.done += 1;
                notify();
                return;
            }

            await appendMessageChunks(
                deps.storage,
                target,
                item.message.id,
                item.hash,
                item.texts.map((text, i) => ({ text, vector: vectors[i] })),
            );

            added += item.texts.length;
            run.done += 1;
            run.chunks = target.chunks.length;
            notify();
        };

        let batchTexts: string[] = [];
        let batchOwners: { item: Prepared; count: number }[] = [];

        const flush = async (): Promise<void> => {
            if (batchTexts.length === 0) {
                return;
            }

            const texts = batchTexts;
            const owners = batchOwners;
            batchTexts = [];
            batchOwners = [];

            const vectors = await embedder.embedDocuments(texts, run.controller.signal);

            if (vectors.length !== texts.length) {
                throw new EmbeddingError(
                    "bad-response",
                    `The embedder returned ${vectors.length} vectors for ${texts.length} texts.`,
                    "Check the embedding endpoint; it is not answering like an embeddings API.",
                );
            }

            let cursor = 0;

            for (const owner of owners) {
                const slice = vectors.slice(cursor, cursor + owner.count);
                cursor += owner.count;

                const collected = pendingVectors.get(owner.item) ?? [];
                collected.push(...slice);
                pendingVectors.set(owner.item, collected);

                // Appending waits for the whole message: one storage write per
                // message keeps a cancelled run consistent.
                if (collected.length === owner.item.texts.length) {
                    pendingVectors.delete(owner.item);
                    await appendPrepared(owner.item, collected);
                }
            }
        };

        for (const item of prepared) {
            if (run.controller.signal.aborted) {
                throw abortError();
            }

            let offset = 0;

            while (offset < item.texts.length) {
                const room = EMBED_BATCH - batchTexts.length;
                const take = Math.min(room, item.texts.length - offset);

                batchTexts.push(...item.texts.slice(offset, offset + take));
                batchOwners.push({ item, count: take });
                offset += take;

                if (batchTexts.length >= EMBED_BATCH) {
                    await flush();
                }
            }

            if (item.texts.length === 0) {
                await appendPrepared(item, []);
            }
        }

        await flush();

        return added;
    }

    function progress(): IndexerProgress {
        if (!running) {
            return {
                running: false,
                cancelling: false,
                chatId: inflight?.chatId ?? null,
                done: 0,
                total: 0,
                chunks: 0,
            };
        }

        return {
            running: true,
            cancelling: running.cancelling,
            chatId: running.chatId,
            done: running.done,
            total: running.total,
            chunks: running.chunks,
            startedAt: running.startedAt,
        };
    }

    async function status(activeModelId: string): Promise<IndexStatus> {
        const chat = deps.getChat();
        const settings = deps.getSettings();
        const embedderModel = deps.getEmbedder().model;

        if (!chat) {
            return {
                chatId: null,
                state: "no-chat",
                chunks: 0,
                messagesIndexed: 0,
                messagesTotal: 0,
                messagesPending: 0,
                activeModelId,
            };
        }

        const messagesTotal = chat.messages.filter((message) => message.text.trim().length > 0).length;

        let index: ChatIndex | null = null;
        let loadError: string | undefined;

        try {
            index = await readIndex(chat.id);
        } catch (error) {
            loadError = describeError(error);
        }

        const plan = planReconcile(index, chat.messages);
        const pending = plan.toEmbed.length;

        let mismatch: string | undefined;

        if (index) {
            const match = modelMatches(index, {
                id: activeModelId,
                // An unknown dimension can only be compared with the index's own.
                dim: embedderModel.dim ?? index.model.dim,
            });

            mismatch = match.ok ? undefined : match.reason;
        }

        const state = deriveState(settings, {
            running: running !== null,
            loadError,
            mismatch,
            failure,
            index,
            pending,
        });

        const status: IndexStatus = {
            chatId: chat.id,
            state,
            chunks: index?.chunks.length ?? 0,
            messagesIndexed: index ? Object.keys(index.messages).length : 0,
            messagesTotal,
            messagesPending: pending,
            activeModelId,
        };

        if (index) {
            status.indexModelId = index.model.id;
        }

        const detail = detailFor(state, { mismatch, pending, lastError: failure?.message ?? loadError });

        if (detail) {
            status.detail = detail;
        }

        const lastError = failure?.message ?? loadError;

        if (lastError) {
            status.lastError = lastError;
        }

        return status;
    }

    async function purge(): Promise<void> {
        const chat = deps.getChat();

        if (!chat) {
            return;
        }

        // A run still appending would resurrect the index right after the purge,
        // so it is cancelled and awaited first.
        if (running && running.chatId === chat.id) {
            running.controller.abort();
        }

        if (inflight && inflight.chatId === chat.id) {
            await inflight.promise.catch(() => undefined);
        }

        await purgeIndex(deps.storage, chat.id);
        dropCacheFor(chat.id);
        clearFailure();
        trace.log("info", "index", `index: purged ${chat.id}`);
        notify();
    }

    async function indexAll(): Promise<void> {
        requireManualRun(deps.getSettings());

        const chat = deps.getChat();

        if (!chat) {
            throw new Error("Open a chat before indexing.");
        }

        await enqueue("all", chat.id);
    }

    async function reindex(): Promise<void> {
        requireManualRun(deps.getSettings());

        const chat = deps.getChat();

        if (!chat) {
            throw new Error("Open a chat before re-indexing.");
        }

        await purge();
        await enqueue("all", chat.id);
    }

    async function reconcileNow(): Promise<void> {
        const settings = deps.getSettings();
        const chat = deps.getChat();

        if (!settings.enabled || !chat) {
            return;
        }

        // Nothing may be sent to an endpoint the user has not been told about.
        if (settings.source === "api" && !settings.apiNoticeAccepted) {
            return;
        }

        await enqueue("reconcile", chat.id);
    }

    async function runScheduled(): Promise<void> {
        try {
            const settings = deps.getSettings();

            if (!settings.enabled || !deps.getChat()) {
                return;
            }

            if (settings.source === "api" && !settings.apiNoticeAccepted) {
                return;
            }

            await reconcileNow();
        } catch (error) {
            // Fire-and-forget: an unhandled rejection here would surface as a
            // console error nobody can act on.
            trace.log("error", "index", `index: scheduled run failed: ${describeError(error)}`);
        }
    }

    return {
        reconcileNow,

        indexAll,

        reindex,

        purge,

        schedule() {
            if (disposed) {
                return;
            }

            if (timer !== undefined) {
                clearTimeout(timer);
            }

            timer = setTimeout(() => {
                timer = undefined;
                void runScheduled();
            }, scheduleMs);
        },

        cancel() {
            if (!running) {
                return;
            }

            running.cancelling = true;
            running.controller.abort();
            notify();
        },

        status,

        progress,

        readIndex,

        onChange(listener) {
            listeners.add(listener);

            return () => {
                listeners.delete(listener);
            };
        },

        dispose() {
            disposed = true;

            if (timer !== undefined) {
                clearTimeout(timer);
                timer = undefined;
            }

            if (running) {
                running.controller.abort();
            }

            listeners.clear();
        },
    };
}

/** State precedence is the one the contract lists, top to bottom. */
function deriveState(
    settings: PluginSettings,
    input: {
        running: boolean;
        loadError?: string;
        mismatch?: string;
        failure: Failure | null;
        index: ChatIndex | null;
        pending: number;
    },
): IndexState {
    if (!settings.enabled) {
        return "disabled";
    }

    if (settings.source === "api" && !settings.apiNoticeAccepted) {
        return "needs-consent";
    }

    if (input.running) {
        return "indexing";
    }

    if (input.failure) {
        // An embedding failure or a dimension surprise is sticky until the next
        // trigger clears it; without this the chat would look up to date while
        // half of it is missing.
        return input.failure.kind;
    }

    if (input.loadError) {
        return "error";
    }

    if (input.mismatch) {
        return "model-mismatch";
    }

    // `stale` wins over `empty`: a chat whose index does not exist but has
    // messages waiting is exactly what "Index all" is for, and it is also the
    // state an API backlog waits in.
    if (input.pending > 0) {
        return "stale";
    }

    if (!input.index || input.index.chunks.length === 0) {
        return "empty";
    }

    return "up-to-date";
}

function detailFor(
    state: IndexState,
    input: { mismatch?: string; pending: number; lastError?: string },
): string | undefined {
    if (state === "model-mismatch") {
        return input.mismatch ?? input.lastError;
    }

    if (state === "stale") {
        return `${input.pending} messages waiting to be indexed`;
    }

    if (state === "error") {
        return input.lastError;
    }

    return undefined;
}

function abortError(): EmbeddingError {
    return new EmbeddingError("aborted", "The embedding request was cancelled.", "Nothing to do; the run was cancelled.");
}

function isAbort(error: unknown): boolean {
    if (error instanceof EmbeddingError) {
        return error.code === "aborted";
    }

    return error instanceof Error && error.name === "AbortError";
}

/** `EmbeddingError.hint` is the part the user can act on, so it is kept. */
function describeIndexError(error: unknown): string {
    if (error instanceof EmbeddingError) {
        return error.hint ? `${error.message} ${error.hint}` : error.message;
    }

    return describeError(error);
}

function describeError(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

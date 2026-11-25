// The seam between the plugin's logic (index.tsx, indexer.ts, host-client.ts) and its UI
// (settings-panel.tsx, modal.tsx). The UI only talks to a MemoryController; the logic side builds
// exactly one in activate(). Types only: keep this file free of runtime code.

import type { Embedder } from "./embedder";
import type { RetrievalResult } from "./retrieval";
import type { PluginSettings } from "./settings";
import type { Trace } from "./trace";
import type { InstallerStatus, ProcessStatus, ServerConfig } from "./server/types";

/** Payload of GET /status on the server module. */
export type ServerStatus = {
    process: ProcessStatus;
    installer: InstallerStatus;
    config: ServerConfig;
    platform: { os: string; arch: string };
};

/**
 * Thin typed wrapper over `api.host.fetch`. Every method rejects with an Error whose message is
 * the server's `{ error: { message } }` text, or an actionable "approve the plugin" message when
 * the host call itself fails (plugin not approved / server module not running).
 *
 * Routes (all relative to /api/plugins/smiley-embeddings-memory/host):
 *   GET /status, POST /config, POST /start, POST /stop,
 *   POST /binary/install, POST /binary/cancel, DELETE /binary,
 *   POST /models/download, POST /models/cancel, DELETE /models, POST /embed
 */
export type HostClient = {
    status(): Promise<ServerStatus>;
    /** Persists the server-relevant subset of the settings (see serverConfigOf). Returns the saved config. */
    saveConfig(config: Record<string, unknown>): Promise<ServerConfig>;
    /** Starts llama-server now (warm-up) and resolves with its status once ready; rejects with the server's message on failure. */
    start(): Promise<ProcessStatus>;
    stop(): Promise<void>;
    installBinary(): Promise<void>;
    cancelBinary(): Promise<void>;
    removeBinary(): Promise<void>;
    downloadModel(): Promise<void>;
    cancelModel(): Promise<void>;
    removeModels(): Promise<void>;
};

export type IndexState =
    | "no-chat" // no chat is open
    | "disabled" // the memory switch is off
    | "needs-consent" // API source and the user has not accepted the privacy notice yet
    | "empty" // nothing indexed yet
    | "indexing"
    | "up-to-date"
    | "stale" // there are messages not indexed yet, or edited ones
    | "model-mismatch" // the index was built with another model
    | "error";

export type IndexStatus = {
    chatId: string | null;
    state: IndexState;
    chunks: number;
    messagesIndexed: number;
    /** Messages with indexable text in the open chat. */
    messagesTotal: number;
    /** Messages that reconcile would embed now (new + edited). */
    messagesPending: number;
    /** Model id stored in the index, when there is one. */
    indexModelId?: string;
    /** Active model id from the current settings. */
    activeModelId: string;
    detail?: string;
    lastError?: string;
};

export type IndexerProgress = {
    running: boolean;
    cancelling: boolean;
    chatId: string | null;
    /** Messages embedded so far / to embed in this run. */
    done: number;
    total: number;
    chunks: number;
    startedAt?: string;
};

export type MemoryController = {
    // settings
    settings(): PluginSettings;
    updateSettings(patch: Partial<PluginSettings>): void;
    subscribe(listener: () => void): () => void; // fires on settings, index, indexer, trace and server changes

    // observability
    trace: Trace;
    host: HostClient;
    embedder(): Embedder;

    // current chat
    activeChatId(): string | null;
    /** The composer draft when the app snapshot exposes it, otherwise null (the viewer then shows a text box). */
    currentDraft(): string | null;

    // indexing
    indexStatus(): Promise<IndexStatus>;
    progress(): IndexerProgress;
    /** Index every message of the open chat that is not indexed yet. Resolves when finished or cancelled. */
    indexAll(): Promise<void>;
    /** Wipe the open chat's index and rebuild it with the active model. */
    reindex(): Promise<void>;
    cancelIndexing(): void;
    /** Delete the open chat's index. */
    purge(): Promise<void>;

    // probes
    /** Embeds a short text with the active source; rejects with EmbeddingError (message + hint). */
    test(): Promise<{ dim: number; latencyMs: number }>;
    /** What would be retrieved right now for `draft`, without sending anything or touching the last-injection record. */
    dryRun(draft: string): Promise<RetrievalResult>;
};

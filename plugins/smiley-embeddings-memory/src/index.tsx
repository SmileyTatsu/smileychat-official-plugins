/**
 * Plugin entry point: the browser half of Smiley Embeddings Memory.
 *
 * `activate()` builds one `MemoryController` and wires the app's extension points
 * to it: settings (Options), the settings panel, the viewer modal, the prompt
 * injector, the message-update middleware, the state subscription that drives
 * background indexing, and the stylesheet. Everything it creates is torn down by
 * the returned cleanup function.
 *
 * Nothing here is allowed to break a generation: the injector never throws and
 * indexing is fire-and-forget.
 */

import type { PluginAppSnapshot, SmileyPluginApi } from "../types/smileychat-plugin-api";

import type { MemoryController } from "./controller";
import { EmbeddingError, createEmbedder, testEmbedding } from "./embedder";
import type { Embedder } from "./embedder";
import { createHostClient } from "./host-client";
import { sweepOrphans } from "./index-store";
import { createIndexer } from "./indexer";
import { buildInjection, runWithTimeout } from "./injection";
import { chatMessagesForIndex, retrievalMessages } from "./messages";
import { MemoryModal } from "./modal";
import { emptyRetrievalResult, retrieveMemories } from "./retrieval";
import type { RetrievalResult } from "./retrieval";
import { DEFAULT_SETTINGS, SETTINGS_KEY, activeModelOf, normalizeSettings, serverConfigOf } from "./settings";
import type { PluginSettings } from "./settings";
import { MemorySettingsPanel } from "./settings-panel";
// `with { type: "text" }` is what hands over the stylesheet itself; a plain CSS
// import would make Bun emit a separate asset and give `styles` its module
// object instead of the text `api.ui.addStyles` needs.
import styles from "./styles.css" with { type: "text" };
import { createTrace } from "./trace";
import type { Trace, TraceLevel, TraceScope } from "./trace";

/** Delay before the orphan sweep, so it never competes with activation. */
const SWEEP_DELAY_MS = 3000;

/** Settings writes are debounced: the wizard changes several fields in a row. */
const CONFIG_PUSH_DELAY_MS = 800;

export async function activate(api: SmileyPluginApi) {
    const trace = createMirroredTrace(api);

    // The embedding API key lives in these settings, which the app stores as
    // plain local JSON (`api.settings`). `connections:secrets` is not used: this
    // key is not a chat-provider key, and the panel says so.
    const handle = await api.settings.register<PluginSettings>({
        key: SETTINGS_KEY,
        defaultValues: DEFAULT_SETTINGS,
        normalize: normalizeSettings,
    });

    let settings = normalizeSettings(handle.get() ?? DEFAULT_SETTINGS);
    let embedder: Embedder = createEmbedder(settings, { hostFetch: api.host.fetch });
    let lastServerConfig = JSON.stringify(serverConfigOf(settings));
    let serverTimer: ReturnType<typeof setTimeout> | undefined;
    let disposed = false;

    const host = createHostClient(api.host.fetch);

    const listeners = new Set<() => void>();
    let notifyQueued = false;

    /** One notification per burst: indexing emits a change per message. */
    function notify(): void {
        if (notifyQueued) {
            return;
        }

        notifyQueued = true;

        queueMicrotask(() => {
            notifyQueued = false;

            for (const listener of [...listeners]) {
                try {
                    listener();
                } catch {
                    // A broken panel must not stop the others.
                }
            }
        });
    }

    /** The active chat as the indexer sees it (visible swipe, cleaned text). */
    function activeChat() {
        const snapshot = api.state.getSnapshot();
        const chat = snapshot?.activeChat;

        if (!snapshot || !chat) {
            return null;
        }

        return { id: chat.id, messages: chatMessagesForIndex(snapshot.messages) };
    }

    const indexer = createIndexer({
        storage: api.storage,
        getSettings: () => settings,
        getEmbedder: () => embedder,
        getChat: activeChat,
        trace,
    });

    /**
     * Rebuilds the embedder and pushes the server-relevant subset of the
     * settings. The push only happens for the local source: an API-only setup
     * has no llama-server to configure, and an unapproved plugin would only
     * produce noise.
     */
    function applySettings(): void {
        embedder = createEmbedder(settings, { hostFetch: api.host.fetch });

        const next = JSON.stringify(serverConfigOf(settings));

        if (settings.source !== "local" || next === lastServerConfig) {
            return;
        }

        lastServerConfig = next;

        if (serverTimer !== undefined) {
            clearTimeout(serverTimer);
        }

        serverTimer = setTimeout(() => {
            serverTimer = undefined;

            if (disposed) {
                return;
            }

            void host
                .saveConfig(serverConfigOf(settings))
                .then(() => trace.log("debug", "server", "config: saved"))
                // Debug, not error: the plugin may simply not be approved yet,
                // and the panel shows that state on its own.
                .catch((error: unknown) => {
                    trace.log("debug", "server", `config: not saved (${describe(error)})`);
                });
        }, CONFIG_PUSH_DELAY_MS);
    }

    /**
     * `api.model.estimateTokens` only accepts a message array and requires the
     * `model:generate` permission this plugin does not declare, so the call is
     * guarded and a plain character estimate keeps the token cap meaningful.
     */
    function estimateTokens(text: string): number {
        try {
            return api.model.estimateTokens([{ role: "system", content: text }]);
        } catch {
            return Math.ceil(text.length / 4);
        }
    }

    /**
     * One retrieval, bounded by the settings timeout, logged either way.
     * `remember` is false for a dry run: it must not replace the record of what
     * the last real turn injected.
     */
    async function retrieve(
        chatId: string,
        messages: { id: string; text: string }[],
        draft: string | undefined,
        remember: boolean,
    ): Promise<RetrievalResult> {
        const current = settings;

        const result = await runWithTimeout(async (signal) => {
            // A message that arrived after the last state change is indexed for
            // the next turn, not after this one finishes.
            indexer.schedule();

            // A corrupt or missing index is "no match", never a failed prompt build.
            const index = await indexer.readIndex(chatId).catch(() => null);

            return retrieveMemories({
                index,
                messages,
                settings: current,
                embedder,
                estimateTokens,
                signal,
                ...(draft === undefined ? {} : { draft }),
            });
        }, current.timeoutMs);

        trace.log("info", "retrieval", result.line);

        if (remember) {
            trace.setLastInjection({
                at: new Date().toISOString(),
                chatId,
                result,
                dryRun: false,
            });
        }

        return result;
    }

    const controller: MemoryController = {
        settings: () => settings,

        updateSettings(patch) {
            // Optimistic so the controls respond immediately; the handle echoes
            // the normalized value back through its subscription.
            settings = normalizeSettings({ ...settings, ...patch });
            applySettings();
            notify();

            void handle
                .set(settings)
                .then((saved) => {
                    settings = normalizeSettings(saved);
                    applySettings();
                    notify();
                })
                .catch((error: unknown) => {
                    trace.log("error", "ui", `settings: ${describe(error)}`);
                });
        },

        subscribe(listener) {
            listeners.add(listener);

            return () => {
                listeners.delete(listener);
            };
        },

        trace,
        host,
        embedder: () => embedder,

        activeChatId: () => api.state.getSnapshot()?.activeChat?.id ?? null,

        // The snapshot carries no composer draft (`PluginAppSnapshot` has no such
        // field), so the viewer shows its own text box for the dry run.
        currentDraft: () => null,

        indexStatus: () => indexer.status(activeModelOf(settings).id),
        progress: () => indexer.progress(),
        indexAll: () => indexer.indexAll(),
        reindex: () => indexer.reindex(),
        cancelIndexing: () => indexer.cancel(),
        purge: () => indexer.purge(),

        async test() {
            try {
                const result = await testEmbedding(embedder);
                trace.log("info", "embed", `test: dim=${result.dim} in ${result.latencyMs} ms`);

                return result;
            } catch (error) {
                // Re-thrown for the panel to show; the log keeps the hint
                // next to the measurement it belongs to.
                trace.log("error", "embed", `test failed: ${describeEmbeddingError(error)}`);
                throw error;
            }
        },

        async dryRun(draft) {
            const snapshot = api.state.getSnapshot();
            const chat = snapshot?.activeChat;

            if (!snapshot || !chat) {
                trace.log("warn", "retrieval", "dry run: no active chat");

                return emptyRetrievalResult("no-queries", "no match: no active chat");
            }

            return retrieve(chat.id, retrievalMessages(snapshot.messages), draft, false);
        },
    };

    // Indexing trigger: chat switches, new messages, deletions. Streaming turns
    // rewrite the last swipe's text without changing the message list, so the
    // signature ignores them and no run is queued per token.
    let chatSignature = "";

    const unsubscribeState = api.state.subscribe((snapshot) => {
        const signature = snapshotSignature(snapshot);

        if (signature === chatSignature) {
            return;
        }

        chatSignature = signature;
        indexer.schedule();
        notify();
    });

    // Edits, swipes and regenerations keep the message id, so they only reach us
    // here. Synchronous and returning `undefined`: a returned message would
    // replace the one the app is saving.
    api.chat.registerMessageUpdateMiddleware(() => {
        indexer.schedule();

        return undefined;
    });

    api.chat.registerPromptInjector(async (context) => {
        try {
            const current = settings;

            if (!current.enabled) {
                return [];
            }

            indexer.schedule();

            const result = await retrieve(
                context.chat.id,
                retrievalMessages(context.messages),
                undefined,
                true,
            );
            const injection = buildInjection(result, current);

            return injection ? [injection] : [];
        } catch (error) {
            // The app aborts the whole prompt build on a rejection, so a broken
            // injection becomes an empty one plus a log line.
            trace.log("error", "retrieval", `injector failed: ${describe(error)}`);

            return [];
        }
    });

    // Deleting a chat leaves its index behind (the app has no delete event), so
    // the registry is swept against the real chat list once the app is settled.
    const sweepTimer = setTimeout(() => {
        void sweepIndexes();
    }, SWEEP_DELAY_MS);

    async function sweepIndexes(): Promise<void> {
        try {
            const chats = await api.actions.listChats();

            // An empty list is "the listing failed", not "every chat is gone";
            // sweeping then would delete indexes that cost money to rebuild.
            if (chats.length === 0) {
                return;
            }

            const removed = await sweepOrphans(api.storage, new Set(chats.map((chat) => chat.id)));

            if (removed.length > 0) {
                trace.log("info", "index", `index: removed ${removed.length} orphaned indexes`, { ids: removed });
            }
        } catch (error) {
            trace.log("debug", "index", `index: orphan sweep skipped (${describe(error)})`);
        }
    }

    api.ui.addStyles(styles);

    api.ui.registerSettingsPanel({
        id: "settings",
        label: "Semantic Memory",
        settingsKey: SETTINGS_KEY,
        render: () => <MemorySettingsPanel controller={controller} />,
    });

    api.ui.registerHeaderAction({
        id: "open-memory",
        label: "Memory",
        // Inline SVG: `lucide-preact` is not a dependency of this plugin, and one
        // icon is not worth adding it.
        renderIcon: () => (
            <svg
                viewBox="0 0 24 24"
                width="17"
                height="17"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
            >
                <rect x="7" y="7" width="10" height="10" rx="2" />
                <path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4" />
            </svg>
        ),
        run() {
            api.ui.openModal({
                id: "memory",
                title: "Semantic Memory",
                render: ({ close }) => <MemoryModal controller={controller} onClose={close} />,
            });
        },
    });

    // Settings, index/indexer and trace changes all reach the panels through the
    // controller; server state reaches them through the trace lines the host
    // calls write, so there is no separate server subscription.
    const unsubscribeSettings = handle.subscribe((next) => {
        settings = normalizeSettings(next);
        applySettings();
        notify();
    });
    const unsubscribeIndexer = indexer.onChange(notify);
    const unsubscribeTrace = trace.onChange(notify);

    return () => {
        disposed = true;
        clearTimeout(sweepTimer);

        if (serverTimer !== undefined) {
            clearTimeout(serverTimer);
            serverTimer = undefined;
        }

        unsubscribeState();
        unsubscribeSettings();
        unsubscribeIndexer();
        unsubscribeTrace();
        // Aborts an in-flight embedding request and clears the debounce timer.
        indexer.dispose();
        listeners.clear();
    };
}

/**
 * The app's logger is the only record that survives a reload, so every trace
 * entry is mirrored at the same level. The in-memory buffer stays the source of
 * truth for the viewer.
 */
function createMirroredTrace(api: SmileyPluginApi): Trace {
    const inner = createTrace();

    return {
        log(level: TraceLevel, scope: TraceScope, message: string, detail?: unknown) {
            inner.log(level, scope, message, detail);

            const line = `[${scope}] ${message}`;
            const record = asDetail(detail);

            if (level === "error") {
                api.logger.error(line, detail);
            } else if (record === undefined) {
                api.logger[level](line);
            } else {
                api.logger[level](line, record);
            }
        },
        logs: () => inner.logs(),
        onChange: (listener) => inner.onChange(listener),
        setLastInjection: (injection) => inner.setLastInjection(injection),
        lastInjection: () => inner.lastInjection(),
        clear: () => inner.clear(),
    };
}

/** `api.logger` takes a record; any other detail is wrapped instead of dropped. */
function asDetail(detail: unknown): Record<string, unknown> | undefined {
    if (detail === undefined) {
        return undefined;
    }

    if (typeof detail === "object" && detail !== null && !Array.isArray(detail)) {
        return detail as Record<string, unknown>;
    }

    return { detail };
}

/**
 * Identity of the visible conversation for the state subscription: the chat and
 * the message ids. Text is deliberately excluded so a streaming turn does not
 * queue an index run per token; edits arrive through the update middleware.
 */
function snapshotSignature(snapshot: PluginAppSnapshot): string {
    const chatId = snapshot.activeChat?.id ?? "";

    return `${chatId}|${snapshot.messages.map((message) => message.id).join(",")}`;
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

/** An `EmbeddingError`'s hint is the actionable half, so it is kept in the log. */
function describeEmbeddingError(error: unknown): string {
    if (error instanceof EmbeddingError) {
        return error.hint ? `${error.message} ${error.hint}` : error.message;
    }

    return describe(error);
}

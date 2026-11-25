/**
 * In-memory trace buffer behind the plugin's log viewer: a bounded ring of
 * recent events plus the last injection, with change notifications for Preact.
 *
 * The buffer exists because the app's logger only reaches the console and the
 * log file; a user debugging "why did it inject nothing this turn" needs the
 * last reasons next to the chat, without opening a terminal.
 */

import type { RetrievalResult } from "./retrieval";

export type TraceLevel = "debug" | "info" | "warn" | "error";

export type TraceScope = "index" | "retrieval" | "server" | "embed" | "ui";

export type TraceEntry = {
    at: string;
    level: TraceLevel;
    scope: TraceScope;
    message: string;
    detail?: unknown;
};

export type LastInjection = {
    at: string;
    chatId: string;
    result: RetrievalResult;
    dryRun: boolean;
};

export type Trace = {
    log(level: TraceLevel, scope: TraceScope, message: string, detail?: unknown): void;
    /** Oldest first, so the viewer can render top to bottom. */
    logs(): TraceEntry[];
    onChange(listener: () => void): () => void;
    setLastInjection(injection: LastInjection): void;
    lastInjection(): LastInjection | null;
    clear(): void;
};

/** Enough for a turn-by-turn debugging session without growing a chat's memory. */
const DEFAULT_CAPACITY = 200;

export function createTrace(capacity = DEFAULT_CAPACITY): Trace {
    const limit = Number.isFinite(capacity) ? Math.max(1, Math.floor(capacity)) : DEFAULT_CAPACITY;

    const entries: TraceEntry[] = [];
    const listeners = new Set<() => void>();
    let injection: LastInjection | null = null;
    let flushScheduled = false;

    /**
     * One notification per burst: a reconcile run logs a line per message, and a
     * re-render per line would make the viewer slower than the indexing.
     */
    const notify = () => {
        if (flushScheduled) {
            return;
        }

        flushScheduled = true;
        queueMicrotask(() => {
            flushScheduled = false;

            for (const listener of [...listeners]) {
                try {
                    listener();
                } catch {
                    // A broken listener must not stop the others or break logging.
                }
            }
        });
    };

    return {
        log(level, scope, message, detail) {
            entries.push({
                at: new Date().toISOString(),
                level,
                scope,
                message,
                // Kept only when given, so entries stay cheap to inspect.
                ...(detail === undefined ? {} : { detail }),
            });

            if (entries.length > limit) {
                entries.splice(0, entries.length - limit);
            }

            notify();
        },
        logs() {
            // A copy: the viewer sorts/filters freely without touching the buffer.
            return entries.slice();
        },
        onChange(listener) {
            listeners.add(listener);

            return () => {
                listeners.delete(listener);
            };
        },
        setLastInjection(next) {
            injection = next;
            notify();
        },
        lastInjection() {
            return injection;
        },
        clear() {
            // The last injection is kept: it is the answer to "what did the last
            // turn inject", which stays useful while the log is cleared.
            entries.length = 0;
            notify();
        },
    };
}

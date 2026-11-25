/**
 * Embedding clients: an OpenAI-compatible HTTP endpoint, or the plugin's own
 * server module through `api.host` (which owns the local `llama-server`).
 *
 * Both paths do the same three things: apply the model's text prefixes, send
 * batches of 16, and return unit-length Float32Arrays ordered like the input.
 */

import { formatDocument, formatQuery } from "./models";
import { activeModelOf } from "./settings";
import type { EmbeddingSource, PluginSettings } from "./settings";
import type { PrefixProfile } from "./models";
import { normalize } from "./vectors";

/** `api.host.fetch`: only this plugin's server module is reachable through it. */
export type HostFetch = (path: string, init?: RequestInit) => Promise<Response>;

export type ActiveModel = {
    id: string;
    source: EmbeddingSource;
    prefixProfile: PrefixProfile;
    /** Known only for catalog models; custom and manual files stay unknown. */
    dim?: number;
};

export type EmbeddingErrorCode =
    | "auth"
    | "unreachable"
    | "not-found"
    | "rate-limit"
    | "server"
    | "bad-response"
    | "timeout"
    | "aborted"
    | "too-long"
    | "not-ready"
    | "not-installed";

/**
 * `message` says what happened, `hint` says what the user can do about it; the
 * settings panel and the log show both, and a bare status line would leave the
 * user with nothing to try.
 */
export class EmbeddingError extends Error {
    code: EmbeddingErrorCode;
    hint: string;

    constructor(code: EmbeddingErrorCode, message: string, hint: string) {
        super(message);
        this.name = "EmbeddingError";
        this.code = code;
        this.hint = hint;
    }
}

export type Embedder = {
    model: ActiveModel;
    embedDocuments(texts: string[], signal?: AbortSignal): Promise<Float32Array[]>;
    embedQueries(texts: string[], signal?: AbortSignal): Promise<Float32Array[]>;
};

export type EmbedderDeps = {
    hostFetch: HostFetch;
    fetchImpl?: typeof fetch;
    /** Injected by tests so retry backoff does not sleep for real. */
    delay?: (ms: number) => Promise<void>;
};

/** Small enough that a slow CPU model answers before the request times out. */
const BATCH_SIZE = 16;
const MAX_ATTEMPTS = 3;
const BACKOFF_BASE_MS = 400;
const REQUEST_TIMEOUT_MS = 60_000;
const PROBE_TEXT = "The quick brown fox jumps over the lazy dog.";

export function createEmbedder(settings: PluginSettings, deps: EmbedderDeps): Embedder {
    const model = activeModelOf(settings);
    const fetchImpl = deps.fetchImpl ?? fetch;
    const wait = deps.delay ?? sleep;

    const embed = async (
        texts: string[],
        kind: "document" | "query",
        signal?: AbortSignal,
    ): Promise<Float32Array[]> => {
        if (texts.length === 0) {
            return [];
        }

        const formatted = texts.map((text) => formatForModel(text, kind, model.prefixProfile, settings.qwenTask));
        const batches = splitBatches(formatted);
        const out: Float32Array[] = [];

        // Sequential on purpose: the local server processes one batch at a time
        // anyway, and a parallel burst on a paid API is harder to stop.
        for (const batch of batches) {
            const vectors =
                model.source === "api"
                    ? await embedApiBatch(batch, settings, fetchImpl, wait, signal)
                    : await embedLocalBatch(batch, deps.hostFetch, signal);

            out.push(...vectors);
        }

        return out;
    };

    return {
        model,
        embedDocuments: (texts, signal) => embed(texts, "document", signal),
        embedQueries: (texts, signal) => embed(texts, "query", signal),
    };
}

/** Embeds one sentence so the settings panel can show dimension and latency. */
export async function testEmbedding(embedder: Embedder): Promise<{ dim: number; latencyMs: number }> {
    const started = performance.now();
    const vectors = await embedder.embedDocuments([PROBE_TEXT]);
    const latencyMs = Math.round(performance.now() - started);

    if (vectors.length !== 1) {
        throw new EmbeddingError(
            "bad-response",
            "The endpoint returned no vector for the test sentence.",
            "Check the base URL and the model name; the endpoint is not answering like an embeddings API.",
        );
    }

    return { dim: vectors[0].length, latencyMs };
}

function formatForModel(text: string, kind: "document" | "query", profile: PrefixProfile, task: string) {
    // `none` is a no-op in both formatters, so the "only when not none" rule of
    // the design is expressed by the profile itself.
    return kind === "document" ? formatDocument(text, profile) : formatQuery(text, profile, task);
}

function splitBatches<T>(items: T[]): T[][] {
    const batches: T[][] = [];

    for (let start = 0; start < items.length; start += BATCH_SIZE) {
        batches.push(items.slice(start, start + BATCH_SIZE));
    }

    return batches;
}

async function embedApiBatch(
    input: string[],
    settings: PluginSettings,
    fetchImpl: typeof fetch,
    wait: (ms: number) => Promise<void>,
    signal?: AbortSignal,
): Promise<Float32Array[]> {
    const url = `${settings.api.baseUrl}/embeddings`;
    const headers: Record<string, string> = { "content-type": "application/json" };

    // An empty key is normal for a local OpenAI-compatible server, and sending
    // `Bearer ` there is rejected by some of them.
    if (settings.api.apiKey) {
        headers.authorization = `Bearer ${settings.api.apiKey}`;
    }

    const body = JSON.stringify({ model: settings.api.model, input, encoding_format: "float" });

    let lastError: EmbeddingError | undefined;

    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
        if (signal?.aborted) {
            throw abortedError();
        }

        try {
            const response = await requestWithDeadline(url, body, headers, fetchImpl, signal);

            if (!response.ok) {
                throw errorForStatus(response.status, await safeText(response), url);
            }

            return validateVectors(parseApiBody(await safeJson(response)));
        } catch (error) {
            const mapped = asEmbeddingError(error);

            // A wrong key or a wrong URL answers the same way every time, so only
            // transient failures are repeated.
            if (!isRetryable(mapped) || attempt === MAX_ATTEMPTS) {
                throw mapped;
            }

            lastError = mapped;
            await wait(BACKOFF_BASE_MS * 2 ** (attempt - 1));
        }
    }

    throw lastError ?? new EmbeddingError("server", `The embedding endpoint at ${url} kept failing.`, "Try again later.");
}

async function requestWithDeadline(
    url: string,
    body: string,
    headers: Record<string, string>,
    fetchImpl: typeof fetch,
    signal?: AbortSignal,
): Promise<Response> {
    const controller = new AbortController();
    let timedOut = false;

    const onAbort = () => controller.abort();

    if (signal) {
        if (signal.aborted) {
            controller.abort();
        } else {
            signal.addEventListener("abort", onAbort, { once: true });
        }
    }

    const timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
    }, REQUEST_TIMEOUT_MS);

    try {
        return await fetchImpl(url, { method: "POST", headers, body, signal: controller.signal });
    } catch (error) {
        if (timedOut) {
            throw new EmbeddingError(
                "timeout",
                `The embedding endpoint did not answer within ${REQUEST_TIMEOUT_MS / 1000} s.`,
                "Try again, or use a local endpoint: a slow remote server delays every message.",
            );
        }

        if (signal?.aborted) {
            throw abortedError();
        }

        // `fetch` rejects with a TypeError for a refused connection *and* for a
        // CORS rejection; the browser does not tell them apart, so the hint covers both.
        throw new EmbeddingError(
            "unreachable",
            `Could not reach the embedding endpoint at ${url}.`,
            "If the endpoint is remote, it must allow this page's origin (CORS). If it is local, start it and allow this page's origin.",
        );
    } finally {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
    }
}

function parseApiBody(raw: unknown): number[][] {
    const data = isRecord(raw) && Array.isArray(raw.data) ? raw.data : null;

    if (!data) {
        throw new EmbeddingError(
            "bad-response",
            "The endpoint replied with something that is not an embeddings response.",
            "Check the base URL (it usually ends in /v1) and that the model name is an embedding model.",
        );
    }

    const ordered: number[][] = new Array(data.length);

    for (const entry of data) {
        const index = isRecord(entry) && typeof entry.index === "number" ? entry.index : -1;
        const embedding = isRecord(entry) && Array.isArray(entry.embedding) ? entry.embedding : null;

        if (!embedding || !Number.isInteger(index) || index < 0 || index >= data.length || ordered[index]) {
            throw new EmbeddingError(
                "bad-response",
                "The endpoint returned vectors without usable indexes.",
                "Check the base URL and the model name; the endpoint is not answering like an OpenAI-compatible API.",
            );
        }

        ordered[index] = embedding.map((value) => Number(value));
    }

    return ordered;
}

/**
 * Shared shape check for both sources: every vector the same size, finite, and
 * normalised, so a later dot product is already a cosine.
 */
function validateVectors(rows: number[][]): Float32Array[] {
    if (rows.length === 0) {
        throw new EmbeddingError(
            "bad-response",
            "The endpoint returned no vectors.",
            "Check the model name; some providers return an empty list for an unknown model.",
        );
    }

    const dim = rows[0].length;

    if (dim === 0) {
        throw new EmbeddingError(
            "bad-response",
            "The endpoint returned empty vectors.",
            "Check the model name; an embedding model must return at least one number per text.",
        );
    }

    const out = rows.map((row) => {
        if (row.length !== dim) {
            throw new EmbeddingError(
                "bad-response",
                `The endpoint returned vectors of different sizes (${dim} and ${row.length}).`,
                "Check the model name; a router that spreads requests across models cannot be indexed.",
            );
        }

        const vector = new Float32Array(dim);

        for (let i = 0; i < dim; i += 1) {
            const value = row[i];

            if (typeof value !== "number" || !Number.isFinite(value)) {
                throw new EmbeddingError(
                    "bad-response",
                    "The endpoint returned a vector with a missing or non-numeric value.",
                    "Check the model name; the endpoint is not answering like an embeddings API.",
                );
            }

            vector[i] = value;
        }

        return normalize(vector);
    });

    return out;
}

function errorForStatus(status: number, body: string, url: string): EmbeddingError {
    const snippet = body.trim().slice(0, 200);

    if (status === 401 || status === 403) {
        return new EmbeddingError(
            "auth",
            `The embedding endpoint refused the API key (HTTP ${status}).`,
            "Check the API key in the plugin settings.",
        );
    }

    if (status === 404) {
        return new EmbeddingError(
            "not-found",
            `The embedding endpoint at ${url} answered 404.`,
            "This server has no /embeddings endpoint or the model name is wrong.",
        );
    }

    if (status === 429) {
        return new EmbeddingError(
            "rate-limit",
            "The embedding endpoint is rate-limiting requests (HTTP 429).",
            "Wait a moment and try again, or lower how much of the chat is indexed.",
        );
    }

    if (status >= 500) {
        return new EmbeddingError(
            "server",
            `The embedding endpoint failed with HTTP ${status}.`,
            "Try again; if it keeps failing, check the endpoint's status page or logs.",
        );
    }

    if (status === 400 || status === 422) {
        return new EmbeddingError(
            "bad-response",
            `The endpoint rejected the request${snippet ? `: ${snippet}` : ` (HTTP ${status}).`}`,
            "Check the model name and that the endpoint is OpenAI-compatible.",
        );
    }

    return new EmbeddingError(
        "bad-response",
        `The embedding endpoint replied HTTP ${status}${snippet ? `: ${snippet}` : "."}`,
        "Check the base URL; it usually ends in /v1.",
    );
}

async function embedLocalBatch(
    texts: string[],
    hostFetch: HostFetch,
    signal?: AbortSignal,
): Promise<Float32Array[]> {
    if (signal?.aborted) {
        throw abortedError();
    }

    let response: Response;

    try {
        response = await hostFetch("/embed", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ texts }),
            signal,
        });
    } catch {
        if (signal?.aborted) {
            throw abortedError();
        }

        // `api.host.fetch` rejects when the module is not approved or not running;
        // it never reaches llama-server, so this is a setup problem, not a crash.
        throw new EmbeddingError(
            "not-ready",
            "The plugin's server module is not reachable.",
            "Approve `server:process` in Options > Plugins > Approvals and make sure the plugin is enabled.",
        );
    }

    if (!response.ok) {
        throw await localErrorFor(response);
    }

    const raw = await safeJson(response);
    const vectors = isRecord(raw) && Array.isArray(raw.vectors) ? raw.vectors : null;

    if (!vectors) {
        throw new EmbeddingError(
            "bad-response",
            "The local embedding server returned no vectors.",
            "This is a plugin bug; check the plugin's server log.",
        );
    }

    return validateVectors(vectors.map((row) => (Array.isArray(row) ? row.map(Number) : [])));
}

async function localErrorFor(response: Response): Promise<EmbeddingError> {
    const raw = await safeJson(response);
    const error = isRecord(raw) && isRecord(raw.error) ? raw.error : null;
    const code = error && typeof error.code === "string" ? error.code : "";
    const detail = error && typeof error.message === "string" ? error.message : "";

    if (code === "not-installed") {
        return new EmbeddingError(
            "not-installed",
            "The local embedding model is not installed yet.",
            "Open the plugin settings and press Install.",
        );
    }

    if (code === "starting-failed") {
        return new EmbeddingError(
            "not-ready",
            `The local embedding server failed to start${detail ? `: ${detail}` : "."}`,
            "Check the plugin's server log; the binary or model path may be wrong.",
        );
    }

    if (code === "not-ready") {
        return new EmbeddingError(
            "not-ready",
            `The local embedding server is not ready${detail ? `: ${detail}` : "."}`,
            "Wait a few seconds and try again; the first load reads the model from disk.",
        );
    }

    if (code === "too-long") {
        return new EmbeddingError(
            "too-long",
            "The text is longer than the local model's context.",
            "Lower the chunk size in the plugin settings.",
        );
    }

    if (code === "bad-request") {
        return new EmbeddingError(
            "bad-response",
            `The local embedding server rejected the request${detail ? `: ${detail}` : "."}`,
            "This is a plugin bug; report it with the server log.",
        );
    }

    if (code === "busy") {
        return new EmbeddingError(
            "server",
            "The local embedding server is busy.",
            "Try again in a moment.",
        );
    }

    // The app itself answers 404/503 when the module is missing or failed to boot.
    if (response.status === 404 || response.status === 503) {
        return new EmbeddingError(
            "not-ready",
            "The plugin's server module is not running.",
            "Approve `server:process` in Options > Plugins > Approvals and make sure the plugin is enabled.",
        );
    }

    return new EmbeddingError(
        "server",
        `The local embedding server failed (HTTP ${response.status})${detail ? `: ${detail}` : "."}`,
        "Check the plugin's server log.",
    );
}

function asEmbeddingError(error: unknown): EmbeddingError {
    if (error instanceof EmbeddingError) {
        return error;
    }

    if (error instanceof Error && error.name === "AbortError") {
        return abortedError();
    }

    return new EmbeddingError(
        "bad-response",
        `The embedding request failed: ${error instanceof Error ? error.message : String(error)}`,
        "Check the base URL and the model name, then try again.",
    );
}

function isRetryable(error: EmbeddingError): boolean {
    return error.code === "rate-limit" || error.code === "server" || error.code === "unreachable";
}

function abortedError(): EmbeddingError {
    return new EmbeddingError("aborted", "The embedding request was cancelled.", "Nothing to do; the app cancelled it.");
}

async function safeJson(response: Response): Promise<unknown> {
    try {
        return await response.json();
    } catch {
        return null;
    }
}

async function safeText(response: Response): Promise<string> {
    try {
        return await response.text();
    } catch {
        return "";
    }
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

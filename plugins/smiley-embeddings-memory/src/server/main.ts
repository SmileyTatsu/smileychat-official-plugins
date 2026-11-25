// The server module's entry point: the small HTTP surface the plugin's browser side calls.
//
// The app loads this file in its Bun server, calls `activate(host)` and routes every request to
// /api/plugins/smiley-embeddings-memory/host/... here (auth, CSRF and the same-origin check are
// already done by the app). Routes, method by method:
//
//   GET    /status           process + installer + config snapshot (what the setup panel polls)
//   POST   /config           save the server config
//   POST   /start            start llama-server now (warm-up)
//   POST   /stop             stop it now
//   POST   /binary/install   download and unpack llama.cpp
//   POST   /binary/cancel    cancel that download
//   DELETE /binary           delete the downloaded builds (stops the server first)
//   POST   /models/download  download the GGUF model
//   POST   /models/cancel    cancel that download
//   DELETE /models           delete the downloaded model files (stops the server first)
//   POST   /embed            { texts: string[] } -> { vectors: number[][] }
//
// SECURITY — this file is the module's whole attack surface, so:
//   * The only thing ever executed is what Installer.resolveLaunch returns: a llama-server under
//     dataDir/bin/<tag>/<backend>/ or the manual path the user saved in config. The argument list
//     is fixed in llama-process.ts and the handful of config values it carries (ctx, device,
//     pooling, paths) are validated and normalised by config.ts. No request field — not a path,
//     not a query, not a header — reaches a command line.
//   * The route table below is closed: a path that is not a key in it is a 404, and a method that
//     is not one of that key's entries is a 405. Adding a route means editing this table.
//   * Request bodies are chat text. They are never logged, never echoed into an error message and
//     never written to disk here; the debug log gets counts and timings only.
//   * Nothing from a request is joined onto a filesystem path: every path the installer and the
//     process manager use comes from config.json (validated) or from their own bookkeeping.

import type { ServerStatus } from "../controller";
import { ConfigError, loadConfig, saveConfig } from "./config";
import { Installer } from "./installer";
import { EmbedError, LlamaProcess, MAX_BATCH_TEXTS, type EmbedErrorCode } from "./llama-process";
import type { InstallerStatus, LaunchSpec, PluginServerHost, PluginServerInstance, ServerConfig } from "./types";

/** Largest `/embed` body accepted. 64 texts of 8000 characters stay well under it. */
const MAX_BODY_BYTES = 2 * 1024 * 1024;

/** Longest single text. Above this llama-server rejects the batch anyway; the browser chunks. */
const MAX_TEXT_CHARS = 8000;

const JSON_HEADERS = { "content-type": "application/json" } as const;

type HttpMethod = "GET" | "POST" | "DELETE";

type Handler = (request: Request) => Promise<Response>;

type RouteEntry = Partial<Record<HttpMethod, Handler>>;

/** Status code for each EmbedError code: the caller acts differently on 413 vs 503 vs 504. */
const EMBED_ERROR_STATUS: Record<EmbedErrorCode, number> = {
    "too-long": 413,
    "bad-request": 400,
    "not-ready": 503,
    timeout: 504,
    server: 502,
};

/** An error that already knows how it should answer. Anything else becomes a 500 `internal`. */
class RouteError extends Error {
    readonly status: number;
    readonly code: string;

    constructor(status: number, code: string, message: string) {
        super(message);
        this.name = "RouteError";
        this.status = status;
        this.code = code;
    }
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function errorResponse(status: number, code: string, message: string, headers?: HeadersInit): Response {
    return Response.json({ error: { code, message } }, { status, headers });
}

/**
 * The table's key form of a subpath.
 *
 * The app hands "/" for a request with no path after `/host` and leaves percent-encoding alone.
 * Every route here is plain ASCII, so a percent-encoded path simply does not match (404) instead
 * of being decoded into something else. One trailing slash is accepted — `/status/` is `/status`
 * — because that is what a hand-written curl or a browser address bar produces; anything more
 * (`/status//`) is a 404.
 */
function routeKey(subpath: string): string {
    if (subpath.length > 1 && subpath.endsWith("/")) {
        return subpath.slice(0, -1);
    }

    return subpath;
}

type BodyResult = { ok: true; value: unknown } | { ok: false; reason: "too-large" | "bad-request" };

/**
 * Reads a JSON body with a hard byte ceiling.
 *
 * The ceiling is enforced while reading, never from `content-length`: a chunked request has no
 * such header and a wrong one costs nothing to send. Reading stops as soon as the limit is
 * crossed, so an oversize body is never assembled in memory.
 */
async function readJsonBody(request: Request, limit = MAX_BODY_BYTES): Promise<BodyResult> {
    const declared = request.headers.get("content-length");

    if (declared !== null && Number(declared) > limit) {
        return { ok: false, reason: "too-large" };
    }

    const body = request.body;

    if (body === null) {
        return { ok: false, reason: "bad-request" };
    }

    const reader = body.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;

    for (;;) {
        const { done, value } = await reader.read();

        if (done) {
            break;
        }

        total += value.byteLength;

        if (total > limit) {
            await reader.cancel().catch(() => undefined);

            return { ok: false, reason: "too-large" };
        }

        chunks.push(value);
    }

    if (total === 0) {
        return { ok: false, reason: "bad-request" };
    }

    const merged = new Uint8Array(total);
    let offset = 0;

    for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.byteLength;
    }

    try {
        return { ok: true, value: JSON.parse(new TextDecoder().decode(merged)) as unknown };
    } catch {
        return { ok: false, reason: "bad-request" };
    }
}

function asRecord(value: unknown, message: string): Record<string, unknown> {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new RouteError(400, "bad-request", message);
    }

    return value as Record<string, unknown>;
}

export async function activate(host: PluginServerHost): Promise<PluginServerInstance> {
    const dataDir = host.dataDir;
    // Both own no resources until they are used: no I/O happens here, so a slow disk cannot fail
    // the start (activate has a 15 s budget).
    const installer = new Installer(host);
    const llamaProcess = new LlamaProcess(host);

    let disposed = false;

    async function config(): Promise<ServerConfig> {
        return loadConfig(dataDir);
    }

    /** resolveLaunch is the only producer of a launch spec; a missing file is a 409, not a 500. */
    async function launchCandidates(current: ServerConfig): Promise<LaunchSpec[]> {
        const launch = await installer.resolveLaunch(current);

        if (!launch.ok) {
            throw new RouteError(409, "not-installed", launch.error);
        }

        return launch.candidates;
    }

    /** ensureReady's message is already actionable ("… failed to start: … Check …"); pass it on. */
    async function ensureReady(candidates: LaunchSpec[], current: ServerConfig): Promise<void> {
        try {
            await llamaProcess.ensureReady(candidates, current.device, current.idleMinutes);
        } catch (error) {
            throw new RouteError(502, "starting-failed", errorMessage(error));
        }
    }

    async function embedTexts(request: Request): Promise<string[]> {
        const body = await readJsonBody(request);

        if (!body.ok) {
            throw new RouteError(
                body.reason === "too-large" ? 413 : 400,
                body.reason === "too-large" ? "too-large" : "bad-request",
                body.reason === "too-large"
                    ? `The request body is larger than ${MAX_BODY_BYTES} bytes`
                    : "The request body must be a JSON object",
            );
        }

        const record = asRecord(body.value, 'The body must be a JSON object with a "texts" array');
        const texts = record.texts;

        if (!Array.isArray(texts)) {
            throw new RouteError(400, "bad-request", '"texts" must be an array of strings');
        }

        if (texts.length < 1 || texts.length > MAX_BATCH_TEXTS) {
            throw new RouteError(400, "bad-request", `"texts" must hold 1 to ${MAX_BATCH_TEXTS} strings, got ${texts.length}`);
        }

        texts.forEach((text, index) => {
            if (typeof text !== "string") {
                throw new RouteError(400, "bad-request", `"texts"[${index}] must be a string`);
            }

            if (text.length > MAX_TEXT_CHARS) {
                throw new RouteError(
                    400,
                    "bad-request",
                    `"texts"[${index}] is ${text.length} characters; split it, the limit is ${MAX_TEXT_CHARS}`,
                );
            }
        });

        return texts as string[];
    }

    const handlers: Record<string, RouteEntry> = {
        "/status": {
            GET: async () => {
                // Read every time: the setup panel polls this, and a config change made in another
                // tab has to show up without a restart.
                const current = await config();
                let installerStatus: InstallerStatus;

                try {
                    installerStatus = await installer.status(current);
                } catch (error) {
                    // status() is a stat/readdir walk; a file removed mid-walk or a permission
                    // change must not turn the route the UI polls into an exception.
                    host.logger.warn("could not read the installer status", error);

                    return errorResponse(500, "internal", "Could not read the installer state; see the plugin log.");
                }

                const payload: ServerStatus = {
                    process: llamaProcess.status(),
                    installer: installerStatus,
                    config: current,
                    platform: { os: process.platform, arch: process.arch },
                };

                return Response.json(payload);
            },
        },
        "/config": {
            POST: async (request) => {
                const body = await readJsonBody(request);

                if (!body.ok) {
                    throw new RouteError(
                        body.reason === "too-large" ? 413 : 400,
                        body.reason === "too-large" ? "too-large" : "bad-request",
                        body.reason === "too-large"
                            ? `The request body is larger than ${MAX_BODY_BYTES} bytes`
                            : "The request body must be JSON",
                    );
                }

                asRecord(body.value, "The body must be a JSON object of config fields");

                let saved: ServerConfig;

                try {
                    saved = await saveConfig(dataDir, body.value);
                } catch (error) {
                    if (error instanceof ConfigError) {
                        throw new RouteError(400, "invalid-config", error.message);
                    }

                    throw error;
                }

                // Saving starts and stops nothing: the next /start or /embed compares the launch
                // spec it resolves from the new config with the running one and restarts only when
                // it actually differs.
                host.logger.debug("config saved", { ctx: saved.ctx, device: saved.device, profile: saved.profile });

                return Response.json(saved);
            },
        },
        "/start": {
            POST: async () => {
                const current = await config();
                const candidates = await launchCandidates(current);

                await ensureReady(candidates, current);

                return Response.json(llamaProcess.status());
            },
        },
        "/stop": {
            POST: async () => {
                await llamaProcess.stop();

                return Response.json({ ok: true });
            },
        },
        "/binary/install": {
            POST: async () => {
                const current = await config();

                // 202: the download runs in the background and is polled through /status.
                return Response.json(installer.startBinaryInstall(current), { status: 202 });
            },
        },
        "/binary/cancel": {
            POST: async () => {
                installer.cancel("binary");

                return Response.json({ ok: true });
            },
        },
        "/binary": {
            DELETE: async () => {
                // The running server holds its binary open (Windows) and would keep serving a build
                // whose folder is gone, so it stops first. Installer refuses while a download runs.
                await llamaProcess.stop();

                try {
                    await installer.removeBinary();
                } catch (error) {
                    throw new RouteError(409, "busy", errorMessage(error));
                }

                return Response.json({ ok: true });
            },
        },
        "/models/download": {
            POST: async () => {
                const current = await config();

                return Response.json(installer.startModelDownload(current), { status: 202 });
            },
        },
        "/models/cancel": {
            POST: async () => {
                installer.cancel("model");

                return Response.json({ ok: true });
            },
        },
        "/models": {
            DELETE: async () => {
                await llamaProcess.stop();

                try {
                    await installer.removeModels();
                } catch (error) {
                    throw new RouteError(409, "busy", errorMessage(error));
                }

                return Response.json({ ok: true });
            },
        },
        "/embed": {
            POST: async (request) => {
                const startedAt = Date.now();
                const texts = await embedTexts(request);
                const current = await config();
                const candidates = await launchCandidates(current);

                // Concurrent tabs call this at the same time; ensureReady shares one start and
                // embed's own dedupe is the only serialisation needed. Nothing else is queued.
                await ensureReady(candidates, current);

                let vectors: number[][];

                try {
                    vectors = await llamaProcess.embed(texts);
                } catch (error) {
                    if (error instanceof EmbedError) {
                        throw new RouteError(EMBED_ERROR_STATUS[error.code], error.code, error.message);
                    }

                    throw error;
                }

                // Counts and timing only: the texts are the user's chat and must not be logged.
                host.logger.debug(`embedded ${texts.length} text(s) in ${Date.now() - startedAt} ms`);

                return Response.json({ vectors });
            },
        },
    };

    async function handle(request: Request, subpath: string): Promise<Response> {
        const key = routeKey(subpath);
        const entry = Object.keys(handlers).includes(key) ? handlers[key] : undefined;

        if (entry === undefined) {
            return errorResponse(404, "not-found", `No route for ${key}`);
        }

        const method = request.method.toUpperCase();
        const allowed = Object.keys(entry);

        if (!allowed.includes(method)) {
            return errorResponse(405, "method-not-allowed", `${method} is not allowed on ${key}`, {
                ...JSON_HEADERS,
                allow: allowed.join(", "),
            });
        }

        try {
            return await (entry as Record<string, Handler>)[method](request);
        } catch (error) {
            if (error instanceof RouteError) {
                return errorResponse(error.status, error.code, error.message);
            }

            // Details go to the log: a stack trace in a response body tells a caller about paths
            // and internals it has no business seeing.
            host.logger.error(`request failed: ${method} ${key}`, error);

            return errorResponse(500, "internal", "The plugin's server module hit an unexpected error; see the plugin log.");
        }
    }

    async function dispose(): Promise<void> {
        // dispose has a 5 s budget: the installer only aborts its tasks and llamaProcess.stop()
        // kills the child through host.spawn, so nothing here waits on the network.
        if (disposed) {
            return;
        }

        disposed = true;
        installer.dispose();
        await llamaProcess.dispose();
    }

    host.logger.debug("server module activated");

    return { handle, dispose };
}

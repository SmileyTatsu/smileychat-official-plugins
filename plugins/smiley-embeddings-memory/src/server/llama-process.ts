// Owns the llama-server process that turns text into vectors.
//
// Four decisions shape this file:
//   * the server starts lazily (the first /embed pays for the load) and is stopped by an idle
//     timer, because a llama-server holding 3 GB of VRAM must not outlive its usefulness;
//   * every argument is fixed here. Nothing from an HTTP request reaches a command line, and the
//     listener is always 127.0.0.1 — the browser never talks to llama-server directly;
//   * the device and layer count shown to the user come from llama-server's own log file
//     (llama-log.ts), not from what was requested: a GPU build with `-ngl 0` still names the GPU;
//   * a start failure under `device: auto` moves on to the next candidate build, and under
//     `gpu`/`cpu` it does not — silently computing on the CPU would be much slower than the user
//     asked for.

import { mkdir, readFile, rm } from "node:fs/promises";
import { basename, dirname, join } from "node:path";

import { parseLlamaLog, type LlamaLogInfo } from "./llama-log";
import type { Backend } from "./releases";
import type { DeviceChoice, LaunchSpec, ManagedProcess, PluginServerHost, ProcessStatus } from "./types";

/**
 * How many texts one `/embed` request may carry.
 *
 * Measured with the shipped flags on a RTX 5060 Ti: 256 texts of 400 characters still succeed on
 * both profiles, so the context is not the limit. 64 is chosen instead: it is twice the largest
 * batch the indexer sends, and it keeps the worst case (Qwen3-4B, about 15 emb/s) near 4 s so a
 * caller mistake cannot block the single plugin request for half a minute.
 */
export const MAX_BATCH_TEXTS = 64;

const LISTEN_HOST = "127.0.0.1";
const LOG_FILE_NAME = "llama-server.log";
const BIND_RETRIES = 2;
const KILL_GRACE_MS = 5000;

/**
 * `-b`/`-ub` for the current context.
 *
 * EmbeddingGemma pools with mean attention, which is non-causal: a single text longer than the
 * physical batch is *not* split, the server answers HTTP 500 "input (N tokens) is too large to
 * process". With llama.cpp's default 512 ubatch that happens at ~2000 characters — inside a
 * 1500-character chunk as soon as the text is CJK instead of English. Measured cost of raising it
 * to 2048 (RTX 5060 Ti, b12686): 133 -> 304 MiB of VRAM on the GPU and 151 -> ~80 emb/s for
 * 400-character chunks, which still beats `-np 1` (47 emb/s) that the design draft assumed.
 * `-np 4` (llama.cpp's own default, unified KV shared by the slots) is what gives the 4 slots
 * that serve the inputs of one batch in parallel; there is no gain in `-np 1`.
 */
function batchArgs(ctx: number): string[] {
    const ubatch = String(Math.max(64, Math.min(2048, ctx)));

    return ["-b", ubatch, "-ub", ubatch, "-np", "4"];
}

const BACKEND_LABEL: Record<Backend | "manual", string> = {
    cuda: "CUDA",
    vulkan: "Vulkan",
    cpu: "CPU",
    metal: "Metal",
    manual: "llama-server",
};

export type LlamaProcessDeps = {
    pickPort: () => Promise<number>;
    fetchImpl: typeof fetch;
    now: () => number;
    startTimeoutMs: number;
    healthIntervalMs: number;
};

export type EmbedErrorCode = "not-ready" | "too-long" | "bad-request" | "server" | "timeout";

export class EmbedError extends Error {
    code: EmbedErrorCode;

    constructor(code: EmbedErrorCode, message: string) {
        super(message);
        this.name = "EmbedError";
        this.code = code;
    }
}

type StartFailureKind = "exit" | "timeout" | "spawn" | "bind";

class StartError extends Error {
    kind: StartFailureKind;

    constructor(kind: StartFailureKind, message: string) {
        super(message);
        this.name = "StartError";
        this.kind = kind;
    }
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

// llama.cpp writes its log file with ANSI colours when the terminal supports them; the escapes
// would end up inside user-facing error texts and inside the parsed device name.
function stripAnsi(text: string): string {
    return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}

function sameSpec(left: LaunchSpec, right: LaunchSpec): boolean {
    return (
        left.binary === right.binary &&
        left.model === right.model &&
        left.backend === right.backend &&
        left.pooling === right.pooling &&
        left.ctx === right.ctx &&
        left.gpuLayers === right.gpuLayers
    );
}

async function pickFreePort(): Promise<number> {
    // Ask the OS for a free port and hand it straight back. Another process can take it in the
    // meantime; that race is accepted here and shows up as a bind error at start, which
    // startCandidate retries on a fresh port.
    const listener = Bun.listen({ hostname: LISTEN_HOST, port: 0, socket: { data() {} } });
    const port = listener.port;
    listener.stop(true);

    return port;
}

function launchEnv(binary: string): Record<string, string> {
    const directory = dirname(binary);
    const env: Record<string, string> = {};

    // Windows resolves the CUDA/Vulkan DLLs next to the executable; the POSIX loaders only look
    // at their path variable, so a build extracted into dataDir would not find its libraries.
    if (process.platform === "linux") {
        env.LD_LIBRARY_PATH = [directory, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":");
    } else if (process.platform === "darwin") {
        env.DYLD_LIBRARY_PATH = [directory, process.env.DYLD_LIBRARY_PATH].filter(Boolean).join(":");
    }

    return env;
}

function launchArgs(spec: LaunchSpec, port: number, logPath: string): string[] {
    return [
        "-m",
        spec.model,
        "--embedding",
        "--pooling",
        spec.pooling,
        "--host",
        LISTEN_HOST,
        "--port",
        String(port),
        "-ngl",
        String(spec.gpuLayers),
        "-c",
        String(spec.ctx),
        ...batchArgs(spec.ctx),
        // b12686 logs "Use --ui/--no-ui (or deprecated --webui/--no-webui)", so `--no-webui` is
        // accepted but deprecated; it is kept because older tags (and a user's manual binary) only
        // know this spelling, and an unknown flag would make llama-server exit.
        "--no-webui",
        // `-lv 4` is what makes the device and offload lines appear at all; `--log-file` is how
        // they are read back, since host.spawn drains the process output into the debug log.
        "--log-file",
        logPath,
        "-lv",
        "4",
    ];
}

const DEFAULT_DEPS: LlamaProcessDeps = {
    pickPort: pickFreePort,
    fetchImpl: fetch,
    now: () => Date.now(),
    startTimeoutMs: 120_000,
    healthIntervalMs: 250,
};

export class LlamaProcess {
    private readonly host: PluginServerHost;
    private readonly deps: LlamaProcessDeps;

    private state: ProcessStatus["state"] = "stopped";
    private process?: ManagedProcess;
    private spec?: LaunchSpec;
    private port?: number;
    private logInfo: LlamaLogInfo = {};
    private startedAt?: string;
    private lastUsedAt?: string;
    private lastError?: string;
    private requestedDevice?: DeviceChoice;

    private starting?: Promise<void>;
    private idleTimer?: ReturnType<typeof setTimeout>;
    private idleMs = 0;
    private disposed = false;

    constructor(host: PluginServerHost, deps: Partial<LlamaProcessDeps> = {}) {
        this.host = host;
        this.deps = { ...DEFAULT_DEPS, ...deps };
    }

    status(): ProcessStatus {
        const status: ProcessStatus = { state: this.state };

        if (this.process !== undefined) {
            status.pid = this.process.pid;
        }

        if (this.port !== undefined) {
            status.port = this.port;
        }

        if (this.spec !== undefined) {
            status.backend = this.backendName();
            status.requestedDevice = this.requestedDevice;

            if (this.logInfo.device !== undefined) {
                status.device = this.logInfo.device;
            }

            if (this.logInfo.layersOffloaded !== undefined) {
                status.layersOffloaded = this.logInfo.layersOffloaded;
            }

            if (this.logInfo.layersTotal !== undefined) {
                status.layersTotal = this.logInfo.layersTotal;
            }
        }

        if (this.startedAt !== undefined) {
            status.startedAt = this.startedAt;
        }

        if (this.lastUsedAt !== undefined) {
            status.lastUsedAt = this.lastUsedAt;
        }

        if (this.lastError !== undefined) {
            status.lastError = this.lastError;
        }

        return status;
    }

    /**
     * Makes sure a server that matches one of `candidates` is listening. Concurrent calls share a
     * single start, so three messages arriving at once during a cold start still launch one process.
     */
    async ensureReady(candidates: LaunchSpec[], requestedDevice: DeviceChoice, idleMinutes: number): Promise<void> {
        if (this.disposed) {
            throw new Error("the llama-server manager was disposed");
        }

        if (candidates.length === 0) {
            throw new Error("no llama-server candidate to start");
        }

        this.requestedDevice = requestedDevice;
        this.configureIdle(idleMinutes);

        if (this.starting !== undefined) {
            await this.starting;

            return;
        }

        // Identical launch spec: the running server is already the right one. Any candidate
        // counts, not just the first, so a server that started as the Vulkan fallback is not
        // restarted (and does not lose its model load) on every later /embed.
        if (this.state === "ready" && this.spec !== undefined && candidates.some((candidate) => sameSpec(candidate, this.spec as LaunchSpec))) {
            this.touch();

            return;
        }

        if (this.state !== "stopped") {
            // A different model/context/binary, or the previous attempt failed: the old process
            // must be gone before the new one binds its port.
            await this.stop();
        }

        this.starting = this.startCandidates(candidates).finally(() => {
            this.starting = undefined;
        });

        await this.starting;
    }

    async embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
        if (this.disposed) {
            throw new EmbedError("not-ready", "the embedding server is not available (plugin disposed)");
        }

        if (this.state !== "ready" || this.port === undefined) {
            throw new EmbedError("not-ready", `llama-server is ${this.state}; call ensureReady() first`);
        }

        if (texts.length === 0) {
            return [];
        }

        if (texts.length > MAX_BATCH_TEXTS) {
            throw new EmbedError("bad-request", `at most ${MAX_BATCH_TEXTS} texts per batch, got ${texts.length}`);
        }

        const url = `http://${LISTEN_HOST}:${this.port}/v1/embeddings`;
        let response: Response;

        try {
            response = await this.deps.fetchImpl(url, {
                method: "POST",
                headers: { "content-type": "application/json" },
                // llama-server ignores the model name but the OpenAI shape requires one.
                body: JSON.stringify({ input: texts, model: "x" }),
                signal,
            });
        } catch (error) {
            if (signal?.aborted) {
                throw new EmbedError("timeout", "the embedding request was aborted");
            }

            // The process is expected to be up, so an unreachable port means it died without the
            // exit watcher seeing it yet: say so instead of reporting a generic network error.
            const message = `could not reach llama-server on port ${this.port}: ${errorMessage(error)}`;
            this.state = "error";
            this.lastError = message;
            this.clearIdle();

            throw new EmbedError("server", message);
        }

        if (!response.ok) {
            const body = await response.text().catch(() => "");
            const message = readServerMessage(body) ?? `llama-server answered HTTP ${response.status}`;

            // The request reached the server, so it is alive: keep the idle timer alive too.
            this.touch();

            if (isTooLong(body)) {
                throw new EmbedError("too-long", message);
            }

            throw new EmbedError(response.status === 400 ? "bad-request" : "server", message);
        }

        let payload: { data?: { index?: unknown; embedding?: unknown }[] };

        try {
            payload = (await response.json()) as typeof payload;
        } catch (error) {
            throw new EmbedError("server", `llama-server returned a body that is not JSON: ${errorMessage(error)}`);
        }

        const entries = payload.data;

        if (!Array.isArray(entries) || entries.length !== texts.length) {
            throw new EmbedError("server", `llama-server returned ${entries?.length ?? 0} vectors for ${texts.length} texts`);
        }

        // The OpenAI contract puts the input position in `index`; the array order is not promised,
        // and a permuted vector in a per-chat index would silently score the wrong chunk.
        const vectors = new Map<number, number[]>();

        for (const entry of entries) {
            const index = entry.index;

            if (typeof index !== "number" || !Number.isInteger(index) || index < 0 || index >= texts.length) {
                throw new EmbedError("server", `llama-server returned an invalid embedding index (${String(index)})`);
            }

            if (!Array.isArray(entry.embedding) || entry.embedding.some((value) => typeof value !== "number")) {
                throw new EmbedError("server", "llama-server returned an embedding that is not a number array");
            }

            vectors.set(index, entry.embedding as number[]);
        }

        this.touch();

        return texts.map((_, index) => {
            const vector = vectors.get(index);

            if (vector === undefined) {
                throw new EmbedError("server", `llama-server did not return an embedding for input ${index}`);
            }

            return vector;
        });
    }

    async stop(): Promise<void> {
        this.clearIdle();

        const proc = this.process;
        const wasRunning = this.state !== "stopped";

        // Cleared before the kill so the exit watcher treats this as an expected exit.
        this.process = undefined;
        this.state = "stopped";
        this.port = undefined;
        this.spec = undefined;
        this.logInfo = {};
        this.startedAt = undefined;

        if (proc !== undefined) {
            try {
                await proc.kill();
            } catch (error) {
                this.host.logger.warn(`llama-server kill failed: ${errorMessage(error)}`);
            }

            // Waiting for the exit matters: the next start picks a port and loads the model, and
            // on Windows the old process still owns its VRAM until it is really gone.
            await Promise.race([proc.exited, sleep(KILL_GRACE_MS)]);
        }

        if (wasRunning) {
            this.host.logger.info("stopped llama-server");
        }
    }

    async dispose(): Promise<void> {
        this.disposed = true;
        await this.stop();
    }

    private configureIdle(idleMinutes: number): void {
        const ms = idleMinutes > 0 ? Math.max(1, idleMinutes * 60_000) : 0;

        if (ms === this.idleMs) {
            return;
        }

        this.idleMs = ms;

        if (this.state === "ready") {
            this.startIdleTimer();
        }
    }

    private startIdleTimer(): void {
        this.clearIdle();

        if (this.idleMs <= 0 || this.state !== "ready") {
            return;
        }

        this.idleTimer = setTimeout(() => {
            this.idleTimer = undefined;

            if (this.state !== "ready") {
                return;
            }

            this.host.logger.info("idle: stopped llama-server");
            void this.stop();
        }, this.idleMs);

        // A pending idle timer must not keep the host process alive when the app wants to exit.
        (this.idleTimer as { unref?: () => void }).unref?.();
    }

    private clearIdle(): void {
        if (this.idleTimer !== undefined) {
            clearTimeout(this.idleTimer);
            this.idleTimer = undefined;
        }
    }

    private touch(): void {
        this.lastUsedAt = new Date(this.deps.now()).toISOString();
        this.startIdleTimer();
    }

    private async startCandidates(candidates: LaunchSpec[]): Promise<void> {
        let fallbackNote: string | undefined;

        for (let index = 0; index < candidates.length; index += 1) {
            const candidate = candidates[index];

            try {
                await this.startCandidate(candidate);
                // startCandidate clears lastError on success; the explanation of a fallback is
                // still worth showing, so it is restored here.
                this.lastError = fallbackNote;

                return;
            } catch (error) {
                const failure = error instanceof StartError ? error : new StartError("spawn", errorMessage(error));
                const done = index === candidates.length - 1;
                const label = BACKEND_LABEL[candidate.backend];

                this.state = "error";

                if (done || this.requestedDevice !== "auto") {
                    this.lastError = `llama-server (${label}) failed to start: ${failure.message}. ${this.failureHint()}`;

                    throw new Error(this.lastError);
                }

                const next = candidates[index + 1];
                fallbackNote = `${label} build failed to start: ${failure.message}; fell back to ${BACKEND_LABEL[next.backend]}`;
                this.lastError = fallbackNote;
                this.host.logger.warn(fallbackNote);
            }
        }
    }

    private failureHint(): string {
        if (this.requestedDevice === "cpu") {
            return "Device is set to CPU; check that the binary and the model file still exist.";
        }

        if (this.requestedDevice === "gpu") {
            return "Device is set to GPU, so the plugin does not fall back to CPU; switch device to auto in the plugin settings to let it try another build.";
        }

        return "No llama-server build could start; check the binary and model paths in the plugin settings.";
    }

    private async startCandidate(spec: LaunchSpec): Promise<void> {
        let lastFailure: StartError | undefined;

        for (let attempt = 0; attempt <= BIND_RETRIES; attempt += 1) {
            try {
                await this.spawnOnce(spec);

                return;
            } catch (error) {
                lastFailure = error instanceof StartError ? error : new StartError("spawn", errorMessage(error));

                // Ports picked from the OS can be taken between the probe and the bind; a new port
                // is all that is needed, so this never counts as a failed backend.
                if (lastFailure.kind === "bind" && attempt < BIND_RETRIES) {
                    this.host.logger.warn(`llama-server could not bind its port, retrying (${attempt + 1}/${BIND_RETRIES})`);
                    continue;
                }

                throw lastFailure;
            }
        }

        throw lastFailure ?? new StartError("spawn", "llama-server did not start");
    }

    private async spawnOnce(spec: LaunchSpec): Promise<void> {
        await mkdir(this.host.dataDir, { recursive: true });

        const logPath = this.logPath();
        // Truncated before every start: the tail of this file is the failure text, and a stale
        // line from the previous run would describe the wrong problem.
        await rm(logPath, { force: true });

        const port = await this.deps.pickPort();
        const startedAt = this.deps.now();

        this.state = "starting";
        this.spec = spec;
        this.port = port;
        this.startedAt = new Date(startedAt).toISOString();
        this.logInfo = {};

        let proc: ManagedProcess;

        try {
            proc = this.host.spawn(spec.binary, launchArgs(spec, port, logPath), {
                cwd: dirname(spec.binary),
                env: launchEnv(spec.binary),
            });
        } catch (error) {
            this.state = "error";
            this.port = undefined;

            throw new StartError("spawn", `could not run ${basename(spec.binary)}: ${errorMessage(error)}`);
        }

        this.process = proc;
        this.host.logger.info(
            `llama-server starting (${BACKEND_LABEL[spec.backend]}, port ${port}, ${basename(spec.binary)}, ctx ${spec.ctx})`,
        );

        try {
            await this.waitForHealth(proc, port, startedAt + this.deps.startTimeoutMs);
        } catch (error) {
            const failure = error instanceof StartError ? error : new StartError("exit", errorMessage(error));
            const tail = await this.logTail();
            const message = tail === "" || failure.message.endsWith(tail) ? failure.message : `${failure.message}: ${tail}`;

            this.state = "error";
            this.process = undefined;
            this.port = undefined;
            this.lastError = message;
            await killQuietly(proc);

            throw new StartError(failure.kind, message);
        }

        this.logInfo = parseLlamaLog(stripAnsi(await this.readLogText()));
        this.state = "ready";
        this.lastError = undefined;
        this.touch();
        this.watchExit(proc);

        this.host.logger.info(
            `llama-server ready in ${this.deps.now() - startedAt} ms (${this.backendName()}, ${this.logInfo.device ?? "cpu"}, layers ${
                this.logInfo.layersOffloaded ?? "?"
            }/${this.logInfo.layersTotal ?? "?"})`,
        );
    }

    private async waitForHealth(proc: ManagedProcess, port: number, deadline: number): Promise<void> {
        let exitCode: number | null | undefined;
        void proc.exited.then((code) => {
            exitCode = code;
        });

        for (;;) {
            if (exitCode !== undefined) {
                const tail = await this.logTail();
                const kind: StartFailureKind = /address already in use|couldn't bind|EADDRINUSE/i.test(tail) ? "bind" : "exit";

                throw new StartError(kind, `exited with code ${exitCode}${tail === "" ? "" : ` (${tail})`}`);
            }

            if (await this.isHealthy(port)) {
                return;
            }

            if (this.deps.now() >= deadline) {
                throw new StartError(
                    "timeout",
                    `did not answer /health within ${Math.round(this.deps.startTimeoutMs / 1000)} s; the first load of a multi-GB model file can take over a minute while antivirus scans it`,
                );
            }

            await sleep(this.deps.healthIntervalMs);
        }
    }

    private async isHealthy(port: number): Promise<boolean> {
        try {
            const response = await this.deps.fetchImpl(`http://${LISTEN_HOST}:${port}/health`, {
                // 503 while the model loads is the normal answer, not an error.
                signal: AbortSignal.timeout(Math.max(1000, this.deps.healthIntervalMs * 4)),
            });

            return response.status === 200;
        } catch {
            return false;
        }
    }

    private watchExit(proc: ManagedProcess): void {
        void proc.exited.then(async (code) => {
            // stop() clears the process before killing it; anything else is a crash we must report.
            if (this.process !== proc || this.state !== "ready") {
                return;
            }

            const tail = await this.logTail();

            this.clearIdle();
            this.process = undefined;
            this.port = undefined;
            this.state = "error";
            // No restart loop: the next ensureReady() starts a fresh process, which is also what
            // the idle timer and the /stop route expect.
            this.lastError = `llama-server exited with code ${code}${tail === "" ? "" : `: ${tail}`}`;

            this.host.logger.error(this.lastError);
        });
    }

    private backendName(): Backend | "manual" | undefined {
        if (this.spec === undefined) {
            return undefined;
        }

        // A user-supplied binary cannot be labelled with a build; the device and layer count from
        // the log still tell the useful part of the story.
        if (this.spec.backend === "manual") {
            return "manual";
        }

        return this.logInfo.backend ?? this.spec.backend;
    }

    private logPath(): string {
        return join(this.host.dataDir, LOG_FILE_NAME);
    }

    private async readLogText(): Promise<string> {
        try {
            return await readFile(this.logPath(), "utf8");
        } catch {
            // The server may fail before it creates the file; the caller reports what it can.
            return "";
        }
    }

    private async logTail(): Promise<string> {
        const lines = stripAnsi(await this.readLogText())
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line !== "");

        return lines[lines.length - 1] ?? "";
    }
}

async function killQuietly(proc: ManagedProcess): Promise<void> {
    try {
        await proc.kill();
    } catch {
        // Already gone; the caller only needs the process stopped.
    }

    await Promise.race([proc.exited, sleep(KILL_GRACE_MS)]);
}

function readServerMessage(body: string): string | undefined {
    try {
        const parsed = JSON.parse(body) as { error?: { message?: unknown } };
        const message = parsed.error?.message;

        return typeof message === "string" && message !== "" ? message : undefined;
    } catch {
        return undefined;
    }
}

/**
 * llama-server answers an over-long input in two different shapes: with causal (last-token)
 * pooling it is a 400 `exceed_context_size_error`, and EmbeddingGemma's non-causal pooling gives a
 * 500 "input (N tokens) is too large to process". Both mean the same thing for the caller: split
 * the text, do not retry.
 */
function isTooLong(body: string): boolean {
    return /exceed_context_size_error|exceeds the available context size|too large to process/i.test(body);
}

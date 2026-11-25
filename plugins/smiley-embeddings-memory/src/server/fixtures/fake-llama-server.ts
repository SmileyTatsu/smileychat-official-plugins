// A stand-in llama-server, started by llama-process.test.ts instead of the real binary.
//
// It is a real process that honours the parts of llama-server's contract the manager relies on:
// it reads the same flags, serves `GET /health` (503 until "loaded") and `POST /v1/embeddings`,
// and writes a `-lv 4` log file shaped like the real one so `llama-log.ts` fills the status.
// What the real binary cannot be talked into on demand — a failed start, a bind conflict, a
// crash after listening, an arbitrarily slow first load — is selected through FAKE_* env vars.

import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

type Flags = Map<string, string>;

function parseFlags(argv: string[]): Flags {
    const flags: Flags = new Map();

    for (let index = 0; index < argv.length; index += 1) {
        const arg = argv[index];

        if (!arg.startsWith("-")) {
            continue;
        }

        const next = argv[index + 1];

        if (next !== undefined && !next.startsWith("-")) {
            flags.set(arg, next);
            index += 1;
        } else {
            flags.set(arg, "1");
        }
    }

    return flags;
}

function envNumber(name: string, fallback: number): number {
    const raw = process.env[name];

    if (raw === undefined || raw === "") {
        return fallback;
    }

    const value = Number(raw);

    return Number.isFinite(value) ? value : fallback;
}

// FNV-1a: the vectors only have to be stable across runs so a test can compare two of them.
function hashText(text: string): number {
    let value = 0x811c9dc5;

    for (let index = 0; index < text.length; index += 1) {
        value ^= text.charCodeAt(index);
        value = Math.imul(value, 0x01000193) >>> 0;
    }

    return value >>> 0;
}

function unitVector(text: string, dim: number): number[] {
    const values: number[] = [];
    let state = hashText(text) || 1;

    for (let index = 0; index < dim; index += 1) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        values.push(state / 0xffffffff - 0.5);
    }

    const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;

    return values.map((value) => value / norm);
}

const DEVICES: Record<string, { device?: string; label: string }> = {
    cuda: { device: "CUDA0", label: "CUDA" },
    vulkan: { device: "Vulkan0", label: "Vulkan" },
    cpu: { label: "CPU" },
};

function logText(port: number, backend: string, gpuLayers: number, ctx: number, slots: number): string {
    const { device, label } = DEVICES[backend] ?? DEVICES.cuda;
    const layers = 25;
    // A GPU build with `-ngl 0` still lists the device and even picks it as "using device";
    // only the offload count shows that the weights stayed in host memory.
    const offloaded = device === undefined ? 0 : gpuLayers > 0 ? layers : 0;
    const lines = [
        "0.00.135.724 I cmn  common_param: common_params_print_info: verbosity = 4 (adjust with the `-lv N` CLI arg)",
        "0.00.135.727 I cmn  common_param: device_info:",
    ];

    if (device !== undefined) {
        lines.push(`0.00.209.149 I cmn  common_param:   - ${device.padEnd(8)}: Fake ${label} GPU (16310 MiB, 15174 MiB free)`);
    }

    lines.push("0.00.209.164 I cmn  common_param:   - CPU     : Fake CPU (32681 MiB, 14159 MiB free)");

    if (device !== undefined && offloaded > 0) {
        lines.push(`0.00.449.485 I llama_prepare_model_devices: using device ${device} (Fake ${label} GPU) (0000:0b:00.0) - 15174 MiB free`);
    }

    if (offloaded > 0) {
        lines.push(`0.00.634.127 I load_tensors: offloading ${offloaded - 1} repeating layers to GPU`);
    }

    lines.push(`0.00.634.128 I load_tensors: offloaded ${offloaded}/${layers} layers to GPU`);
    lines.push("0.00.634.131 I load_tensors:   CPU_Mapped model buffer size =    72.00 MiB");

    if (device !== undefined && offloaded > 0) {
        lines.push(`0.00.634.132 I load_tensors:        ${device} model buffer size =    80.48 MiB`);
    }

    lines.push(`0.00.900.000 I llama_context: n_ctx = ${ctx}, n_batch = 2048, n_ubatch = 2048`);
    lines.push(`0.00.950.000 I srv    load_model: initializing, n_slots = ${slots}, n_ctx_slot = ${ctx}`);
    lines.push(`0.01.000.000 I srv  server_on_li: listening on http://127.0.0.1:${port}`);

    return lines.join("\n");
}

const flags = parseFlags(process.argv.slice(2));
const port = Number(flags.get("--port") ?? "0");
const ctx = Number(flags.get("-c") ?? "4096");
const gpuLayers = Number(flags.get("-ngl") ?? "0");
const parallel = Number(flags.get("-np") ?? "4");
const backend = process.env.FAKE_BACKEND ?? "cuda";
const logFile = flags.get("--log-file");
const healthDelayMs = envNumber("FAKE_HEALTH_DELAY_MS", 0);
const dim = envNumber("FAKE_DIM", 8);
const maxChars = envNumber("FAKE_MAX_CHARS", 100_000);
const exitAfterMs = envNumber("FAKE_EXIT_AFTER_MS", 0);

function writeLog(text: string): void {
    if (logFile === undefined) {
        return;
    }

    mkdirSync(dirname(logFile), { recursive: true });
    writeFileSync(logFile, text);
}

if (process.env.FAKE_FAIL === "1") {
    // Shape of a real failure: llama.cpp logs an error and exits 1 before the socket is up.
    writeLog([
        "0.00.120.000 I cmn  common_param: device_info:",
        `0.00.140.000 E llama_model_load: error loading model: don't know how to load this model (FAKE_FAIL)`,
        "0.00.140.010 E srv    load_model: failed to load model, terminated",
    ].join("\n"));
    process.exit(1);
}

// Simulates losing the port race: llama.cpp prints the bind error and exits 1.
const eaddrFile = process.env.FAKE_EADDR_ONCE_FILE;

if (eaddrFile !== undefined && !existsSync(eaddrFile)) {
    writeFileSync(eaddrFile, "1");
    writeLog([
        "0.00.300.000 E srv  server_on_li: couldn't bind HTTP server socket, port " + String(port),
        "0.00.300.010 E srv  server_on_li: Address already in use",
    ].join("\n"));
    process.exit(1);
}

writeLog(logText(port, backend, gpuLayers, ctx, parallel));

let startedAt = performance.now();
let loaded = healthDelayMs <= 0;

if (healthDelayMs > 0) {
    // The real server answers 503 with `{"status":"loading model"}` while it loads.
    setTimeout(() => {
        loaded = true;
    }, healthDelayMs);
}

if (exitAfterMs > 0) {
    setTimeout(() => {
        if (logFile !== undefined) {
            appendFileSync(logFile, `\n0.02.000.000 E srv  update_slots: fake crash (FAKE_EXIT_AFTER_MS=${exitAfterMs})`);
        }

        process.exit(3);
    }, exitAfterMs);
}

const server = serve();

function serve(): ReturnType<typeof Bun.serve> {
    try {
        return Bun.serve({
            hostname: "127.0.0.1",
            port,
            // The manager keeps one connection per request; the default 10 s idle timeout is fine,
            // but a test may sit on a slow batch, so no timeout at all keeps the fixture out of the way.
            idleTimeout: 0,
            fetch(request) {
                const url = new URL(request.url);

                if (request.method === "GET" && url.pathname === "/health") {
                    if (!loaded) {
                        return Response.json({ status: "loading model" }, { status: 503 });
                    }

                    return Response.json({ status: "ok" });
                }

                if (request.method === "POST" && url.pathname === "/v1/embeddings") {
                    return handleEmbeddings(request);
                }

                return Response.json({ error: { code: 404, message: "not found" } }, { status: 404 });
            },
        });
    } catch (error) {
        // A lost port race looks like this in the real server; the manager retries on a new port
        // only when it can recognise the reason.
        writeLog(
            ["0.00.300.000 E srv  server_on_li: couldn't bind HTTP server socket, port " + String(port), "0.00.300.010 E srv  server_on_li: Address already in use"].join("\n"),
        );
        process.stderr.write(`fake llama-server could not bind: ${(error as Error).message}\n`);
        process.exit(1);
    }
}

async function handleEmbeddings(request: Request): Promise<Response> {
    const body = (await request.json().catch(() => null)) as { input?: unknown } | null;
    const input = body?.input;
    const texts = typeof input === "string" ? [input] : Array.isArray(input) ? input : null;

    if (texts === null || texts.some((text) => typeof text !== "string")) {
        return Response.json({ error: { code: 400, message: "input must be a string or an array of strings" } }, { status: 400 });
    }

    const tooLong = texts.find((text) => (text as string).length > maxChars);

    if (tooLong !== undefined) {
        const tokens = Math.ceil((tooLong as string).length / 4);

        // EmbeddingGemma pools with mean (non-causal) attention: a text longer than the physical
        // batch is answered with 500 "too large to process", not with exceed_context_size_error.
        if (process.env.FAKE_TOO_LONG_500 === "1") {
            return Response.json(
                {
                    error: {
                        code: 500,
                        message: `input (${tokens} tokens) is too large to process. increase the physical batch size (current batch size: 2048)`,
                        type: "server_error",
                    },
                },
                { status: 500 },
            );
        }

        // Same body llama-server sends when the request does not fit: the manager maps it to
        // EmbedError("too-long") instead of a generic server failure.
        return Response.json(
            {
                error: {
                    code: 400,
                    message: `request (${tokens} tokens) exceeds the available context size (${ctx} tokens), try increasing it`,
                    type: "exceed_context_size_error",
                    n_prompt_tokens: tokens,
                    n_ctx: ctx,
                },
            },
            { status: 400 },
        );
    }

    return Response.json({
        object: "list",
        model: "x",
        data: (texts as string[]).map((text, index) => ({
            object: "embedding",
            index,
            embedding: unitVector(text, dim),
        })),
        usage: { prompt_tokens: texts.reduce((sum, text) => sum + Math.ceil((text as string).length / 4), 0), total_tokens: 0 },
    });
}

process.stdout.write(`fake llama-server listening on http://127.0.0.1:${server.port}\n`);

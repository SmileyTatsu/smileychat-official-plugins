// Route-level tests for the server module entry: every route, the closed table, the error codes,
// and the bundled artifact the app actually loads.
//
// The llama-server itself is the fixture from llama-process.test.ts: a real process started by a
// fake host.spawn. The installer is the real one, driven by files seeded on disk, so no test here
// touches the network.

import { afterEach, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { builtinModules } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

import { MAX_BATCH_TEXTS } from "./llama-process";
import { activate } from "./main";
import type { PluginServerHost, PluginServerInstance } from "./types";

const FIXTURE = join(import.meta.dir, "fixtures", "fake-llama-server.ts");
const PLUGIN_ROOT = join(import.meta.dir, "..", "..");
const MARKER = ".installed.json";
const TAG = "b12686";

// Which build resolveLaunch picks for `device: "cpu"` on this host. macOS has one universal
// Metal binary that `-ngl 0` runs on the CPU, everywhere else it is the CPU build.
const BACKEND = process.platform === "darwin" ? "metal" : "cpu";
// The fake host reads the build out of the binary path the manager spawned, the way a real machine
// differs by the build that was installed.
const BUILD_KEY = BACKEND === "metal" ? "cuda" : BACKEND;

type SpawnRecord = { command: string; args: string[]; pid: number; killed: boolean };

type Behaviors = Record<string, Record<string, string>>;

type LogLine = { level: string; message: string };

type FakeHost = PluginServerHost & { spawns: SpawnRecord[]; logs: LogLine[] };

type App = {
    host: FakeHost;
    instance: PluginServerInstance;
    binaryPath: string;
    modelPath: string;
};

const apps: App[] = [];
const hosts: FakeHost[] = [];

function makeHost(behaviors: Behaviors = {}): FakeHost {
    const dataDir = mkdtempSync(join(tmpdir(), "smiley-main-"));
    const spawns: SpawnRecord[] = [];
    const logs: LogLine[] = [];
    const record = (level: string) => (message: string) => logs.push({ level, message });

    const host: FakeHost = {
        pluginId: "smiley-embeddings-memory",
        dataDir,
        logger: { debug: record("debug"), info: record("info"), warn: record("warn"), error: record("error") },
        spawns,
        logs,
        spawn(command, args = [], options = {}) {
            const behavior = behaviors[BUILD_KEY] ?? {};
            const child = Bun.spawn([process.execPath, FIXTURE, ...args], {
                env: { ...process.env, ...options.env, FAKE_BACKEND: BUILD_KEY, ...behavior },
                stdout: "ignore",
                stderr: "ignore",
            });

            spawns.push({ command, args, pid: child.pid, killed: false });
            const spawn = spawns[spawns.length - 1];

            return {
                pid: child.pid,
                exited: child.exited,
                kill: async () => {
                    spawn.killed = true;
                    child.kill();
                    await child.exited;
                },
            };
        },
    };

    hosts.push(host);

    return host;
}

/** A build that looks installed: the marker plus the binary it names, one level down as in a zip. */
async function seedBuild(dataDir: string): Promise<string> {
    const dir = join(dataDir, "bin", TAG, BACKEND);
    const binary = "build/bin/llama-server.exe";
    const binaryPath = join(dir, binary);

    await mkdir(dirname(binaryPath), { recursive: true });
    await writeFile(binaryPath, "binary");
    await writeFile(
        join(dir, MARKER),
        JSON.stringify({ tag: TAG, backend: BACKEND, assets: ["fake.zip"], installedAt: new Date().toISOString(), binary }),
    );

    return binaryPath;
}

async function writeConfig(dataDir: string, patch: Record<string, unknown>): Promise<void> {
    const config = {
        profile: "default",
        ctx: 4096,
        idleMinutes: 0,
        device: "cpu",
        gpuBackend: "auto",
        releaseTag: TAG,
        ...patch,
    };

    await writeFile(join(dataDir, "config.json"), `${JSON.stringify(config, null, 4)}\n`, "utf8");
}

/**
 * activate() plus the files a working install has. `install: "none"` leaves the data dir empty
 * apart from a config that points at files that do not exist.
 */
async function makeApp(
    options: { behaviors?: Behaviors; install?: "full" | "none"; config?: Record<string, unknown> } = {},
): Promise<App> {
    const host = makeHost(options.behaviors ?? {});
    const install = options.install ?? "full";

    const binaryPath = install === "full" ? await seedBuild(host.dataDir) : join(host.dataDir, "bin", "missing", "llama-server");
    // Under models/ so DELETE /models, which removes that folder, is what erases it.
    const modelPath = join(host.dataDir, "models", install === "full" ? "manual.gguf" : "missing.gguf");

    if (install === "full") {
        await mkdir(dirname(modelPath), { recursive: true });
        await writeFile(modelPath, "model");
    }

    await writeConfig(host.dataDir, { modelPath, ...options.config });

    const instance = await activate(host);
    const app: App = { host, instance, binaryPath, modelPath };

    apps.push(app);

    return app;
}

function track(app: App): App {
    apps.push(app);

    return app;
}

function handle(app: App): NonNullable<PluginServerInstance["handle"]> {
    const handler = app.instance.handle;

    if (!handler) {
        throw new Error("activate() returned no handle()");
    }

    return handler;
}

function request(app: App, method: string, subpath: string, body?: unknown): Promise<Response> {
    const init: RequestInit = { method };

    if (body !== undefined) {
        init.body = typeof body === "string" ? body : JSON.stringify(body);
    }

    return Promise.resolve(handle(app)(new Request(`http://127.0.0.1/api/plugins/smiley-embeddings-memory/host${subpath}`, init), subpath));
}

async function read<T = Record<string, any>>(response: Response): Promise<T> {
    return (await response.json()) as T;
}

function isAlive(pid: number): boolean {
    try {
        process.kill(pid, 0);

        return true;
    } catch {
        return false;
    }
}

afterEach(async () => {
    for (const app of apps.splice(0)) {
        await app.instance.dispose?.();
    }

    for (const host of hosts.splice(0)) {
        // Anything a failed start left behind is killed here so no fake server survives the run.
        for (const spawn of host.spawns) {
            if (isAlive(spawn.pid)) {
                try {
                    process.kill(spawn.pid);
                } catch {
                    // gone already
                }
            }
        }

        rmSync(host.dataDir, { recursive: true, force: true });
    }
});

describe("GET /status", () => {
    test("reports the process, the installed files, the config and the platform", async () => {
        const app = await makeApp();
        const response = await request(app, "GET", "/status");

        expect(response.status).toBe(200);
        expect(response.headers.get("content-type")).toContain("application/json");

        const body = await read(response);

        expect(body.process).toEqual({ state: "stopped" });
        expect(body.installer.binary).toMatchObject({ installed: true, tag: TAG, backend: BACKEND, path: app.binaryPath });
        expect(body.installer.model).toMatchObject({ installed: true, path: app.modelPath });
        expect(body.config).toMatchObject({ device: "cpu", modelPath: app.modelPath, profile: "default" });
        expect(body.platform).toEqual({ os: process.platform, arch: process.arch });
    });

    test("answers a fresh data dir with the defaults instead of an error", async () => {
        const host = makeHost();
        const app = track({ host, instance: await activate(host), binaryPath: "", modelPath: "" });
        const body = await read(await request(app, "GET", "/status"));

        expect(body.process.state).toBe("stopped");
        expect(body.config).toMatchObject({ profile: "default", ctx: 4096, device: "auto", releaseTag: TAG });
        expect(body.installer.binary.installed).toBe(false);
        expect(body.installer.model.installed).toBe(false);
    });

    test("accepts one trailing slash and nothing else", async () => {
        const app = await makeApp();

        const plain = await read(await request(app, "GET", "/status"));
        const slashed = await read(await request(app, "GET", "/status/"));

        expect(slashed).toEqual(plain);
        expect((await request(app, "GET", "/status//")).status).toBe(404);
        expect((await request(app, "GET", "/status/x")).status).toBe(404);
    });
});

describe("the route table is closed", () => {
    test("unknown paths are 404 with a JSON error body", async () => {
        const app = await makeApp();

        for (const subpath of ["/", "", "/nope", "/config/extra", "/constructor", "/__proto__"]) {
            const response = await request(app, "GET", subpath);
            const body = await read<{ error: { code: string; message: string } }>(response);

            expect(response.status).toBe(404);
            expect(body.error.code).toBe("not-found");
            expect(typeof body.error.message).toBe("string");
        }
    });

    test("a known path with the wrong method is 405 and names the allowed ones", async () => {
        const app = await makeApp();
        const cases: [string, string, string][] = [
            ["GET", "/embed", "POST"],
            ["POST", "/status", "GET"],
            ["PUT", "/config", "POST"],
            ["DELETE", "/stop", "POST"],
            // Bun normalises a made-up method such as CONSTRUCTOR to GET, so these two are the
            // real methods a caller can send that the table does not list.
            ["HEAD", "/status", "GET"],
            ["OPTIONS", "/embed", "POST"],
        ];

        for (const [method, subpath, allow] of cases) {
            const response = await request(app, method, subpath);

            expect(response.status).toBe(405);
            expect(response.headers.get("allow")).toBe(allow);
            expect((await read<{ error: { code: string } }>(response)).error.code).toBe("method-not-allowed");
        }
    });
});

describe("unexpected failures", () => {
    test("answer 500 internal, with the details only in the log", async () => {
        const host = makeHost();
        // A directory where config.json should be: the atomic rename cannot replace it, so the
        // write fails with something that is not a ConfigError.
        await mkdir(join(host.dataDir, "config.json"), { recursive: true });

        const app = track({ host, instance: await activate(host), binaryPath: "", modelPath: "" });
        const response = await request(app, "POST", "/config", { ctx: 2048 });
        const body = await read<{ error: { code: string; message: string } }>(response);

        expect(response.status).toBe(500);
        expect(body.error.code).toBe("internal");
        // Generic text: no path, no stack trace.
        expect(body.error.message).not.toContain(host.dataDir);
        expect(app.host.logs.some((line) => line.level === "error")).toBe(true);
    });
});

describe("POST /config", () => {
    test("saves, answers with the saved config and shows up in /status", async () => {
        const app = await makeApp();
        const response = await request(app, "POST", "/config", { ctx: 2048, device: "gpu", gpuBackend: "vulkan" });

        expect(response.status).toBe(200);

        const saved = await read(response);

        expect(saved).toMatchObject({ ctx: 2048, device: "gpu", gpuBackend: "vulkan", modelPath: app.modelPath });
        // Persisted, not just echoed, and merged over the previous file.
        expect(JSON.parse(await readFile(join(app.host.dataDir, "config.json"), "utf8"))).toMatchObject({
            ctx: 2048,
            device: "gpu",
            profile: "default",
        });

        const status = await read(await request(app, "GET", "/status"));

        expect(status.config.ctx).toBe(2048);
    });

    test("rejects a config config.ts does not accept without writing anything", async () => {
        const app = await makeApp();
        const before = await readFile(join(app.host.dataDir, "config.json"), "utf8");
        const response = await request(app, "POST", "/config", { device: "quantum" });
        const body = await read<{ error: { code: string; message: string } }>(response);

        expect(response.status).toBe(400);
        expect(body.error.code).toBe("invalid-config");
        expect(body.error.message).toContain("device");
        expect(await readFile(join(app.host.dataDir, "config.json"), "utf8")).toBe(before);
    });

    test("rejects a non-object and broken JSON as bad-request", async () => {
        const app = await makeApp();

        for (const body of ["[]", '"text"', "42", "null", "{oops", ""]) {
            const response = await request(app, "POST", "/config", body);

            expect(response.status).toBe(400);
            expect((await read<{ error: { code: string } }>(response)).error.code).toBe("bad-request");
        }
    });
});

describe("POST /embed", () => {
    test("returns one vector per text and starts the server once", async () => {
        const app = await makeApp();
        const response = await request(app, "POST", "/embed", { texts: ["alpha", "beta", "gamma"] });

        expect(response.status).toBe(200);

        const { vectors } = await read<{ vectors: number[][] }>(response);

        expect(vectors).toHaveLength(3);
        expect(vectors[0]).toHaveLength(8);
        expect(Math.hypot(...vectors[0])).toBeCloseTo(1, 6);
        expect(vectors[0]).not.toEqual(vectors[1]);
        expect(app.host.spawns).toHaveLength(1);

        // A later /start reuses the running server instead of restarting it.
        const started = await read(await request(app, "POST", "/start"));

        expect(started.state).toBe("ready");
        expect(app.host.spawns).toHaveLength(1);
    });

    test("refuses malformed batches before starting anything", async () => {
        const app = await makeApp();
        const cases: unknown[] = [
            {},
            { texts: "one" },
            { texts: [] },
            { texts: Array.from({ length: MAX_BATCH_TEXTS + 1 }, (_, index) => `t${index}`) },
            { texts: ["ok", 1] },
            { texts: ["ok", null] },
            { texts: ["x".repeat(8001)] },
        ];

        for (const body of cases) {
            const response = await request(app, "POST", "/embed", body);

            expect(response.status).toBe(400);
            expect((await read<{ error: { code: string } }>(response)).error.code).toBe("bad-request");
        }

        expect(app.host.spawns).toHaveLength(0);
    });

    test("answers 413 for a body over 2 MB, whatever content-length claims", async () => {
        const app = await makeApp();
        const stream = (bytes: number): ReadableStream<Uint8Array> => {
            const chunk = new Uint8Array(64 * 1024).fill(0x61);
            let sent = 0;

            return new ReadableStream({
                pull(controller) {
                    if (sent >= bytes) {
                        controller.close();

                        return;
                    }

                    const size = Math.min(chunk.byteLength, bytes - sent);

                    controller.enqueue(chunk.subarray(0, size));
                    sent += size;
                },
            });
        };

        const honest = await handle(app)(
            new Request("http://127.0.0.1/api/plugins/smiley-embeddings-memory/host/embed", {
                method: "POST",
                body: stream(2 * 1024 * 1024 + 1),
                duplex: "half",
            } as RequestInit),
            "/embed",
        );
        const lying = await handle(app)(
            new Request("http://127.0.0.1/api/plugins/smiley-embeddings-memory/host/embed", {
                method: "POST",
                body: stream(3 * 1024 * 1024),
                headers: { "content-length": "5" },
                duplex: "half",
            } as RequestInit),
            "/embed",
        );

        expect(honest.status).toBe(413);
        expect((await read<{ error: { code: string } }>(honest)).error.code).toBe("too-large");
        // The declared length is not trusted: the cap is applied while reading.
        expect(lying.status).toBe(413);
        expect(app.host.spawns).toHaveLength(0);
    });

    test("answers 409 not-installed when nothing is on disk", async () => {
        const app = await makeApp({ install: "none" });

        for (const subpath of ["/embed", "/start"]) {
            const response = await request(app, "POST", subpath, subpath === "/embed" ? { texts: ["a"] } : undefined);

            expect(response.status).toBe(409);
            expect((await read<{ error: { code: string; message: string } }>(response)).error.code).toBe("not-installed");
        }

        expect(app.host.spawns).toHaveLength(0);
    });

    test("maps an over-long text to 413 too-long", async () => {
        const app = await makeApp({ behaviors: { [BUILD_KEY]: { FAKE_MAX_CHARS: "32" } } });
        const response = await request(app, "POST", "/embed", { texts: ["x".repeat(4000)] });
        const body = await read<{ error: { code: string; message: string } }>(response);

        expect(response.status).toBe(413);
        expect(body.error.code).toBe("too-long");
        expect(body.error.message).toContain("exceeds the available context size");
    });

    test("never logs the text it embeds", async () => {
        const app = await makeApp();

        await request(app, "POST", "/embed", { texts: ["a secret only the user knows"] });

        expect(app.host.logs.some((line) => line.message.includes("a secret only the user knows"))).toBe(false);
        expect(app.host.logs.some((line) => line.message.includes("embedded 1 text"))).toBe(true);
    });
});

describe("POST /start and POST /stop", () => {
    test("answers 502 starting-failed with the manager's actionable message", async () => {
        const app = await makeApp({ behaviors: { [BUILD_KEY]: { FAKE_FAIL: "1" } } });
        const response = await request(app, "POST", "/start");
        const body = await read<{ error: { code: string; message: string } }>(response);

        expect(response.status).toBe(502);
        expect(body.error.code).toBe("starting-failed");
        expect(body.error.message).toContain("failed to start");
    });

    test("stops the running server and stays quiet when nothing runs", async () => {
        const app = await makeApp();

        await request(app, "POST", "/start");
        const pid = app.host.spawns[0].pid;

        expect((await request(app, "POST", "/stop")).status).toBe(200);
        expect(isAlive(pid)).toBe(false);

        const second = await request(app, "POST", "/stop");

        expect(second.status).toBe(200);
        expect(await read(second)).toEqual({ ok: true });
    });
});

describe("install and download routes", () => {
    test("answer 202 with the task status, and cancel is a no-op when idle", async () => {
        const app = await makeApp();
        // A manual binary path makes startBinaryInstall answer from the config, so this test never
        // reaches the network (a manual model path, set by makeApp, does the same for the model).
        const manualBinary = join(app.host.dataDir, "manual", "llama-server.exe");

        await mkdir(dirname(manualBinary), { recursive: true });
        await writeFile(manualBinary, "binary");
        await request(app, "POST", "/config", { binaryPath: manualBinary });

        const binary = await request(app, "POST", "/binary/install");
        const model = await request(app, "POST", "/models/download");

        expect(binary.status).toBe(202);
        expect(model.status).toBe(202);
        expect((await read<{ message?: string }>(binary)).message).toContain("manual binary path");
        expect((await read<{ message?: string }>(model)).message).toContain("manual model path");

        for (const subpath of ["/binary/cancel", "/models/cancel"]) {
            const response = await request(app, "POST", subpath);

            expect(response.status).toBe(200);
            expect(await read(response)).toEqual({ ok: true });
        }
    });

    test("DELETE /binary stops the server before removing the builds", async () => {
        const app = await makeApp();

        await request(app, "POST", "/start");
        const pid = app.host.spawns[0].pid;

        const response = await request(app, "DELETE", "/binary");

        expect(response.status).toBe(200);
        expect(await read(response)).toEqual({ ok: true });
        expect(isAlive(pid)).toBe(false);
        expect(existsSync(join(app.host.dataDir, "bin"))).toBe(false);

        const status = await read(await request(app, "GET", "/status"));

        expect(status.installer.binary.installed).toBe(false);
        // Only the build went; the model file stays.
        expect(status.installer.model.installed).toBe(true);
    });

    test("DELETE /models stops the server before removing the model files", async () => {
        const app = await makeApp();

        await request(app, "POST", "/start");
        const pid = app.host.spawns[0].pid;

        const response = await request(app, "DELETE", "/models");

        expect(response.status).toBe(200);
        expect(isAlive(pid)).toBe(false);
        expect(existsSync(app.modelPath)).toBe(false);
        // The build is untouched: only the model was deleted.
        expect(existsSync(app.binaryPath)).toBe(true);
    });
});

describe("dispose", () => {
    test("kills the fake server and can be called twice", async () => {
        const app = await makeApp();

        await request(app, "POST", "/start");
        const pid = app.host.spawns[0].pid;

        expect(isAlive(pid)).toBe(true);

        await app.instance.dispose?.();
        await app.instance.dispose?.();

        expect(isAlive(pid)).toBe(false);
    });
});

describe("the bundled artifact", () => {
    test("builds, imports only builtins and answers /status", async () => {
        const outDir = mkdtempSync(join(tmpdir(), "smiley-bundle-"));

        try {
            const outfile = join(outDir, "server.js");
            const build = spawnSync(
                process.execPath,
                ["build", join(PLUGIN_ROOT, "src", "server", "main.ts"), "--target", "bun", "--format", "esm", "--outfile", outfile],
                { cwd: PLUGIN_ROOT, encoding: "utf8" },
            );

            expect(build.status).toBe(0);
            expect(existsSync(outfile)).toBe(true);

            const text = await readFile(outfile, "utf8");
            const specifiers = [
                ...[...text.matchAll(/\bfrom\s*["']([^"']+)["']/g)].map((match) => match[1]),
                ...[...text.matchAll(/^\s*import\s*["']([^"']+)["']/gm)].map((match) => match[1]),
            ];
            // Bun rewrites `node:fs` to `fs` when bundling, so both spellings count as builtin.
            const builtins = new Set([...builtinModules, "bun"]);
            const bare = specifiers.filter((specifier) => !builtins.has(specifier.replace(/^node:/, "")));

            // The app hands the module nothing but the runtime's own modules, so a dependency left
            // as a bare specifier would fail to resolve when the module is loaded.
            expect(bare).toEqual([]);
            // fflate is the plugin's only dependency and the build inlines it.
            expect(text).toContain("UnzipInflate");

            console.info(`[bundle] dist/server.js would be ${Buffer.byteLength(text)} bytes`);

            const module = (await import(pathToFileURL(outfile).href)) as { activate: typeof activate };

            expect(typeof module.activate).toBe("function");

            const host = makeHost();
            const app = track({ host, instance: await module.activate(host), binaryPath: "", modelPath: "" });
            const response = await handle(app)(
                new Request("http://127.0.0.1/api/plugins/smiley-embeddings-memory/host/status"),
                "/status",
            );
            const body = await read(response);

            expect(response.status).toBe(200);
            expect(body.process.state).toBe("stopped");
            expect(body.config.profile).toBe("default");
            expect(Object.keys(body).sort()).toEqual(["config", "installer", "platform", "process"]);
        } finally {
            rmSync(outDir, { recursive: true, force: true });
        }
    });
});

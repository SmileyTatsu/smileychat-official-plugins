import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { EmbedError, LlamaProcess, MAX_BATCH_TEXTS } from "./llama-process";
import type { HostLogger, LaunchSpec, PluginServerHost } from "./types";

// The fixture is started as a real process with the same argument list the manager would give
// llama-server; only `command` is dropped, so the plugin never needs the multi-GB binary on disk.
const FIXTURE = join(import.meta.dir, "fixtures", "fake-llama-server.ts");

type SpawnRecord = { command: string; args: string[]; cwd?: string; env?: Record<string, string>; pid: number; killed: boolean };

type Behaviors = Record<string, Record<string, string>>;

type FakeHost = PluginServerHost & { spawns: SpawnRecord[]; logs: string[] };

const live: LlamaProcess[] = [];
const hosts: FakeHost[] = [];

function makeHost(behaviors: Behaviors = {}): FakeHost {
    const dataDir = mkdtempSync(join(tmpdir(), "llama-process-"));
    const spawns: SpawnRecord[] = [];
    const logs: string[] = [];
    const logger: HostLogger = {
        debug: () => {},
        info: (message) => logs.push(message),
        warn: (message) => logs.push(message),
        error: (message) => logs.push(message),
    };

    const host: FakeHost = {
        pluginId: "smiley-embeddings-memory",
        dataDir,
        logger,
        spawns,
        logs,
        spawn(command, args = [], options = {}) {
            // Behaviour is keyed by the build in `command` (cuda/vulkan/cpu), which is how a real
            // machine differs too: the binary decides the device it can log. The map is read on
            // every spawn so a test can flip it mid-run.
            const build = command.includes("vulkan") ? "vulkan" : command.includes("cpu") ? "cpu" : "cuda";
            const behavior = behaviors[build] ?? {};
            const child = Bun.spawn([process.execPath, FIXTURE, ...args], {
                env: { ...process.env, ...options.env, FAKE_BACKEND: build, ...behavior },
                stdout: "ignore",
                stderr: "ignore",
            });

            spawns.push({ command, args, cwd: options.cwd, env: options.env, pid: child.pid, killed: false });
            const record = spawns[spawns.length - 1];

            return {
                pid: child.pid,
                exited: child.exited,
                kill: async () => {
                    record.killed = true;
                    child.kill();
                    await child.exited;
                },
            };
        },
    };

    hosts.push(host);

    return host;
}

function makeProcess(host: FakeHost, overrides: { startTimeoutMs?: number } = {}): LlamaProcess {
    const manager = new LlamaProcess(host, {
        startTimeoutMs: overrides.startTimeoutMs ?? 10_000,
        healthIntervalMs: 25,
    });

    live.push(manager);

    return manager;
}

function spec(overrides: Partial<LaunchSpec> = {}): LaunchSpec {
    return {
        binary: join("fake-bin", "cuda", "llama-server.exe"),
        model: join("fake-models", "embeddinggemma.gguf"),
        backend: "cuda",
        gpuLayers: 99,
        pooling: "mean",
        ctx: 4096,
        ...overrides,
    };
}

const vulkanSpec = (): LaunchSpec => spec({ binary: join("fake-bin", "vulkan", "llama-server.exe"), backend: "vulkan" });

async function waitFor(check: () => boolean, timeoutMs = 8000): Promise<void> {
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
        if (check()) {
            return;
        }

        await Bun.sleep(20);
    }

    throw new Error("condition was not met in time");
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
    for (const manager of live.splice(0)) {
        await manager.dispose();
    }

    for (const host of hosts.splice(0)) {
        // Anything the manager failed to stop itself (a failed start) is killed here so no fake
        // server survives the test run.
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

describe("LlamaProcess", () => {
    test("starts the server, reads device info from its log and embeds", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "gpu", 0);

        const status = manager.status();
        expect(status.state).toBe("ready");
        expect(status.backend).toBe("cuda");
        expect(status.device).toBe("CUDA0");
        expect(status.layersOffloaded).toBe(25);
        expect(status.layersTotal).toBe(25);
        expect(status.pid).toBeGreaterThan(0);
        expect(status.port).toBeGreaterThan(0);
        expect(status.requestedDevice).toBe("gpu");
        expect(status.lastError).toBeUndefined();
        expect(typeof status.startedAt).toBe("string");
        expect(typeof status.lastUsedAt).toBe("string");

        const vectors = await manager.embed(["alpha", "beta", "gamma"]);
        expect(vectors).toHaveLength(3);
        expect(vectors[0]).toHaveLength(8);
        expect(Math.hypot(...vectors[0])).toBeCloseTo(1, 6);
        expect(vectors[0]).not.toEqual(vectors[1]);

        // `index` decides the position: the same text alone must produce the vector it had inside
        // the batch, otherwise a permuted answer would silently mislabel chunks.
        const single = await manager.embed(["beta"]);
        expect(single[0]).toEqual(vectors[1]);
    });

    test("spawns with the fixed argument list and never with a caller-supplied value", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec({ ctx: 3072, pooling: "last" })], "gpu", 0);

        const spawn = host.spawns[0];
        const value = (flag: string): string | undefined => spawn.args[spawn.args.indexOf(flag) + 1];

        expect(spawn.command).toBe(spec().binary);
        expect(spawn.cwd).toBe(join("fake-bin", "cuda"));
        expect(value("--host")).toBe("127.0.0.1");
        expect(value("--pooling")).toBe("last");
        expect(value("-c")).toBe("3072");
        expect(value("-b")).toBe("2048");
        expect(value("-ub")).toBe("2048");
        expect(value("-np")).toBe("4");
        expect(value("-ngl")).toBe("99");
        expect(value("-lv")).toBe("4");
        expect(value("--log-file")).toBe(join(host.dataDir, "llama-server.log"));
        expect(spawn.args).toContain("--embedding");
        expect(spawn.args).toContain("--no-webui");
    });

    test("shares one start between concurrent ensureReady calls", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await Promise.all([
            manager.ensureReady([spec()], "auto", 0),
            manager.ensureReady([spec()], "auto", 0),
            manager.ensureReady([spec()], "auto", 0),
        ]);

        expect(host.spawns).toHaveLength(1);
        expect(manager.status().state).toBe("ready");
    });

    test("reuses a running server with an identical launch spec", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        const firstPid = manager.status().pid;

        await manager.ensureReady([spec()], "auto", 0);

        expect(host.spawns).toHaveLength(1);
        expect(manager.status().pid).toBe(firstPid as number);
    });

    test("restarts when the launch spec changes", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        await manager.ensureReady([spec({ ctx: 2048 })], "auto", 0);

        expect(host.spawns).toHaveLength(2);
        expect(host.spawns[0].killed).toBe(true);
        expect(manager.status().state).toBe("ready");
        expect(manager.status().pid).toBe(host.spawns[1].pid);
    });

    test("falls back to the next candidate under auto and says so", async () => {
        const behaviors: Behaviors = { cuda: { FAKE_FAIL: "1" } };
        const host = makeHost(behaviors);
        const manager = makeProcess(host);

        await manager.ensureReady([spec(), vulkanSpec()], "auto", 0);

        const status = manager.status();
        expect(status.state).toBe("ready");
        expect(status.backend).toBe("vulkan");
        expect(status.device).toBe("Vulkan0");
        expect(host.spawns).toHaveLength(2);
        expect(status.lastError).toContain("CUDA build failed to start");
        expect(status.lastError).toContain("fell back to Vulkan");
        expect(host.logs.some((line) => line.includes("fell back to Vulkan"))).toBe(true);
    });

    test("does not fall back when the device is gpu", async () => {
        const host = makeHost({ cuda: { FAKE_FAIL: "1" } });
        const manager = makeProcess(host);

        const failure = await manager.ensureReady([spec(), vulkanSpec()], "gpu", 0).catch((error: Error) => error);

        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toContain("llama-server (CUDA) failed to start");
        expect((failure as Error).message).toContain("does not fall back to CPU");
        expect(host.spawns).toHaveLength(1);
        expect(manager.status().state).toBe("error");
        expect(manager.status().lastError).toContain("failed to start");
    });

    test("reports a cpu run when a GPU build offloads nothing", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        // The real trap from the design notes: `-ngl 0` still logs `using device CUDA0`.
        await manager.ensureReady([spec({ gpuLayers: 0, backend: "manual" })], "gpu", 0);

        const status = manager.status();
        expect(status.device).toBe("CUDA0");
        expect(status.backend).toBe("manual");
        expect(status.layersOffloaded).toBe(0);
        expect(status.layersTotal).toBe(25);
    });

    test("stops the server when the idle timer expires", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0.005);
        const pid = manager.status().pid as number;

        await waitFor(() => manager.status().state === "stopped");

        expect(host.logs.some((line) => line === "idle: stopped llama-server")).toBe(true);
        expect(isAlive(pid)).toBe(false);
        expect(manager.status().pid).toBeUndefined();
    });

    test("keeps the server alive with idleMinutes 0", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        const pid = manager.status().pid as number;

        await Bun.sleep(200);

        expect(manager.status().state).toBe("ready");
        expect(isAlive(pid)).toBe(true);
    });

    test("marks an unexpected exit as an error and starts again on the next call", async () => {
        const behaviors: Behaviors = { cuda: { FAKE_EXIT_AFTER_MS: "150" } };
        const host = makeHost(behaviors);
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        await waitFor(() => manager.status().state === "error");

        expect(manager.status().lastError).toContain("exited with code 3");
        expect(host.logs.some((line) => line.includes("exited with code 3"))).toBe(true);
        // No automatic restart loop.
        await Bun.sleep(150);
        expect(host.spawns).toHaveLength(1);

        delete behaviors.cuda;
        await manager.ensureReady([spec()], "auto", 0);

        expect(host.spawns).toHaveLength(2);
        expect(manager.status().state).toBe("ready");
    });

    test("times out a start with the first-load hint", async () => {
        const host = makeHost({ cuda: { FAKE_HEALTH_DELAY_MS: "8000" } });
        const manager = makeProcess(host, { startTimeoutMs: 400 });

        const failure = await manager.ensureReady([spec()], "gpu", 0).catch((error: Error) => error);

        expect((failure as Error).message).toContain("antivirus");
        expect(manager.status().state).toBe("error");
        expect(manager.status().lastError).toContain("did not answer /health");
        await waitFor(() => !isAlive(host.spawns[0].pid));
    });

    test("maps an over-long input to a too-long error", async () => {
        const host = makeHost({ cuda: { FAKE_MAX_CHARS: "32" } });
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        const failure = await manager.embed(["short", "x".repeat(64)]).catch((error: unknown) => error);

        expect(failure).toBeInstanceOf(EmbedError);
        expect((failure as EmbedError).code).toBe("too-long");
        expect((failure as EmbedError).message).toContain("exceeds the available context size");
        // A rejected input is the caller's problem: the server is still up.
        expect(manager.status().state).toBe("ready");
    });

    test("maps a gemma-style 500 'too large to process' to too-long", async () => {
        // EmbeddingGemma's non-causal pooling reports the same problem as HTTP 500, and the
        // caller must still see "split this text", not "the server is broken".
        const host = makeHost({ cuda: { FAKE_MAX_CHARS: "32", FAKE_TOO_LONG_500: "1" } });
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        const failure = await manager.embed(["x".repeat(64)]).catch((error: unknown) => error);

        expect((failure as EmbedError).code).toBe("too-long");
        expect((failure as EmbedError).message).toContain("too large to process");
        expect(manager.status().state).toBe("ready");
    });

    test("rejects a batch above MAX_BATCH_TEXTS and returns an empty batch untouched", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);

        const oversized = await manager
            .embed(Array.from({ length: MAX_BATCH_TEXTS + 1 }, (_, index) => `text ${index}`))
            .catch((error: unknown) => error);

        expect((oversized as EmbedError).code).toBe("bad-request");
        expect(await manager.embed([])).toEqual([]);
        expect(host.logs.some((line) => line.startsWith("llama-server starting"))).toBe(true);
    });

    test("refuses to embed before the server is ready", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        const failure = await manager.embed(["a"]).catch((error: unknown) => error);

        expect((failure as EmbedError).code).toBe("not-ready");
        expect(host.spawns).toHaveLength(0);
    });

    test("retries a port that was taken between probe and bind", async () => {
        const behaviors: Behaviors = {};
        const host = makeHost(behaviors);
        // The fixture fails its first start with llama.cpp's bind error, then behaves normally.
        behaviors.cuda = { FAKE_EADDR_ONCE_FILE: join(host.dataDir, "bind-once") };
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "gpu", 0);

        expect(host.spawns).toHaveLength(2);
        expect(manager.status().state).toBe("ready");
        expect(host.logs.some((line) => line.includes("could not bind its port"))).toBe(true);
    });

    test("stop kills the process and clears the status", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        const pid = manager.status().pid as number;

        await manager.stop();

        expect(isAlive(pid)).toBe(false);
        expect(manager.status().state).toBe("stopped");
        expect(manager.status().pid).toBeUndefined();
        expect(manager.status().port).toBeUndefined();
        expect(manager.status().backend).toBeUndefined();
    });

    test("dispose stops the server and refuses further starts", async () => {
        const host = makeHost();
        const manager = makeProcess(host);

        await manager.ensureReady([spec()], "auto", 0);
        const pid = manager.status().pid as number;
        await manager.dispose();

        expect(isAlive(pid)).toBe(false);
        await expect(manager.ensureReady([spec()], "auto", 0)).rejects.toThrow(/disposed/);
        await expect(manager.embed(["a"])).rejects.toThrow(/not available/);
        expect(host.spawns).toHaveLength(1);
    });
});

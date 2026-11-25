import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { unzipSync, zipSync } from "fflate";

import { LOCAL_MODELS } from "../models";
import type { DownloadOptions } from "./download";
import { DownloadError } from "./download";
import { Installer, type InstallerDeps } from "./installer";
import type { Platform } from "./releases";
import { DEFAULT_SERVER_CONFIG, type HostLogger, type ServerConfig, type TaskStatus } from "./types";

// Same shape as the fixture in releases.test.ts: the real nvidia-smi header, trimmed.
const CUDA_13 = "| NVIDIA-SMI 570.00      Driver Version: 570.00      CUDA Version: 13.0     |";

const WIN_X64: Platform = { os: "win32", arch: "x64" };
const LINUX_ARM64: Platform = { os: "linux", arch: "arm64" };

const MARKER = ".installed.json";

const CUDA_ASSET = "llama-b12686-bin-win-cuda-13.5-x64.zip";
const CUDART_ASSET = "cudart-llama-bin-win-cuda-13.5-x64.zip";

let workDir = "";

beforeEach(async () => {
    workDir = await mkdtemp(join(tmpdir(), "smiley-installer-"));
});

afterEach(async () => {
    await rm(workDir, { recursive: true, force: true });
});

function silentLogger(): HostLogger {
    const noop = (): void => undefined;

    return { debug: noop, info: noop, warn: noop, error: noop };
}

function config(patch: Partial<ServerConfig> = {}): ServerConfig {
    return { ...DEFAULT_SERVER_CONFIG, ...patch };
}

function sha256(bytes: Uint8Array): string {
    return new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
}

function zip(files: Record<string, string>): Uint8Array {
    const entries: Record<string, Uint8Array> = {};

    for (const [name, text] of Object.entries(files)) {
        entries[name] = new TextEncoder().encode(text);
    }

    return zipSync(entries);
}

type AssetFixture = { name: string; bytes: Uint8Array; digest?: string };

function assetUrl(tag: string, name: string): string {
    return `https://github.com/ggml-org/llama.cpp/releases/download/${tag}/${name}`;
}

/** One GitHub release object, as `GET /releases/tags/<tag>` returns it. */
function releasePayload(tag: string, assets: AssetFixture[]): unknown {
    return {
        tag_name: tag,
        assets: assets.map((asset) => ({
            name: asset.name,
            size: asset.bytes.length,
            browser_download_url: assetUrl(tag, asset.name),
            digest: asset.digest ?? `sha256:${sha256(asset.bytes)}`,
        })),
    };
}

function bodiesOf(tag: string, assets: AssetFixture[]): Map<string, Uint8Array> {
    return new Map(assets.map((asset) => [assetUrl(tag, asset.name), asset.bytes] as [string, Uint8Array]));
}

type DownloadCall = {
    url: string;
    dest: string;
    expectedBytes?: number;
    sha256?: string;
    /** Bytes of `.part` already on disk when the call started: proves the resume. */
    resumedFrom: number;
};

type HarnessOptions = {
    bodies?: Map<string, Uint8Array>;
    fetchJson?: InstallerDeps["fetchJson"];
    platform?: Platform;
    nvidiaSmi?: string | null;
};

type Harness = {
    installer: Installer;
    calls: DownloadCall[];
    fetchCalls: string[];
    extractCalls: string[];
    /** Whether the install marker was already there when an archive was unpacked. */
    markerSeenDuringExtract: boolean[];
    control: { hook: (() => void) | null; hold: Promise<void> | null; paceMs: number };
};

function harness(options: HarnessOptions = {}): Harness {
    const bodies = options.bodies ?? new Map<string, Uint8Array>();
    const calls: DownloadCall[] = [];
    const fetchCalls: string[] = [];
    const extractCalls: string[] = [];
    const markerSeenDuringExtract: boolean[] = [];
    const control: Harness["control"] = { hook: null, hold: null, paceMs: 2 };

    // Stands in for downloadFile: same .part-then-rename dance, no socket, and it reports
    // progress twice per asset so the aggregated counter can be observed.
    const download: InstallerDeps["download"] = async (downloadOptions: DownloadOptions) => {
        const body = bodies.get(downloadOptions.url);

        if (!body) {
            throw new DownloadError("http", `HTTP 404 for ${downloadOptions.url}`);
        }

        await mkdir(dirname(downloadOptions.dest), { recursive: true });

        let resumedFrom = 0;

        try {
            resumedFrom = (await stat(`${downloadOptions.dest}.part`)).size;
        } catch {
            resumedFrom = 0;
        }

        calls.push({
            url: downloadOptions.url,
            dest: downloadOptions.dest,
            expectedBytes: downloadOptions.expectedBytes,
            sha256: downloadOptions.sha256,
            resumedFrom,
        });

        const total = downloadOptions.expectedBytes ?? body.length;
        const half = Math.max(1, Math.ceil(body.length / 2));

        await writeFile(`${downloadOptions.dest}.part`, body.subarray(0, half));
        downloadOptions.onProgress?.({ received: half, total });
        await Bun.sleep(control.paceMs);

        const hook = control.hook;

        if (hook) {
            control.hook = null;
            hook();
        }

        if (downloadOptions.signal?.aborted) {
            throw new DownloadError("aborted", "download aborted");
        }

        if (control.hold) {
            await control.hold;

            if (downloadOptions.signal?.aborted) {
                throw new DownloadError("aborted", "download aborted");
            }
        }

        await writeFile(`${downloadOptions.dest}.part`, body);
        downloadOptions.onProgress?.({ received: body.length, total });
        await rename(`${downloadOptions.dest}.part`, downloadOptions.dest);
    };

    const fetchJson: InstallerDeps["fetchJson"] = async (url, signal) => {
        fetchCalls.push(url);

        if (!options.fetchJson) {
            throw new Error(`unexpected fetch: ${url}`);
        }

        return options.fetchJson(url, signal);
    };

    // Real zip parsing (fflate), so "find llama-server inside a nested folder" is exercised on
    // files that a real archive would produce.
    const extract: InstallerDeps["extract"] = async (archive, dest) => {
        extractCalls.push(archive);
        markerSeenDuringExtract.push(await fileExists(join(dest, MARKER)));

        const data = new Uint8Array(await Bun.file(archive).arrayBuffer());
        const written: string[] = [];

        for (const [name, bytes] of Object.entries(unzipSync(data))) {
            const target = join(dest, name);

            await mkdir(dirname(target), { recursive: true });
            await writeFile(target, bytes);
            written.push(name);
        }

        return written;
    };

    const installer = new Installer(
        { dataDir: workDir, logger: silentLogger() },
        {
            download,
            fetchJson,
            extract,
            runNvidiaSmi: async () => options.nvidiaSmi ?? null,
            platform: options.platform ?? WIN_X64,
        },
    );

    return { installer, calls, fetchCalls, extractCalls, markerSeenDuringExtract, control };
}

async function fileExists(path: string): Promise<boolean> {
    try {
        return (await stat(path)).isFile();
    } catch {
        return false;
    }
}

async function waitForTask(
    installer: Installer,
    serverConfig: ServerConfig,
    which: "binary" | "model",
    onSnapshot?: (task: TaskStatus) => void,
): Promise<TaskStatus> {
    const deadline = Date.now() + 8000;

    for (;;) {
        const status = await installer.status(serverConfig);
        const task = which === "binary" ? status.binary.task : status.model.task;

        onSnapshot?.(task);

        if (task.state !== "running" && task.state !== "idle") {
            return task;
        }

        if (Date.now() > deadline) {
            throw new Error(`task stayed ${task.state}: ${task.message ?? ""}`);
        }

        await Bun.sleep(1);
    }
}

/** A build that looks installed: marker plus the binary it names, one level down as in a zip. */
async function installFakeBuild(tag: string, backend: string): Promise<string> {
    const dir = join(workDir, "bin", tag, backend);
    const binary = "build/bin/llama-server.exe";
    const binaryPath = join(dir, binary);

    await mkdir(dirname(binaryPath), { recursive: true });
    await writeFile(binaryPath, "binary");
    await writeFile(
        join(dir, MARKER),
        JSON.stringify({ tag, backend, assets: ["fake.zip"], installedAt: new Date().toISOString(), binary }),
    );

    return binaryPath;
}

async function writeFileOfSize(path: string, bytes: number): Promise<string> {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, new Uint8Array(bytes));

    return path;
}

/** The two-archive CUDA install the Windows path takes, and the bodies its URLs serve. */
function cudaInstallFixtures(tag: string): { assets: AssetFixture[]; bodies: Map<string, Uint8Array> } {
    const assets: AssetFixture[] = [
        { name: `llama-${tag}-bin-win-cuda-13.5-x64.zip`, bytes: zip({ "build/bin/llama-server.exe": "server", "build/bin/ggml.dll": "dll" }) },
        { name: `cudart-llama-bin-win-cuda-13.5-x64.zip`, bytes: zip({ "cudart64_12.dll": "cudart" }) },
    ];

    return { assets, bodies: bodiesOf(tag, assets) };
}

/** One release that offers both a CPU and a Vulkan build, for the device-choice tests. */
function cpuVulkanFixtures(tag: string): { assets: AssetFixture[]; bodies: Map<string, Uint8Array> } {
    const assets: AssetFixture[] = [
        { name: `llama-${tag}-bin-win-cpu-x64.zip`, bytes: zip({ "llama-server.exe": "cpu-build" }) },
        { name: `llama-${tag}-bin-win-vulkan-x64.zip`, bytes: zip({ "llama-server.exe": "vulkan-build" }) },
    ];

    return { assets, bodies: bodiesOf(tag, assets) };
}

describe("status", () => {
    test("reports nothing installed on an empty data dir", async () => {
        const { installer } = harness();
        const status = await installer.status(config());

        expect(status.binary).toEqual({
            installed: false,
            bytesOnDisk: 0,
            installedBackends: [],
            task: { state: "idle", received: 0, total: 0 },
        });
        expect(status.model).toEqual({
            installed: false,
            file: LOCAL_MODELS.default.file,
            path: join(workDir, "models", LOCAL_MODELS.default.file),
            expectedBytes: LOCAL_MODELS.default.bytes,
            bytesOnDisk: 0,
            task: { state: "idle", received: 0, total: 0 },
        });
    });

    test("reports an installed build and the model of the selected profile", async () => {
        const binaryPath = await installFakeBuild("b12686", "cpu");
        const spec = LOCAL_MODELS.strong;
        const modelPath = await writeFileOfSize(join(workDir, "models", spec.file), 128);
        const { installer } = harness();
        const status = await installer.status(config({ profile: "strong", device: "cpu" }));

        expect(status.binary).toMatchObject({ installed: true, tag: "b12686", backend: "cpu", path: binaryPath });
        expect(status.binary.bytesOnDisk).toBeGreaterThan(0);
        expect(status.model).toMatchObject({ installed: false, file: spec.file, path: modelPath });
        expect(status.model.bytesOnDisk).toBe(128);
    });

    test("does not offer a build whose marker is incomplete", async () => {
        const dir = join(workDir, "bin", "b12686", "cpu");

        await mkdir(dir, { recursive: true });
        await writeFile(join(dir, MARKER), JSON.stringify({ backend: "cpu", binary: "build/bin/llama-server" }));

        const { installer } = harness();
        const status = await installer.status(config());

        expect(status.binary.installed).toBe(false);
    });
});

describe("device-aware status", () => {
    test("a cpu build counts for device cpu and auto, not for gpu", async () => {
        const binaryPath = await installFakeBuild("b12686", "cpu");
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        expect((await installer.status(config({ device: "cpu" }))).binary).toMatchObject({
            installed: true,
            backend: "cpu",
            tag: "b12686",
            path: binaryPath,
            installedBackends: ["cpu"],
        });

        expect((await installer.status(config({ device: "auto" }))).binary.installed).toBe(true);
        // gpu means gpu: a CPU build is not an answer to "run on the GPU".
        expect((await installer.status(config({ device: "gpu" }))).binary).toMatchObject({
            installed: false,
            installedBackends: ["cpu"],
        });
    });

    test("a vulkan build is not installed for device cpu (the panel-skip bug)", async () => {
        await installFakeBuild("b12686", "vulkan");
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        const cpu = (await installer.status(config({ device: "cpu" }))).binary;

        expect(cpu.installed).toBe(false);
        // No path/tag/backend either: nothing may make the panel look ready for cpu.
        expect(cpu.path).toBeUndefined();
        expect(cpu.backend).toBeUndefined();
        expect(cpu.tag).toBeUndefined();
        // But the UI can still say what is there.
        expect(cpu.installedBackends).toEqual(["vulkan"]);

        expect((await installer.status(config({ device: "gpu" }))).binary).toMatchObject({ installed: true, backend: "vulkan" });
        expect((await installer.status(config({ device: "auto" }))).binary).toMatchObject({ installed: true, backend: "vulkan" });
    });

    test("with both builds it picks per device and lists them", async () => {
        const cpuPath = await installFakeBuild("b12686", "cpu");
        const vulkanPath = await installFakeBuild("b12686", "vulkan");
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        expect((await installer.status(config({ device: "cpu" }))).binary).toMatchObject({ path: cpuPath, backend: "cpu" });
        expect((await installer.status(config({ device: "gpu" }))).binary).toMatchObject({
            path: vulkanPath,
            backend: "vulkan",
        });
        // GPU first under auto.
        expect((await installer.status(config({ device: "auto" }))).binary).toMatchObject({
            path: vulkanPath,
            backend: "vulkan",
            installedBackends: ["vulkan", "cpu"],
        });
    });

    test("cuda plus cpu", async () => {
        const cudaPath = await installFakeBuild("b12686", "cuda");
        const cpuPath = await installFakeBuild("b12686", "cpu");
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        expect((await installer.status(config({ device: "auto" }))).binary).toMatchObject({
            path: cudaPath,
            backend: "cuda",
            installedBackends: ["cuda", "cpu"],
        });
        expect((await installer.status(config({ device: "gpu" }))).binary.path).toBe(cudaPath);
        expect((await installer.status(config({ device: "cpu" }))).binary.path).toBe(cpuPath);
    });

    test("gpuBackend narrows which GPU build counts", async () => {
        await installFakeBuild("b12686", "vulkan");
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        // Status does not probe, so a cuda build would count under "cuda"; a vulkan one never does.
        expect((await installer.status(config({ device: "gpu", gpuBackend: "cuda" }))).binary.installed).toBe(false);
        expect((await installer.status(config({ device: "gpu", gpuBackend: "vulkan" }))).binary.installed).toBe(true);

        await rm(join(workDir, "bin"), { recursive: true, force: true });
        await installFakeBuild("b12686", "cuda");

        expect((await installer.status(config({ device: "gpu", gpuBackend: "vulkan" }))).binary.installed).toBe(false);
        expect((await installer.status(config({ device: "gpu", gpuBackend: "cuda" }))).binary.installed).toBe(true);
        expect((await installer.status(config({ device: "gpu" }))).binary.installed).toBe(true);
    });

    test("a manual path decides installed, but the disk is still listed", async () => {
        await installFakeBuild("b12686", "vulkan");
        const manual = await writeFileOfSize(join(workDir, "manual", "llama-server.exe"), 8);
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        expect((await installer.status(config({ device: "cpu", binaryPath: manual }))).binary).toMatchObject({
            installed: true,
            backend: "manual",
            path: manual,
            installedBackends: ["vulkan"],
        });
        expect(
            (await installer.status(config({ device: "cpu", binaryPath: join(workDir, "gone", "llama-server.exe") })))
                .binary.installed,
        ).toBe(false);
    });
});

describe("resolveLaunch refusal text", () => {
    test("names what is missing for each device and the next step", async () => {
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: null });
        const next = "Open Semantic Memory settings and press Install and test.";

        const expected: Record<string, string> = {
            auto: `The local embedding program is not installed yet. ${next}`,
            gpu: `No GPU build of llama.cpp is installed. ${next}`,
            cpu: `No CPU build of llama.cpp is installed. ${next}`,
        };

        for (const [device, message] of Object.entries(expected)) {
            const resolved = await installer.resolveLaunch(
                config({ device: device as "auto" | "gpu" | "cpu", modelPath }),
            );

            expect(resolved.ok).toBe(false);

            if (resolved.ok) {
                return;
            }

            // The server's own message: main.ts must pass it on, not replace it with "model missing".
            expect(resolved.error).toBe(message);
        }
    });

    test("a build for the wrong device still refuses with the device's own text", async () => {
        await installFakeBuild("b12686", "vulkan");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        const resolved = await installer.resolveLaunch(config({ device: "cpu", modelPath }));

        expect(resolved.ok).toBe(false);

        if (resolved.ok) {
            return;
        }

        expect(resolved.error).toBe(
            "No CPU build of llama.cpp is installed. Open Semantic Memory settings and press Install and test.",
        );
    });
});

describe("binary install", () => {
    test("downloads both assets, aggregates progress, unpacks them together and marks last", async () => {
        const fixtures = cudaInstallFixtures("b12686");
        const { installer, calls, fetchCalls, extractCalls, markerSeenDuringExtract } = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });

        const serverConfig = config({ releaseTag: "b12686" });
        const started = installer.startBinaryInstall(serverConfig);
        const total = fixtures.assets[0].bytes.length + fixtures.assets[1].bytes.length;

        expect(started.state).toBe("running");

        const snapshots: TaskStatus[] = [];
        const final = await waitForTask(installer, serverConfig, "binary", (task) => snapshots.push({ ...task }));

        expect(final.state).toBe("done");
        expect(final.received).toBe(total);
        expect(final.total).toBe(total);

        // One counter across both assets: no snapshot ever restarts at the second asset's size.
        // (The first snapshot is taken before the plan's sizes are known, so it has total 0.)
        const planned = snapshots.filter((task) => task.total > 0);

        expect(planned.length).toBeGreaterThan(1);
        expect(planned.every((task) => task.total === total)).toBe(true);
        expect(planned.some((task) => task.received > fixtures.assets[0].bytes.length)).toBe(true);
        expect(planned.map((task) => task.received)).toEqual(
            [...planned.map((task) => task.received)].sort((left, right) => left - right),
        );

        expect(fetchCalls).toEqual([
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b12686",
        ]);
        expect(calls.map((call) => call.url)).toEqual(fixtures.assets.map((asset) => assetUrl("b12686", asset.name)));
        expect(calls.map((call) => call.dest)).toEqual(
            fixtures.assets.map((asset) => join(workDir, "downloads", asset.name)),
        );
        expect(calls[0].expectedBytes).toBe(fixtures.assets[0].bytes.length);
        // `sha256:` stripped, lowercase hex as downloadFile wants it.
        expect(calls[0].sha256).toBe(sha256(fixtures.assets[0].bytes));

        // Both archives go into one folder so the CUDA runtime sits beside the binary.
        const buildDir = join(workDir, "bin", "b12686", "cuda");
        const marker = JSON.parse(await readFile(join(buildDir, MARKER), "utf8")) as Record<string, unknown>;

        expect(marker).toMatchObject({ tag: "b12686", backend: "cuda", binary: "build/bin/llama-server.exe" });
        expect(marker.assets).toEqual(fixtures.assets.map((asset) => asset.name));
        expect(typeof marker.installedAt).toBe("string");
        expect(await fileExists(join(buildDir, "build", "bin", "llama-server.exe"))).toBe(true);
        expect(await fileExists(join(buildDir, "cudart64_12.dll"))).toBe(true);
        // The marker is written after both extractions, never before.
        expect(markerSeenDuringExtract).toEqual([false, false]);
        expect(extractCalls).toHaveLength(2);
        // Archives are deleted once unpacked.
        expect(await fileExists(join(workDir, "downloads", fixtures.assets[0].name))).toBe(false);
        expect(await fileExists(calls[0].dest)).toBe(false);

        const status = await installer.status(serverConfig);

        expect(status.binary).toMatchObject({
            installed: true,
            tag: "b12686",
            backend: "cuda",
            path: join(buildDir, "build", "bin", "llama-server.exe"),
        });
        expect(status.binary.task.state).toBe("done");
    });

    test("device cpu installs the cpu build even when a gpu build is on disk", async () => {
        const fixtures = cpuVulkanFixtures("b12686");

        await installFakeBuild("b12686", "vulkan");

        const vulkanMarkerPath = join(workDir, "bin", "b12686", "vulkan", MARKER);
        const vulkanMarker = await readFile(vulkanMarkerPath, "utf8");
        const harnessed = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ device: "cpu", releaseTag: "b12686" });

        harnessed.installer.startBinaryInstall(serverConfig);

        const done = await waitForTask(harnessed.installer, serverConfig, "binary");

        expect(done.state).toBe("done");
        // The cpu asset, and only it: the Vulkan folder already satisfies nothing for device cpu.
        expect(harnessed.calls.map((call) => call.url)).toEqual([assetUrl("b12686", fixtures.assets[0].name)]);
        expect(await fileExists(join(workDir, "bin", "b12686", "cpu", MARKER))).toBe(true);
        expect(await readFile(vulkanMarkerPath, "utf8")).toBe(vulkanMarker);

        // Now cpu is complete, so a second press is the no-op.
        harnessed.calls.length = 0;
        harnessed.installer.startBinaryInstall(serverConfig);

        const again = await waitForTask(harnessed.installer, serverConfig, "binary");

        expect(again.state).toBe("done");
        expect(again.message).toContain("already installed");
        expect(harnessed.calls).toEqual([]);
    });

    test("device gpu installs a gpu build even when only the cpu build is on disk", async () => {
        const fixtures = cpuVulkanFixtures("b12686");

        await installFakeBuild("b12686", "cpu");

        const harnessed = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ device: "gpu", releaseTag: "b12686" });

        harnessed.installer.startBinaryInstall(serverConfig);

        const done = await waitForTask(harnessed.installer, serverConfig, "binary");

        expect(done.state).toBe("done");
        // This release has no CUDA asset, so Vulkan is the GPU plan; the cpu build is not a no-op.
        expect(harnessed.calls.map((call) => call.url)).toEqual([assetUrl("b12686", fixtures.assets[1].name)]);
        expect(await fileExists(join(workDir, "bin", "b12686", "vulkan", MARKER))).toBe(true);
    });

    test("a second install of a complete folder downloads nothing", async () => {
        const fixtures = cudaInstallFixtures("b12686");
        const harnessed = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ releaseTag: "b12686" });

        harnessed.installer.startBinaryInstall(serverConfig);
        await waitForTask(harnessed.installer, serverConfig, "binary");
        expect(harnessed.calls).toHaveLength(2);

        harnessed.calls.length = 0;
        harnessed.installer.startBinaryInstall(serverConfig);

        const again = await waitForTask(harnessed.installer, serverConfig, "binary");

        expect(again.state).toBe("done");
        expect(again.message).toContain("already installed");
        expect(harnessed.calls).toEqual([]);
        expect(harnessed.extractCalls).toHaveLength(2);
    });

    test("cancel keeps the .part file and the next start resumes from it", async () => {
        const fixtures = cudaInstallFixtures("b12686");
        const harnessed = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ releaseTag: "b12686" });
        const partPath = `${join(workDir, "downloads", fixtures.assets[0].name)}.part`;
        const half = Math.max(1, Math.ceil(fixtures.assets[0].bytes.length / 2));

        harnessed.control.hook = () => harnessed.installer.cancel("binary");
        harnessed.installer.startBinaryInstall(serverConfig);

        const cancelled = await waitForTask(harnessed.installer, serverConfig, "binary");

        expect(cancelled.state).toBe("cancelled");
        expect(cancelled.message).toContain("partial download is kept");
        expect((await stat(partPath)).size).toBe(half);

        harnessed.installer.startBinaryInstall(serverConfig);

        const resumed = await waitForTask(harnessed.installer, serverConfig, "binary");

        expect(resumed.state).toBe("done");
        // The second run saw the kept bytes and asked for the rest.
        expect(harnessed.calls[1].resumedFrom).toBe(half);
        expect(await fileExists(partPath)).toBe(false);
    });

    test("a running install is not started twice", async () => {
        const fixtures = cudaInstallFixtures("b12686");
        const harnessed = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ releaseTag: "b12686" });

        harnessed.control.hold = Promise.resolve();
        harnessed.installer.startBinaryInstall(serverConfig);

        const second = harnessed.installer.startBinaryInstall(serverConfig);

        expect(second.state).toBe("running");

        await waitForTask(harnessed.installer, serverConfig, "binary");
        expect(harnessed.calls.length).toBeLessThanOrEqual(2);
    });

    test("a manual binary path blocks the download", async () => {
        const { installer } = harness();
        const started = installer.startBinaryInstall(config({ binaryPath: join(workDir, "llama-server.exe") }));

        expect(started.state).toBe("error");
        expect(started.message).toBe("A manual binary path is set; clear it to download one");
        expect(started.error).toBe(started.message);
    });

    test("an empty plan explains what to do", async () => {
        const { installer } = harness({
            platform: LINUX_ARM64,
            nvidiaSmi: CUDA_13,
            fetchJson: async () => releasePayload("b12686", [{ name: "llama-b12686-bin-ubuntu-arm64.tar.gz", bytes: zip({ "llama-server": "x" }) }]),
        });
        const serverConfig = config({ device: "gpu" });

        installer.startBinaryInstall(serverConfig);

        const failed = await waitForTask(installer, serverConfig, "binary");

        expect(failed.state).toBe("error");
        expect(failed.error).toContain("No GPU build of llama.cpp fits this system");
    });

    test("a release without binaries is reported instead of crashing", async () => {
        const { installer } = harness({ fetchJson: async () => ({ message: "Not Found" }) });
        const serverConfig = config({ releaseTag: "b999" });

        installer.startBinaryInstall(serverConfig);

        const failed = await waitForTask(installer, serverConfig, "binary");

        expect(failed.state).toBe("error");
        expect(failed.error).toContain('llama.cpp release "b999" was not found on GitHub');
    });

    test("a download error names the failing asset", async () => {
        const fixtures = cudaInstallFixtures("b12686");
        const { installer } = harness({
            // Only the first asset is served; the cudart one 404s.
            bodies: new Map([[assetUrl("b12686", fixtures.assets[0].name), fixtures.assets[0].bytes]]),
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ releaseTag: "b12686" });

        installer.startBinaryInstall(serverConfig);

        const failed = await waitForTask(installer, serverConfig, "binary");

        expect(failed.state).toBe("error");
        expect(failed.error).toContain(fixtures.assets[1].name);
        expect(failed.error).toContain("HTTP 404");
    });

    test("`latest` reads the release list, a pinned tag reads the tag endpoint", async () => {
        const cpuZip1 = zip({ "llama-server.exe": "b1" });
        const cpuZip7 = zip({ "llama-server.exe": "b7" });
        const older = releasePayload("b1", [{ name: "llama-b1-bin-win-cpu-x64.zip", bytes: cpuZip1 }]);
        const newer = releasePayload("b7", [{ name: "llama-b7-bin-win-cpu-x64.zip", bytes: cpuZip7 }]);
        const bodies = new Map([
            [assetUrl("b1", "llama-b1-bin-win-cpu-x64.zip"), cpuZip1],
            [assetUrl("b7", "llama-b7-bin-win-cpu-x64.zip"), cpuZip7],
        ]);

        const latest = harness({ bodies, fetchJson: async () => [older, newer] });
        const latestConfig = config({ device: "cpu", releaseTag: "latest" });

        latest.installer.startBinaryInstall(latestConfig);
        await waitForTask(latest.installer, latestConfig, "binary");

        expect(latest.fetchCalls).toEqual(["https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"]);
        // Newest b#### with assets wins, and the older build is not touched.
        expect(await fileExists(join(workDir, "bin", "b7", "cpu", MARKER))).toBe(true);
        expect(await fileExists(join(workDir, "bin", "b1", "cpu", MARKER))).toBe(false);

        await rm(join(workDir, "bin"), { recursive: true, force: true });

        const pinned = harness({ bodies, fetchJson: async () => newer });
        const pinnedConfig = config({ device: "cpu", releaseTag: "b7" });

        pinned.installer.startBinaryInstall(pinnedConfig);
        await waitForTask(pinned.installer, pinnedConfig, "binary");

        expect(pinned.fetchCalls).toEqual(["https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b7"]);
        expect(await fileExists(join(workDir, "bin", "b7", "cpu", MARKER))).toBe(true);
    });
});

describe("model download", () => {
    test("uses the catalog repo, file, size and sha256 for the default profile", async () => {
        const spec = LOCAL_MODELS.default;
        const body = new Uint8Array(4096);
        const url = `https://huggingface.co/${spec.repo}/resolve/main/${spec.file}`;
        const { installer, calls, fetchCalls } = harness({ bodies: new Map([[url, body]]) });
        const serverConfig = config({ profile: "default" });

        installer.startModelDownload(serverConfig);

        const done = await waitForTask(installer, serverConfig, "model");

        expect(done.state).toBe("done");
        // The static catalog needs no API call at all.
        expect(fetchCalls).toEqual([]);
        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({
            url,
            dest: join(workDir, "models", spec.file),
            expectedBytes: spec.bytes,
            sha256: spec.sha256,
        });
    });

    test("looks the custom file up in the HF tree and keeps lfs size and oid", async () => {
        const custom = { repo: "owner/custom-gguf", file: "my-model.gguf", pooling: "last" } as const;
        const body = new Uint8Array(4321);
        const url = `https://huggingface.co/${custom.repo}/resolve/main/${custom.file}`;
        const tree = [
            { type: "directory", path: "sub" },
            { type: "file", path: custom.file, size: 129, lfs: { oid: "a".repeat(64), size: body.length } },
        ];
        const { installer, calls, fetchCalls } = harness({
            bodies: new Map([[url, body]]),
            fetchJson: async () => tree,
            nvidiaSmi: null,
        });
        const serverConfig = config({ profile: "custom", customModel: custom });

        installer.startModelDownload(serverConfig);

        const done = await waitForTask(installer, serverConfig, "model");

        expect(done.state).toBe("done");
        expect(fetchCalls).toEqual([`https://huggingface.co/api/models/${custom.repo}/tree/main?recursive=true`]);
        expect(calls[0]).toMatchObject({
            url,
            dest: join(workDir, "models", "owner__custom-gguf__my-model.gguf"),
            // lfs.size, not the pointer size the tree reports at the top level.
            expectedBytes: body.length,
            sha256: "a".repeat(64),
        });

        const status = await installer.status(serverConfig);

        expect(status.model).toMatchObject({
            installed: true,
            file: "owner__custom-gguf__my-model.gguf",
            expectedBytes: body.length,
        });
    });

    test("reports a custom file the repo does not contain", async () => {
        const custom = { repo: "owner/custom-gguf", file: "missing.gguf", pooling: "mean" } as const;
        const { installer } = harness({ fetchJson: async () => [{ type: "file", path: "other.gguf", size: 10 }] });
        const serverConfig = config({ profile: "custom", customModel: custom });

        installer.startModelDownload(serverConfig);

        const failed = await waitForTask(installer, serverConfig, "model");

        expect(failed.state).toBe("error");
        expect(failed.error).toBe("File missing.gguf not found in owner/custom-gguf");
    });

    test("a manual model path blocks the download", async () => {
        const { installer } = harness();
        const started = installer.startModelDownload(config({ modelPath: join(workDir, "m.gguf") }));

        expect(started.state).toBe("error");
        expect(started.message).toBe("A manual model path is set; clear it to download one");
    });

    test("an already downloaded model is not fetched again", async () => {
        // Custom keeps the test to a few bytes: the catalog files are 176 MB / 2.4 GB.
        const custom = { repo: "owner/custom-gguf", file: "my-model.gguf", pooling: "last" } as const;
        const path = await writeFileOfSize(join(workDir, "models", "owner__custom-gguf__my-model.gguf"), 64);
        const tree = [{ type: "file", path: custom.file, size: 64, lfs: { oid: "b".repeat(64), size: 64 } }];
        const { installer, calls } = harness({ fetchJson: async () => tree });
        const serverConfig = config({ profile: "custom", customModel: custom });

        installer.startModelDownload(serverConfig);

        const done = await waitForTask(installer, serverConfig, "model");

        expect(done.state).toBe("done");
        expect(done.message).toContain("already downloaded");
        expect(calls).toEqual([]);

        const status = await installer.status(serverConfig);

        expect(status.model).toMatchObject({ installed: true, path });
    });
});

describe("resolveLaunch", () => {
    test("orders installed builds by device preference", async () => {
        await installFakeBuild("b12686", "cpu");
        await installFakeBuild("b12686", "vulkan");
        await installFakeBuild("b12686", "cuda");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        const auto = await installer.resolveLaunch(config({ modelPath }));

        expect(auto.ok).toBe(true);

        if (!auto.ok) {
            return;
        }

        expect(auto.candidates.map((candidate) => candidate.backend)).toEqual(["cuda", "vulkan", "cpu"]);
        expect(auto.candidates.map((candidate) => candidate.gpuLayers)).toEqual([99, 99, 0]);
        expect(auto.candidates[0].model).toBe(modelPath);
        expect(auto.candidates[0].pooling).toBe("mean");
        expect(auto.candidates[0].ctx).toBe(DEFAULT_SERVER_CONFIG.ctx);
    });

    test("device cpu only offers the cpu build with no offload", async () => {
        await installFakeBuild("b12686", "cpu");
        await installFakeBuild("b12686", "cuda");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        const resolved = await installer.resolveLaunch(config({ modelPath, device: "cpu" }));

        expect(resolved.ok).toBe(true);

        if (!resolved.ok) {
            return;
        }

        expect(resolved.candidates.map((candidate) => candidate.backend)).toEqual(["cpu"]);
        expect(resolved.candidates[0].gpuLayers).toBe(0);
    });

    test("on macOS the one metal build also serves device cpu with no offload", async () => {
        await installFakeBuild("b12686", "metal");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ platform: { os: "darwin", arch: "arm64" } });

        const cpu = await installer.resolveLaunch(config({ modelPath, device: "cpu" }));
        const auto = await installer.resolveLaunch(config({ modelPath }));

        expect(cpu.ok && auto.ok).toBe(true);

        if (!cpu.ok || !auto.ok) {
            return;
        }

        expect(cpu.candidates).toHaveLength(1);
        expect(cpu.candidates[0]).toMatchObject({ backend: "metal", gpuLayers: 0 });
        expect(auto.candidates[0].gpuLayers).toBe(99);
    });

    test("an NVIDIA GPU with no reported CUDA version still prefers cuda", async () => {
        // Newer drivers print `CUDA UMD Version`; releases.ts reports nvidia without a major and
        // planBinaries then offers the newest CUDA build, so the ordering has to agree.
        const bannerWithoutVersion = "| NVIDIA-SMI 590.00      Driver Version: 590.00      |";

        await installFakeBuild("b12686", "vulkan");
        const cudaPath = await installFakeBuild("b12686", "cuda");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: bannerWithoutVersion });

        const gpu = await installer.resolveLaunch(config({ modelPath, device: "gpu" }));

        expect(gpu.ok).toBe(true);

        if (!gpu.ok) {
            return;
        }

        expect(gpu.candidates.map((candidate) => candidate.backend)).toEqual(["cuda", "vulkan"]);
        expect(gpu.candidates[0].binary).toBe(cudaPath);
        expect((await installer.status(config())).binary.backend).toBe("cuda");
    });

    test("device gpu never falls back to the cpu build", async () => {
        await installFakeBuild("b12686", "cpu");
        await installFakeBuild("b12686", "vulkan");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        const gpu = await installer.resolveLaunch(config({ modelPath, device: "gpu" }));

        expect(gpu.ok).toBe(true);

        if (!gpu.ok) {
            return;
        }

        expect(gpu.candidates.map((candidate) => candidate.backend)).toEqual(["vulkan"]);

        const cpuOnly = harness({ nvidiaSmi: CUDA_13 });

        await rm(join(workDir, "bin", "b12686", "vulkan"), { recursive: true, force: true });

        const missing = await cpuOnly.installer.resolveLaunch(config({ modelPath, device: "gpu" }));

        expect(missing.ok).toBe(false);

        if (missing.ok) {
            return;
        }

        expect(missing.error).toBe("No GPU build of llama.cpp is installed. Open Semantic Memory settings and press Install and test.");
    });

    test("without an NVIDIA GPU the cuda build is skipped", async () => {
        await installFakeBuild("b12686", "cuda");
        await installFakeBuild("b12686", "cpu");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: null });

        const resolved = await installer.resolveLaunch(config({ modelPath }));

        expect(resolved.ok).toBe(true);

        if (!resolved.ok) {
            return;
        }

        expect(resolved.candidates.map((candidate) => candidate.backend)).toEqual(["cpu"]);
    });

    test("`latest` resolves the newest installed tag", async () => {
        await installFakeBuild("b12686", "cpu");
        const newest = await installFakeBuild("b13000", "cpu");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: null });

        const resolved = await installer.resolveLaunch(config({ modelPath, device: "cpu", releaseTag: "latest" }));

        expect(resolved.ok).toBe(true);

        if (!resolved.ok) {
            return;
        }

        expect(resolved.candidates.map((candidate) => candidate.binary)).toEqual([newest]);
    });

    test("a pinned tag ignores builds of other tags", async () => {
        await installFakeBuild("b13000", "cpu");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: null });

        const resolved = await installer.resolveLaunch(config({ modelPath, device: "cpu", releaseTag: "b12686" }));

        expect(resolved.ok).toBe(false);
    });

    test("a manual binary path is used as given and still honours -ngl", async () => {
        const binaryPath = await writeFileOfSize(join(workDir, "custom", "llama-server"), 8);
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: CUDA_13 });

        const gpu = await installer.resolveLaunch(config({ binaryPath, modelPath }));
        const cpu = await installer.resolveLaunch(config({ binaryPath, modelPath, device: "cpu" }));

        expect(gpu.ok && cpu.ok).toBe(true);

        if (!gpu.ok || !cpu.ok) {
            return;
        }

        expect(gpu.candidates).toHaveLength(1);
        expect(gpu.candidates[0]).toMatchObject({ binary: binaryPath, backend: "manual", gpuLayers: 99 });
        expect(cpu.candidates[0].gpuLayers).toBe(0);

        const missing = await installer.resolveLaunch(config({ binaryPath: join(workDir, "nope", "llama-server.exe"), modelPath }));

        expect(missing.ok).toBe(false);

        if (missing.ok) {
            return;
        }

        expect(missing.error).toContain("Manual llama-server path not found");
        expect(missing.error).toContain(join(workDir, "nope", "llama-server.exe"));
    });

    test("an incomplete model file is rejected with the retry hint", async () => {
        await installFakeBuild("b12686", "cpu");
        const spec = LOCAL_MODELS.default;
        const partial = await writeFileOfSize(join(workDir, "models", spec.file), 1024);
        const { installer } = harness({ nvidiaSmi: null });

        const resolved = await installer.resolveLaunch(config({ device: "cpu" }));

        expect(resolved.ok).toBe(false);

        if (resolved.ok) {
            return;
        }

        expect(resolved.error).toBe("Model file is incomplete, download it again");
        expect(partial.endsWith(spec.file)).toBe(true);
    });

    test("a missing model file names the path", async () => {
        await installFakeBuild("b12686", "cpu");
        const { installer } = harness({ nvidiaSmi: null });

        const resolved = await installer.resolveLaunch(config({ device: "cpu" }));

        expect(resolved.ok).toBe(false);

        if (resolved.ok) {
            return;
        }

        expect(resolved.error).toContain("Model file not found");
        expect(resolved.error).toContain(join(workDir, "models", LOCAL_MODELS.default.file));
    });

    test("a custom profile uses its pooling and its repo-prefixed file", async () => {
        await installFakeBuild("b12686", "cpu");
        const custom = { repo: "owner/custom-gguf", file: "my-model.gguf", pooling: "last" } as const;
        const modelPath = await writeFileOfSize(join(workDir, "models", "owner__custom-gguf__my-model.gguf"), 32);
        const { installer } = harness({ nvidiaSmi: null });

        const resolved = await installer.resolveLaunch(
            config({ device: "cpu", profile: "custom", customModel: custom }),
        );

        expect(resolved.ok).toBe(true);

        if (!resolved.ok) {
            return;
        }

        expect(resolved.candidates[0]).toMatchObject({ model: modelPath, pooling: "last" });
    });

    test("a manual model path keeps the pooling of the selected profile", async () => {
        await installFakeBuild("b12686", "cpu");
        const modelPath = await writeFileOfSize(join(workDir, "manual.gguf"), 64);
        const { installer } = harness({ nvidiaSmi: null });

        const resolved = await installer.resolveLaunch(config({ device: "cpu", profile: "strong", modelPath }));

        expect(resolved.ok).toBe(true);

        if (!resolved.ok) {
            return;
        }

        expect(resolved.candidates[0].pooling).toBe("last");
    });
});

describe("removal", () => {
    test("removeBinary deletes bin and downloads once nothing is running", async () => {
        const fixtures = cudaInstallFixtures("b12686");
        const harnessed = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ releaseTag: "b12686" });

        harnessed.installer.startBinaryInstall(serverConfig);
        await waitForTask(harnessed.installer, serverConfig, "binary");

        expect((await harnessed.installer.status(serverConfig)).binary.bytesOnDisk).toBeGreaterThan(0);

        await harnessed.installer.removeBinary();

        const status = await harnessed.installer.status(serverConfig);

        expect(status.binary).toMatchObject({ installed: false, bytesOnDisk: 0 });
        expect(status.binary.task.state).toBe("idle");
        expect(await fileExists(join(workDir, "bin"))).toBe(false);
        expect(await fileExists(join(workDir, "downloads"))).toBe(false);
    });

    test("removeBinary and removeModels refuse while a task is running", async () => {
        const fixtures = cudaInstallFixtures("b12686");
        const harnessed = harness({
            bodies: fixtures.bodies,
            fetchJson: async () => releasePayload("b12686", fixtures.assets),
            nvidiaSmi: CUDA_13,
        });
        const serverConfig = config({ releaseTag: "b12686" });
        let release = (): void => undefined;

        harnessed.control.hold = new Promise<void>((resolve) => {
            release = resolve;
        });

        harnessed.installer.startBinaryInstall(serverConfig);

        await expect(harnessed.installer.removeBinary()).rejects.toThrow(/cancel it before deleting/);

        // dispose() aborts, and the held download then reports the abort: the task ends cancelled.
        harnessed.installer.dispose();
        release();

        const final = await waitForTask(harnessed.installer, serverConfig, "binary");

        expect(final.state).toBe("cancelled");

        await expect(harnessed.installer.removeModels()).resolves.toBeUndefined();
        expect(await harnessed.installer.removeBinary()).toBeUndefined();
        expect(await fileExists(join(workDir, "bin"))).toBe(false);
    });

    test("removeModels refuses while a model task is running", async () => {
        const spec = LOCAL_MODELS.default;
        const url = `https://huggingface.co/${spec.repo}/resolve/main/${spec.file}`;
        const harnessed = harness({ bodies: new Map([[url, new Uint8Array(64)]]) });
        const serverConfig = config({ profile: "default" });
        let release = (): void => undefined;

        harnessed.control.hold = new Promise<void>((resolve) => {
            release = resolve;
        });

        harnessed.installer.startModelDownload(serverConfig);

        await expect(harnessed.installer.removeModels()).rejects.toThrow(/cancel it before deleting/);

        harnessed.control.hold = null;
        release();
        await waitForTask(harnessed.installer, serverConfig, "model");

        await harnessed.installer.removeModels();

        expect(await fileExists(join(workDir, "models"))).toBe(false);
    });

    test("a disposed installer refuses new tasks", async () => {
        const { installer } = harness();

        installer.dispose();

        expect(installer.startBinaryInstall(config()).state).toBe("error");
        expect(installer.startModelDownload(config()).state).toBe("error");
    });
});

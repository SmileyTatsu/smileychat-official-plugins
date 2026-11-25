// llama.cpp + GGUF model installer: what the setup panel calls, and what llama-process reads to
// start a server.
//
// On-disk layout (everything under host.dataDir):
//   downloads/<asset>                                  archives while they download (.part resumes)
//   bin/<tag>/<backend>/.installed.json                written last; its presence = a usable build
//   bin/<tag>/<backend>/...                            the unpacked release (llama-server + DLLs)
//   models/<file>.gguf                                 the embedding model
//
// Nothing here is allowed to touch the network or the disk outside those folders, and nothing
// from config.json is used as an argument here: this file only produces paths and numbers that
// config.ts already validated.

import type { Dirent } from "node:fs";
import { mkdir, readdir, readFile, rm, stat, unlink, writeFile } from "node:fs/promises";
import { basename, join, relative } from "node:path";

import { LOCAL_MODELS } from "../models";
import { extractArchive } from "./archive";
import { DownloadError, downloadFile, isAllowedDownloadUrl } from "./download";
import { parseNvidiaSmi, parseReleases, pickRelease, planBinaries, type Backend, type GpuInfo, type Platform, type Release } from "./releases";
import type {
    DeviceChoice,
    HostLogger,
    InstallerStatus,
    LaunchSpec,
    ServerConfig,
    TaskStatus,
} from "./types";

export type InstallerDeps = {
    /** JSON from an allow-listed host; throws with a clear message on any non-2xx answer. */
    fetchJson: (url: string, signal?: AbortSignal) => Promise<unknown>;
    download: typeof downloadFile;
    extract: (archive: string, dest: string) => Promise<string[]>;
    /** Raw `nvidia-smi` output, or null when it is not installed / fails / times out. */
    runNvidiaSmi: () => Promise<string | null>;
    platform: Platform;
};

const RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30";
const RELEASE_TAGS_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags";
const HUGGINGFACE_URL = "https://huggingface.co";

const USER_AGENT = "smiley-embeddings-memory (SmileyChat plugin)";
const MARKER_FILE = ".installed.json";
const NVIDIA_SMI_TIMEOUT_MS = 5000;
const MAX_WALK_DEPTH = 8;

const MANUAL_BINARY_MESSAGE = "A manual binary path is set; clear it to download one";
const MANUAL_MODEL_MESSAGE = "A manual model path is set; clear it to download one";

/**
 * The one not-installed text, shared by /start, /embed (through resolveLaunch) and the setup panel.
 *
 * It states the FACT only: the client already appends its own hint for `not-installed` ("Open
 * Semantic Memory settings and press Install and test."), so repeating the instruction here
 * printed the same sentence twice in the error box. device `auto` also gets user words instead of
 * "installed for auto", which described our setting rather than the user's problem.
 */
function noBinaryMessage(device: DeviceChoice): string {
    if (device === "gpu") {
        return "No GPU build of llama.cpp is installed.";
    }

    if (device === "cpu") {
        return "No CPU build of llama.cpp is installed.";
    }

    return "The local embedding program is not installed yet.";
}

type InstallMarker = {
    tag: string;
    backend: Backend;
    assets: string[];
    installedAt: string;
    /** Path of llama-server inside the build folder, relative and "/"-separated. */
    binary: string;
};

type InstalledBuild = {
    tag: string;
    backend: Backend;
    binaryPath: string;
};

type ExpectedModel = { file: string; path: string; expectedBytes?: number };

type ModelTarget = ExpectedModel & { url: string; sha256?: string };

type TaskSlot = {
    status: TaskStatus;
    controller: AbortController | null;
};

const BACKEND_VALUES: readonly Backend[] = ["cuda", "vulkan", "cpu", "metal"];

function idleTask(): TaskSlot {
    return { status: { state: "idle", received: 0, total: 0 }, controller: null };
}

function copyTask(status: TaskStatus): TaskStatus {
    return { ...status };
}

function setProgress(status: TaskStatus, received: number, total: number): void {
    status.received = Math.max(0, Math.round(received));
    status.total = Math.max(0, Math.round(total));
}

function finishTask(status: TaskStatus, message: string): void {
    status.state = "done";
    status.message = message;
    delete status.error;

    if (status.total > 0) {
        status.received = status.total;
    }
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function isAbort(error: unknown, signal?: AbortSignal): boolean {
    if (error instanceof DownloadError && error.code === "aborted") {
        return true;
    }

    if (signal?.aborted) {
        return true;
    }

    return error instanceof Error && (error.name === "AbortError" || error.name === "TimeoutError");
}

/** Actionable text for a failed asset download; the asset name is the part the user can act on. */
function describeDownloadFailure(error: unknown, assetName: string): string {
    if (error instanceof DownloadError) {
        switch (error.code) {
            case "host-not-allowed":
                return `The download URL for ${assetName} is not on the allow list (api.github.com, github.com, *.githubusercontent.com)`;
            case "size-mismatch":
                return `${assetName} downloaded to the wrong size (${error.message}); press Install again to retry`;
            case "hash-mismatch":
                return `${assetName} failed its checksum (${error.message}); press Install again to retry`;
            case "http":
                return `Downloading ${assetName} failed: ${error.message}. GitHub may be rate limiting; wait a few minutes and retry`;
            default:
                return `Downloading ${assetName} failed: ${error.message}. Check the connection and retry`;
        }
    }

    return `Downloading ${assetName} failed: ${errorMessage(error)}`;
}

function sha256OfDigest(digest: string | undefined): string | undefined {
    if (!digest) {
        return undefined;
    }

    // GitHub sends `sha256:<hex>`; downloadFile wants the bare hex.
    const bare = digest.replace(/^sha256:/i, "").trim();

    return /^[a-f0-9]{64}$/i.test(bare) ? bare : undefined;
}

async function isFile(path: string): Promise<boolean> {
    try {
        return (await stat(path)).isFile();
    } catch {
        return false;
    }
}

async function fileSizeOrNull(path: string): Promise<number | null> {
    try {
        const info = await stat(path);

        return info.isFile() ? info.size : null;
    } catch {
        return null;
    }
}

/** Recursive size of a folder. Missing folders are 0; unreadable entries are skipped. */
async function directoryBytes(root: string): Promise<number> {
    const entries = await readdir(root, { withFileTypes: true }).catch((): Dirent[] => []);
    let total = 0;

    for (const entry of entries) {
        const full = join(root, entry.name);

        if (entry.isDirectory()) {
            total += await directoryBytes(full);
        } else if (entry.isFile()) {
            total += (await fileSizeOrNull(full)) ?? 0;
        }
    }

    return total;
}

async function directoryNames(root: string): Promise<string[]> {
    const entries = await readdir(root, { withFileTypes: true }).catch((): Dirent[] => []);

    return entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
}

async function walkFiles(root: string, depth = 0): Promise<string[]> {
    if (depth > MAX_WALK_DEPTH) {
        return [];
    }

    const entries = await readdir(root, { withFileTypes: true }).catch((): Dirent[] => []);
    const files: string[] = [];

    for (const entry of entries) {
        const full = join(root, entry.name);

        if (entry.isDirectory()) {
            files.push(...(await walkFiles(full, depth + 1)));
        } else if (entry.isFile()) {
            files.push(full);
        }
    }

    return files;
}

/**
 * llama-server inside an unpacked release. The archives nest it one or two levels down
 * (`build/bin/llama-server` in some builds), so it is searched for instead of assumed.
 */
async function findServerBinary(root: string, platform: Platform): Promise<string | null> {
    const files = await walkFiles(root);
    const wanted = platform.os === "win32" ? "llama-server.exe" : "llama-server";
    const other = platform.os === "win32" ? "llama-server" : "llama-server.exe";

    const exact = files.find((file) => basename(file).toLowerCase() === wanted);
    const fallback = files.find((file) => basename(file).toLowerCase() === other);
    const found = exact ?? fallback;

    return found ? relative(root, found).split("\\").join("/") : null;
}

function isSafeRelative(name: string): boolean {
    if (!name || name.startsWith("/") || /^[a-zA-Z]:/.test(name)) {
        return false;
    }

    return !name.split(/[\\/]/).includes("..");
}

/** A marker only counts when it is complete and the binary it names is still on disk. */
async function readMarker(dir: string): Promise<{ backend: Backend; binary: string } | null> {
    let raw: unknown;

    try {
        raw = JSON.parse(await readFile(join(dir, MARKER_FILE), "utf8"));
    } catch {
        return null;
    }

    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        return null;
    }

    const record = raw as Record<string, unknown>;
    const backend = typeof record.backend === "string" ? record.backend : "";
    const binary = typeof record.binary === "string" ? record.binary : "";

    if (!(BACKEND_VALUES as readonly string[]).includes(backend) || !isSafeRelative(binary)) {
        return null;
    }

    if (!(await isFile(join(dir, binary)))) {
        return null;
    }

    return { backend: backend as Backend, binary };
}

function gpuLayersFor(config: ServerConfig, backend: Backend | "manual"): number {
    // `cpu` device means "do not offload", whatever binary is around; on macOS the one Metal
    // binary runs on the CPU with the same flag.
    return config.device === "cpu" || backend === "cpu" ? 0 : 99;
}

/**
 * Backend preference for this machine, in the order planBinaries would try them. Duplicated
 * here in the small form resolveLaunch needs: planBinaries orders *release assets*, and this
 * orders *installed folders*, which have no assets to match against.
 *
 * `gpu === null` means "not probed" (status() must not spawn nvidia-smi). A CUDA build is then
 * counted as usable, because only a real start can prove the driver cannot run it, and a wrong
 * "not installed" would send the user back to the installer for 600 MB they do not need.
 */
function backendOrder(platform: Platform, gpu: GpuInfo | null, config: ServerConfig): Backend[] {
    const wantsGpu = config.device !== "cpu";
    const wantsCpu = config.device !== "gpu";

    if (platform.os === "darwin") {
        if (platform.arch === "arm64") {
            // One universal binary for both Metal and `-ngl 0`.
            return ["metal"];
        }

        // Intel macs only get the plain CPU build; `gpu` has nothing to fall back to.
        return wantsGpu && !wantsCpu ? [] : ["cpu"];
    }

    const order: Backend[] = [];
    // Windows ships GPU builds for both arches; Linux only for x64 (arm64 is CPU-only).
    const gpuAssets = platform.os === "win32" || (platform.os === "linux" && platform.arch === "x64");
    // Same rule cudaMajorsFor applies in releases.ts: an NVIDIA machine whose driver does not
    // report a CUDA version ("CUDA UMD Version" on newer drivers) still gets the CUDA build
    // first, because planBinaries offers it and a start would otherwise use a build this
    // ordering never proposes.
    const cudaUsable =
        gpu === null || (gpu.nvidia && (gpu.cudaMajor === undefined || gpu.cudaMajor >= 12));

    if (wantsGpu && gpuAssets) {
        if (config.gpuBackend !== "vulkan" && cudaUsable) {
            order.push("cuda");
        }

        if (config.gpuBackend !== "cuda") {
            order.push("vulkan");
        }
    }

    if (wantsCpu) {
        order.push("cpu");
    }

    return order;
}

function newestTag(builds: InstalledBuild[]): string | undefined {
    let best: { tag: string; build: number } | undefined;

    for (const build of builds) {
        const match = /^b(\d+)$/.exec(build.tag);

        if (!match) {
            continue;
        }

        const number = Number(match[1]);

        if (!best || number > best.build) {
            best = { tag: build.tag, build: number };
        }
    }

    return best?.tag;
}

/** Installed builds of the tag the config asks for; `latest` means the newest installed tag. */
function buildsForTag(builds: InstalledBuild[], config: ServerConfig): InstalledBuild[] {
    const tag = config.releaseTag === "latest" ? newestTag(builds) : config.releaseTag;

    return builds.filter((build) => build.tag === tag);
}

/**
 * Installed builds for the configured tag, most preferred first, filtered by the device choice.
 * This is the single selection rule: resolveLaunch uses it with real GPU info, status() with
 * `null`, and both therefore agree on what "installed for this device" means.
 */
function preferBuilds(
    builds: InstalledBuild[],
    config: ServerConfig,
    platform: Platform,
    gpu: GpuInfo | null,
): InstalledBuild[] {
    if (builds.length === 0) {
        return [];
    }

    const sameTag = buildsForTag(builds, config);
    const ordered: InstalledBuild[] = [];

    for (const backend of backendOrder(platform, gpu, config)) {
        ordered.push(...sameTag.filter((build) => build.backend === backend));
    }

    return ordered;
}

/** Every backend unpacked on disk for the tag, in a fixed order so the UI can diff it. */
function installedBackendsFor(builds: InstalledBuild[], config: ServerConfig): Backend[] {
    const present = new Set(buildsForTag(builds, config).map((build) => build.backend));

    return BACKEND_VALUES.filter((backend) => present.has(backend));
}

function detectPlatform(): Platform {
    // llama.cpp only publishes x64 and arm64 builds; anything else (ia32) runs the x64 build or
    // not at all, so it is mapped to x64 instead of inventing a third value.
    const arch: Platform["arch"] = process.arch === "arm64" ? "arm64" : "x64";

    if (process.platform === "win32") {
        return { os: "win32", arch };
    }

    if (process.platform === "darwin") {
        return { os: "darwin", arch };
    }

    // Termux reports Linux but has no /usr/lib and no GPU builds; the manual path is the
    // supported route there, and the android platform value says so in the error messages.
    const termux = Boolean(process.env.TERMUX_VERSION) || (process.env.PREFIX ?? "").includes("com.termux");

    return { os: termux ? "android" : "linux", arch };
}

/** `nvidia-smi` with a hard timeout: it is the only way to learn the CUDA major version. */
async function defaultRunNvidiaSmi(): Promise<string | null> {
    let proc: Bun.Subprocess<"ignore", "pipe", "pipe">;

    try {
        proc = Bun.spawn(["nvidia-smi"], { stdout: "pipe", stderr: "pipe" });
    } catch {
        // Not installed: Bun.spawn throws ENOENT synchronously.
        return null;
    }

    const timer = setTimeout(() => {
        try {
            proc.kill();
        } catch {
            // Already gone.
        }
    }, NVIDIA_SMI_TIMEOUT_MS);

    try {
        const [text, code] = await Promise.all([new Response(proc.stdout).text(), proc.exited]);

        return code === 0 ? text : null;
    } catch {
        return null;
    } finally {
        clearTimeout(timer);
    }
}

/** GitHub/Hugging Face JSON, allow-list checked before the request so a typo cannot leak a call. */
async function defaultFetchJson(url: string, signal?: AbortSignal): Promise<unknown> {
    if (!isAllowedDownloadUrl(url)) {
        throw new Error(`refusing to call ${url}: the host is not on the download allow list`);
    }

    const response = await fetch(url, {
        headers: { "user-agent": USER_AGENT, accept: "application/vnd.github+json" },
        signal,
    });

    if (response.status === 403) {
        // GitHub answers 403 (not 429) once the anonymous 60 requests/hour limit is spent.
        const reset = response.headers.get("x-ratelimit-reset");
        const minutes = reset
            ? Math.max(1, Math.ceil((Number(reset) * 1000 - Date.now()) / 60_000))
            : undefined;
        const when = minutes ? ` Try again in about ${minutes} minute(s).` : " Wait a few minutes and retry.";

        await response.body?.cancel().catch(() => undefined);
        throw new Error(`GitHub refused the request (HTTP 403): its rate limit is exhausted.${when}`);
    }

    if (!response.ok) {
        await response.body?.cancel().catch(() => undefined);
        throw new Error(`HTTP ${response.status} for ${url}`);
    }

    try {
        return await response.json();
    } catch {
        throw new Error(`${url} did not return valid JSON`);
    }
}

export class Installer {
    private readonly dataDir: string;
    private readonly logger: HostLogger;
    private readonly deps: InstallerDeps;
    private readonly binary: TaskSlot = idleTask();
    private readonly model: TaskSlot = idleTask();
    /** Sizes learned from the Hugging Face tree API, so a resolved custom model can be size-checked. */
    private readonly knownModelSizes = new Map<string, number>();
    private disposed = false;

    constructor(host: { dataDir: string; logger: HostLogger }, deps: Partial<InstallerDeps> = {}) {
        this.dataDir = host.dataDir;
        this.logger = host.logger;
        this.deps = {
            fetchJson: deps.fetchJson ?? defaultFetchJson,
            download: deps.download ?? downloadFile,
            extract: deps.extract ?? ((archive, dest) => extractArchive(archive, dest)),
            runNvidiaSmi: deps.runNvidiaSmi ?? defaultRunNvidiaSmi,
            platform: deps.platform ?? detectPlatform(),
        };
    }

    private binDir(): string {
        return join(this.dataDir, "bin");
    }

    private downloadsDir(): string {
        return join(this.dataDir, "downloads");
    }

    private modelsDir(): string {
        return join(this.dataDir, "models");
    }

    /** Cheap snapshot for the setup panel: stat/readdir only, no network and no process probe. */
    async status(config: ServerConfig): Promise<InstallerStatus> {
        const builds = await this.listBuilds();
        const [binBytes, downloadBytes, modelBytes] = await Promise.all([
            directoryBytes(this.binDir()),
            directoryBytes(this.downloadsDir()),
            directoryBytes(this.modelsDir()),
        ]);

        const binary: InstallerStatus["binary"] = {
            installed: false,
            // The archives in downloads/ belong to the binary install; .part files are included.
            bytesOnDisk: binBytes + downloadBytes,
            installedBackends: installedBackendsFor(builds, config),
            task: copyTask(this.binary.status),
        };

        if (config.binaryPath) {
            binary.path = config.binaryPath;
            binary.backend = "manual";
            binary.installed = await isFile(config.binaryPath);
        } else {
            // The same selection resolveLaunch makes, with `null` GPU info: status must not spawn
            // nvidia-smi. What it must never do is call a build the device choice excludes
            // "installed" — device cpu with only the Vulkan build on disk is missing, not ready,
            // and saying otherwise makes the panel skip the download and Test fail later.
            const build = preferBuilds(builds, config, this.deps.platform, null)[0];

            if (build) {
                binary.installed = true;
                binary.tag = build.tag;
                binary.backend = build.backend;
                binary.path = build.binaryPath;
            }
        }

        const expected = this.expectedModel(config);
        const model: InstallerStatus["model"] = {
            installed: false,
            bytesOnDisk: modelBytes,
            task: copyTask(this.model.status),
        };

        if (expected) {
            const size = await fileSizeOrNull(expected.path);

            model.file = expected.file;
            model.path = expected.path;
            model.expectedBytes = expected.expectedBytes;
            model.installed =
                size !== null && (expected.expectedBytes === undefined || size === expected.expectedBytes);
        }

        return { binary, model };
    }

    startBinaryInstall(config: ServerConfig): TaskStatus {
        if (this.binary.status.state === "running" && this.binary.controller) {
            return copyTask(this.binary.status);
        }

        if (this.disposed) {
            return this.failStart(this.binary, "The installer was shut down");
        }

        if (config.binaryPath) {
            this.binary.status = { state: "error", message: MANUAL_BINARY_MESSAGE, error: MANUAL_BINARY_MESSAGE, received: 0, total: 0 };

            return copyTask(this.binary.status);
        }

        const controller = new AbortController();

        this.binary.controller = controller;
        this.binary.status = { state: "running", message: "Checking the llama.cpp release…", received: 0, total: 0 };

        void this.installBinary(config, controller.signal).catch((error: unknown) => {
            this.settleFailure(this.binary, error, controller.signal, "llama.cpp install");
        });

        return copyTask(this.binary.status);
    }

    startModelDownload(config: ServerConfig): TaskStatus {
        if (this.model.status.state === "running" && this.model.controller) {
            return copyTask(this.model.status);
        }

        if (this.disposed) {
            return this.failStart(this.model, "The installer was shut down");
        }

        if (config.modelPath) {
            this.model.status = { state: "error", message: MANUAL_MODEL_MESSAGE, error: MANUAL_MODEL_MESSAGE, received: 0, total: 0 };

            return copyTask(this.model.status);
        }

        const controller = new AbortController();

        this.model.controller = controller;
        this.model.status = { state: "running", message: "Preparing the model download…", received: 0, total: 0 };

        void this.installModel(config, controller.signal).catch((error: unknown) => {
            this.settleFailure(this.model, error, controller.signal, "model download");
        });

        return copyTask(this.model.status);
    }

    /** Abort a running task. The `.part` file is kept, so the next start resumes instead of restarting. */
    cancel(which: "binary" | "model"): void {
        const slot = which === "binary" ? this.binary : this.model;

        if (!slot.controller || slot.status.state !== "running") {
            return;
        }

        slot.status.message = "Cancelling…";
        slot.controller.abort();
    }

    async removeBinary(): Promise<void> {
        if (this.binary.status.state === "running" && this.binary.controller) {
            throw new Error("A llama.cpp download is running; cancel it before deleting the files");
        }

        this.binary.controller?.abort();
        await rm(this.binDir(), { recursive: true, force: true });
        await rm(this.downloadsDir(), { recursive: true, force: true });
        this.binary.status = idleTask().status;
        this.logger.info("llama.cpp files deleted");
    }

    async removeModels(): Promise<void> {
        if (this.model.status.state === "running" && this.model.controller) {
            throw new Error("A model download is running; cancel it before deleting the files");
        }

        this.model.controller?.abort();
        await rm(this.modelsDir(), { recursive: true, force: true });
        this.model.status = idleTask().status;
        this.logger.info("model files deleted");
    }

    async resolveLaunch(
        config: ServerConfig,
    ): Promise<{ ok: true; candidates: LaunchSpec[] } | { ok: false; error: string }> {
        const binaries = await this.resolveBinary(config);

        if (!binaries.ok) {
            return { ok: false, error: binaries.error };
        }

        const model = await this.resolveModel(config);

        if (!model.ok) {
            return { ok: false, error: model.error };
        }

        return {
            ok: true,
            candidates: binaries.candidates.map((candidate) => ({
                binary: candidate.binary,
                backend: candidate.backend,
                gpuLayers: candidate.gpuLayers,
                model: model.path,
                pooling: model.pooling,
                ctx: config.ctx,
            })),
        };
    }

    /** Stop both tasks; called from the plugin's dispose. */
    dispose(): void {
        this.disposed = true;
        this.binary.controller?.abort();
        this.model.controller?.abort();
    }

    private failStart(slot: TaskSlot, message: string): TaskStatus {
        slot.status = { state: "error", message, error: message, received: 0, total: 0 };

        return copyTask(slot.status);
    }

    private settleFailure(slot: TaskSlot, error: unknown, signal: AbortSignal, label: string): void {
        const { received, total } = slot.status;

        if (isAbort(error, signal)) {
            slot.status = {
                state: "cancelled",
                message: "Cancelled; the partial download is kept so the next attempt resumes",
                received,
                total,
            };
            this.logger.info(`${label} cancelled`);
        } else {
            const message = errorMessage(error);

            slot.status = { state: "error", message, error: message, received, total };
            this.logger.warn(`${label} failed`, message);
        }

        slot.controller = null;
    }

    private async listBuilds(): Promise<InstalledBuild[]> {
        const builds: InstalledBuild[] = [];

        for (const tag of await directoryNames(this.binDir())) {
            for (const backendName of await directoryNames(join(this.binDir(), tag))) {
                const dir = join(this.binDir(), tag, backendName);
                const marker = await readMarker(dir);

                if (!marker) {
                    continue;
                }

                builds.push({ tag, backend: marker.backend, binaryPath: join(dir, marker.binary) });
            }
        }

        return builds;
    }

    /**
     * The model file for `config`, without network access. A custom model's expected size is only
     * known once the Hugging Face tree has been read, so it is remembered from the last download
     * in this process (after a restart the size check is skipped, not failed).
     */
    private expectedModel(config: ServerConfig): ExpectedModel | undefined {
        if (config.modelPath) {
            return { file: basename(config.modelPath), path: config.modelPath };
        }

        if (config.profile === "custom") {
            const custom = config.customModel;

            if (!custom) {
                return undefined;
            }

            // The repo prefix keeps two custom models with the same file name apart.
            const file = `${custom.repo.replace(/\//g, "__")}__${custom.file}`;

            return {
                file,
                path: join(this.modelsDir(), file),
                expectedBytes: this.knownModelSizes.get(`${custom.repo}/${custom.file}`),
            };
        }

        const spec = LOCAL_MODELS[config.profile];

        return { file: spec.file, path: join(this.modelsDir(), spec.file), expectedBytes: spec.bytes };
    }

    private async resolveBinary(
        config: ServerConfig,
    ): Promise<
        { ok: true; candidates: { binary: string; backend: Backend | "manual"; gpuLayers: number }[] } | { ok: false; error: string }
    > {
        if (config.binaryPath) {
            if (!(await isFile(config.binaryPath))) {
                return { ok: false, error: `Manual llama-server path not found: ${config.binaryPath}` };
            }

            return {
                ok: true,
                candidates: [{ binary: config.binaryPath, backend: "manual", gpuLayers: gpuLayersFor(config, "manual") }],
            };
        }

        const gpu = await this.detectGpu();
        const builds = preferBuilds(await this.listBuilds(), config, this.deps.platform, gpu);

        if (builds.length === 0) {
            return { ok: false, error: noBinaryMessage(config.device) };
        }

        return {
            ok: true,
            candidates: builds.map((build) => ({
                binary: build.binaryPath,
                backend: build.backend,
                gpuLayers: gpuLayersFor(config, build.backend),
            })),
        };
    }

    private async resolveModel(
        config: ServerConfig,
    ): Promise<{ ok: true; path: string; pooling: "mean" | "last" } | { ok: false; error: string }> {
        if (config.modelPath) {
            if (!(await isFile(config.modelPath))) {
                return { ok: false, error: `Manual model path not found: ${config.modelPath}` };
            }

            return { ok: true, path: config.modelPath, pooling: this.poolingFor(config) };
        }

        const expected = this.expectedModel(config);

        if (!expected) {
            return { ok: false, error: "A custom model needs its Hugging Face repo and file; set them in the setup" };
        }

        const size = await fileSizeOrNull(expected.path);

        if (size === null) {
            return { ok: false, error: `Model file not found: ${expected.path}. Open the setup and download it.` };
        }

        if (expected.expectedBytes !== undefined && size !== expected.expectedBytes) {
            return { ok: false, error: "Model file is incomplete, download it again" };
        }

        return { ok: true, path: expected.path, pooling: this.poolingFor(config) };
    }

    private poolingFor(config: ServerConfig): "mean" | "last" {
        if (config.profile === "custom") {
            return config.customModel?.pooling ?? "mean";
        }

        return LOCAL_MODELS[config.profile].pooling;
    }

    private async detectGpu(): Promise<GpuInfo> {
        try {
            const output = await this.deps.runNvidiaSmi();

            return output === null ? { nvidia: false } : parseNvidiaSmi(output);
        } catch {
            return { nvidia: false };
        }
    }

    private async fetchRelease(tag: string, signal: AbortSignal): Promise<Release | undefined> {
        // `/releases/latest` answers with a `v1.0.0` tag that has no binaries, so `latest` reads
        // the list and pickRelease takes the highest b#### that has assets.
        const url = tag === "latest" ? RELEASES_URL : `${RELEASE_TAGS_URL}/${encodeURIComponent(tag)}`;
        const json = await this.deps.fetchJson(url, signal);
        // The tags endpoint answers with one release object, the list endpoint with an array.
        const releases = parseReleases(Array.isArray(json) ? json : [json]);

        return pickRelease(releases, tag);
    }

    private async installBinary(config: ServerConfig, signal: AbortSignal): Promise<void> {
        const slot = this.binary;

        slot.status.message = "Checking the llama.cpp release…";

        const release = await this.fetchRelease(config.releaseTag, signal);

        if (!release) {
            throw new Error(`llama.cpp release "${config.releaseTag}" was not found on GitHub`);
        }

        const gpu = await this.detectGpu();
        const [plan] = planBinaries(release, this.deps.platform, gpu, {
            device: config.device,
            gpuBackend: config.gpuBackend,
        });

        if (!plan) {
            throw new Error(
                "No GPU build of llama.cpp fits this system (platform, detected GPU); choose CPU or set a manual binary path",
            );
        }

        const dest = join(this.binDir(), release.tag, plan.backend);
        const markerPath = join(dest, MARKER_FILE);
        const total = plan.assets.reduce((sum, asset) => sum + Math.max(0, asset.size), 0);

        if (await readMarker(dest)) {
            // Everything the plan needs sits in this folder already; a reinstall must not
            // re-download 600 MB of CUDA.
            setProgress(slot.status, total, total);
            finishTask(slot.status, `llama.cpp ${release.tag} (${plan.backend}) is already installed.`);
            this.logger.info("llama.cpp already installed", { tag: release.tag, backend: plan.backend });

            return;
        }

        await mkdir(dest, { recursive: true });
        // A marker left over from an interrupted reinstall would make the next run skip files
        // that are no longer there.
        await unlink(markerPath).catch(() => undefined);

        const downloads = this.downloadsDir();
        let completedBytes = 0;

        for (const asset of plan.assets) {
            if (signal.aborted) {
                throw new DownloadError("aborted", "download aborted");
            }

            const destFile = join(downloads, asset.name);
            const base = completedBytes;

            slot.status.message = `Downloading ${asset.name}`;

            try {
                await this.deps.download({
                    url: asset.url,
                    dest: destFile,
                    expectedBytes: asset.size > 0 ? asset.size : undefined,
                    sha256: sha256OfDigest(asset.digest),
                    signal,
                    onProgress: (progress) => setProgress(slot.status, base + progress.received, total),
                });
            } catch (error) {
                if (isAbort(error, signal)) {
                    throw error;
                }

                throw new Error(describeDownloadFailure(error, asset.name));
            }

            completedBytes += Math.max(0, asset.size);
            setProgress(slot.status, completedBytes, total);
        }

        // All archives of a plan go into the same folder: the CUDA build needs the cudart DLLs
        // beside llama-server.exe to start.
        for (const asset of plan.assets) {
            if (signal.aborted) {
                throw new DownloadError("aborted", "download aborted");
            }

            const archive = join(downloads, asset.name);

            slot.status.message = `Unpacking ${asset.name}`;

            try {
                await this.deps.extract(archive, dest);
            } catch (error) {
                throw new Error(`Unpacking ${asset.name} failed: ${errorMessage(error)}. Try again or choose another backend`);
            }

            await unlink(archive).catch(() => undefined);
        }

        const binary = await findServerBinary(dest, this.deps.platform);

        if (!binary) {
            throw new Error(
                `llama-server was not inside the ${release.tag} ${plan.backend} build; try another backend or set a manual binary path`,
            );
        }

        const marker: InstallMarker = {
            tag: release.tag,
            backend: plan.backend,
            assets: plan.assets.map((asset) => asset.name),
            installedAt: new Date().toISOString(),
            binary,
        };

        // Written last: this file is what makes the folder count as installed.
        await writeFile(markerPath, `${JSON.stringify(marker, null, 4)}\n`, "utf8");

        setProgress(slot.status, total, total);
        finishTask(slot.status, `llama.cpp ${release.tag} (${plan.backend}) is ready.`);
        this.logger.info("llama.cpp installed", { tag: release.tag, backend: plan.backend, binary });
    }

    private async customModelInfo(
        custom: NonNullable<ServerConfig["customModel"]>,
        signal: AbortSignal,
    ): Promise<{ size?: number; sha256?: string }> {
        const url = `${HUGGINGFACE_URL}/api/models/${custom.repo}/tree/main?recursive=true`;
        const json = await this.deps.fetchJson(url, signal);
        const entry = findTreeEntry(json, custom.file);

        if (!entry) {
            throw new Error(`File ${custom.file} not found in ${custom.repo}`);
        }

        if (entry.size !== undefined) {
            this.knownModelSizes.set(`${custom.repo}/${custom.file}`, entry.size);
        }

        return entry;
    }

    private async installModel(config: ServerConfig, signal: AbortSignal): Promise<void> {
        const slot = this.model;
        const expected = this.expectedModel(config);

        if (!expected) {
            throw new Error("A custom model needs its Hugging Face repo and file; set them in the setup");
        }

        let target: ModelTarget;

        if (config.profile === "custom") {
            const custom = config.customModel;

            if (!custom) {
                throw new Error("A custom model needs its Hugging Face repo and file; set them in the setup");
            }

            slot.status.message = `Looking up ${custom.file} in ${custom.repo}…`;

            const info = await this.customModelInfo(custom, signal);

            target = {
                ...expected,
                expectedBytes: info.size,
                url: `${HUGGINGFACE_URL}/${custom.repo}/resolve/main/${custom.file}`,
                sha256: info.sha256,
            };
        } else {
            const spec = LOCAL_MODELS[config.profile];

            target = {
                ...expected,
                expectedBytes: spec.bytes,
                url: `${HUGGINGFACE_URL}/${spec.repo}/resolve/main/${spec.file}`,
                sha256: spec.sha256,
            };
        }

        if (signal.aborted) {
            throw new DownloadError("aborted", "download aborted");
        }

        const existing = await fileSizeOrNull(target.path);

        if (
            existing !== null &&
            (target.expectedBytes === undefined ? existing > 0 : existing === target.expectedBytes)
        ) {
            setProgress(slot.status, existing, existing);
            finishTask(slot.status, `${target.file} is already downloaded.`);

            return;
        }

        slot.status.message = `Downloading ${target.file}`;

        try {
            await this.deps.download({
                url: target.url,
                dest: target.path,
                expectedBytes: target.expectedBytes,
                sha256: target.sha256,
                signal,
                onProgress: (progress) =>
                    setProgress(slot.status, progress.received, progress.total || target.expectedBytes || 0),
            });
        } catch (error) {
            if (isAbort(error, signal)) {
                throw error;
            }

            throw new Error(describeDownloadFailure(error, target.file));
        }

        const size = (await fileSizeOrNull(target.path)) ?? 0;

        setProgress(slot.status, size, size);
        finishTask(slot.status, `${target.file} is ready.`);
        this.logger.info("model downloaded", { file: target.file, bytes: size });
    }
}

/** The HF tree entry for `file`: `lfs.size`/`lfs.oid` are the real size and sha256 of an LFS file. */
function findTreeEntry(json: unknown, file: string): { size?: number; sha256?: string } | undefined {
    if (!Array.isArray(json)) {
        return undefined;
    }

    for (const item of json) {
        if (!item || typeof item !== "object") {
            continue;
        }

        const entry = item as Record<string, unknown>;

        if (entry.type === "directory" || entry.path !== file) {
            continue;
        }

        const lfs = entry.lfs && typeof entry.lfs === "object" ? (entry.lfs as Record<string, unknown>) : undefined;
        const lfsSize = typeof lfs?.size === "number" && lfs.size > 0 ? lfs.size : undefined;
        const plainSize = typeof entry.size === "number" && entry.size > 0 ? entry.size : undefined;
        // For an LFS file the top-level `size` is the pointer, not the content.
        const size = lfsSize ?? plainSize;
        const oid = typeof lfs?.oid === "string" ? lfs.oid.trim().toLowerCase() : "";
        const sha256 = /^[a-f0-9]{64}$/.test(oid) ? oid : undefined;

        return { size, sha256 };
    }

    return undefined;
}

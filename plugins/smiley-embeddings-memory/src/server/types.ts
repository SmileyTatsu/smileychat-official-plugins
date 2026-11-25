// Contracts shared by the server-module files. The app does not publish types for the
// server host (see SmileyChat server/plugin-server-runtime.ts), so the shape the module relies
// on is written down here once.

import type { Backend } from "./releases";

export type HostLogger = {
    debug(message: string, detail?: unknown): void;
    info(message: string, detail?: unknown): void;
    warn(message: string, detail?: unknown): void;
    error(message: string, detail?: unknown): void;
};

export type ManagedProcess = {
    readonly pid: number;
    /** Resolves with the exit code (null when killed by a signal). */
    readonly exited: Promise<number | null>;
    kill(): Promise<void>;
};

export type PluginServerHost = {
    pluginId: string;
    /** Absolute folder that survives updates; everything the module writes goes here. */
    dataDir: string;
    logger: HostLogger;
    /** No shell: pass the real executable. stdout/stderr are drained to `logger.debug`, never exposed. */
    spawn(
        command: string,
        args?: string[],
        options?: { cwd?: string; env?: Record<string, string> },
    ): ManagedProcess;
};

export type PluginServerInstance = {
    handle?: (request: Request, subpath: string) => Response | Promise<Response>;
    dispose?: () => void | Promise<void>;
};

export type DeviceChoice = "auto" | "gpu" | "cpu";
export type GpuBackendChoice = "auto" | "cuda" | "vulkan";
export type ModelProfileChoice = "default" | "strong" | "custom";

/** What the browser saves through `POST /config`. Validated by config.ts; nothing here reaches a command line verbatim. */
export type ServerConfig = {
    profile: ModelProfileChoice;
    /** Hugging Face repo and file for `profile: "custom"`; ignored otherwise. */
    customModel?: { repo: string; file: string; pooling: "mean" | "last" };
    /** Manual paths win over downloaded files when set. */
    binaryPath?: string;
    modelPath?: string;
    /** `-c` for llama-server. */
    ctx: number;
    /** Minutes without a request before llama-server is stopped; 0 keeps it running. */
    idleMinutes: number;
    device: DeviceChoice;
    gpuBackend: GpuBackendChoice;
    /** A llama.cpp tag such as `b12686`, or `latest`. */
    releaseTag: string;
};

export const DEFAULT_SERVER_CONFIG: ServerConfig = {
    profile: "default",
    ctx: 4096,
    idleMinutes: 10,
    device: "auto",
    gpuBackend: "auto",
    releaseTag: "b12686",
};

/** Everything llama-process needs to start the server; produced by the installer from config + files on disk. */
export type LaunchSpec = {
    binary: string;
    model: string;
    /** Which build the binary is; `manual` when the user pointed at their own. */
    backend: Backend | "manual";
    /** 0 for CPU, 99 otherwise. */
    gpuLayers: number;
    pooling: "mean" | "last";
    ctx: number;
};

export type ProcessState = "stopped" | "starting" | "ready" | "error";

export type ProcessStatus = {
    state: ProcessState;
    pid?: number;
    port?: number;
    /** Backend and device as reported by the llama-server log, not as requested. */
    backend?: Backend | "manual";
    device?: string;
    layersOffloaded?: number;
    layersTotal?: number;
    /** What the user asked for; differs from `backend` after an `auto` fallback. */
    requestedDevice?: DeviceChoice;
    startedAt?: string;
    lastUsedAt?: string;
    lastError?: string;
};

export type TaskState = "idle" | "running" | "done" | "error" | "cancelled";

export type TaskStatus = {
    state: TaskState;
    /** Short human text such as `Downloading llama-b12686-bin-win-cuda-13.5-x64.zip`. */
    message?: string;
    received: number;
    total: number;
    error?: string;
};

export type InstallerStatus = {
    binary: {
        installed: boolean;
        tag?: string;
        backend?: Backend | "manual";
        path?: string;
        bytesOnDisk: number;
        task: TaskStatus;
    };
    model: {
        installed: boolean;
        file?: string;
        path?: string;
        expectedBytes?: number;
        bytesOnDisk: number;
        task: TaskStatus;
    };
};

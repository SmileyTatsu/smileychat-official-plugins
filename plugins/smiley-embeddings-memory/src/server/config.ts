// The server module's config.json: written through `POST /config`, read back by llama-process.
//
// SECURITY: nothing in this file may become a command-line argument except the numbers and
// enums validated here and the two paths (binaryPath/modelPath), which are forced to be
// absolute and to carry the exact file name llama-server needs. Every other key is dropped, so
// a hand-edited or hostile config.json cannot smuggle a flag into the spawn call.

import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, join } from "node:path";

import {
    DEFAULT_SERVER_CONFIG,
    type DeviceChoice,
    type GpuBackendChoice,
    type ModelProfileChoice,
    type ServerConfig,
} from "./types";

export class ConfigError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "ConfigError";
    }
}

const PROFILE_VALUES: readonly ModelProfileChoice[] = ["default", "strong", "custom"];
const DEVICE_VALUES: readonly DeviceChoice[] = ["auto", "gpu", "cpu"];
const GPU_BACKEND_VALUES: readonly GpuBackendChoice[] = ["auto", "cuda", "vulkan"];
const POOLING_VALUES = ["mean", "last"] as const;

const CTX_MIN = 512;
const CTX_MAX = 32768;
const IDLE_MINUTES_MAX = 1440;

const TAG_PATTERN = /^(latest|b\d+)$/;
// `owner/repo` as Hugging Face writes it; both halves must start with an alphanumeric so `.`
// and `..` can never be a segment on their own.
const REPO_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;
const MODEL_FILE_PATTERN = /^[A-Za-z0-9._-]+\.gguf$/;
// llama.cpp's server binary, optionally with an extension such as `.exe`.
const BINARY_NAME_PATTERN = /^llama-server(\.[a-z0-9]+)?$/i;
const GGUF_EXTENSION = /\.gguf$/i;

function configPath(dataDir: string): string {
    return join(dataDir, "config.json");
}

function asRecord(input: unknown, label: string): Record<string, unknown> {
    if (!input || typeof input !== "object" || Array.isArray(input)) {
        throw new ConfigError(`${label} must be a JSON object`);
    }

    return input as Record<string, unknown>;
}

function pickEnum<T extends string>(value: unknown, allowed: readonly T[], label: string, problems: string[]): T | undefined {
    if (value === undefined) {
        return undefined;
    }

    if (typeof value === "string" && (allowed as readonly string[]).includes(value)) {
        return value as T;
    }

    problems.push(`${label} must be one of ${allowed.join(", ")}`);

    return undefined;
}

function clampInteger(value: unknown, min: number, max: number, label: string, problems: string[]): number | undefined {
    if (value === undefined) {
        return undefined;
    }

    if (typeof value !== "number" || !Number.isFinite(value)) {
        problems.push(`${label} must be a number`);

        return undefined;
    }

    // HTML number inputs happily send `4096.5`; rounding is kinder than rejecting the form.
    return Math.min(max, Math.max(min, Math.round(value)));
}

function pickPatternedString(
    value: unknown,
    pattern: RegExp,
    label: string,
    hint: string,
    problems: string[],
): string | undefined {
    if (value === undefined) {
        return undefined;
    }

    if (typeof value !== "string") {
        problems.push(`${label} must be a string`);

        return undefined;
    }

    if (!pattern.test(value)) {
        problems.push(`${label} ${hint}`);

        return undefined;
    }

    return value;
}

function hasParentSegment(path: string): boolean {
    return path.split(/[\\/]/).includes("..");
}

function normalizeBinaryPath(value: unknown, problems: string[]): string | undefined {
    if (value === undefined || value === null) {
        return undefined;
    }

    if (typeof value !== "string") {
        problems.push("binaryPath must be a string");

        return undefined;
    }

    // An empty string is how the UI clears the setting; it is not a path.
    if (value.trim() === "") {
        return undefined;
    }

    if (!isAbsolute(value)) {
        problems.push(`binaryPath must be an absolute path, got "${value}"`);

        return undefined;
    }

    if (hasParentSegment(value)) {
        problems.push('binaryPath must not contain ".." segments');

        return undefined;
    }

    const name = basename(value);

    if (!BINARY_NAME_PATTERN.test(name)) {
        problems.push(`binaryPath must point at llama-server or llama-server.exe, got "${name}"`);

        return undefined;
    }

    return value;
}

function normalizeModelPath(value: unknown, problems: string[]): string | undefined {
    if (value === undefined || value === null) {
        return undefined;
    }

    if (typeof value !== "string") {
        problems.push("modelPath must be a string");

        return undefined;
    }

    if (value.trim() === "") {
        return undefined;
    }

    if (!isAbsolute(value)) {
        problems.push(`modelPath must be an absolute path, got "${value}"`);

        return undefined;
    }

    if (hasParentSegment(value)) {
        problems.push('modelPath must not contain ".." segments');

        return undefined;
    }

    if (!GGUF_EXTENSION.test(value)) {
        problems.push(`modelPath must end with .gguf, got "${basename(value)}"`);

        return undefined;
    }

    return value;
}

function normalizeCustomModel(
    value: unknown,
    problems: string[],
): NonNullable<ServerConfig["customModel"]> | undefined {
    if (value === undefined || value === null) {
        return undefined;
    }

    if (typeof value !== "object" || Array.isArray(value)) {
        problems.push("customModel must be an object with repo, file and pooling");

        return undefined;
    }

    const raw = value as Record<string, unknown>;

    const repo = pickPatternedString(
        raw.repo,
        REPO_PATTERN,
        "customModel.repo",
        'must look like "owner/repo"',
        problems,
    );
    const file = pickPatternedString(
        raw.file,
        MODEL_FILE_PATTERN,
        "customModel.file",
        "must be a single .gguf file name",
        problems,
    );
    const pooling = pickEnum(raw.pooling, POOLING_VALUES, "customModel.pooling", problems);

    // `..` matches the character class above, so reject it explicitly: the name is turned into
    // a path under models/ and must stay one level deep.
    if (file && file.includes("..")) {
        problems.push('customModel.file must not contain ".."');

        return undefined;
    }

    if (repo === undefined || file === undefined || pooling === undefined) {
        return undefined;
    }

    return { repo, file, pooling };
}

function validate(input: Record<string, unknown>): { config: ServerConfig; problems: string[] } {
    const problems: string[] = [];

    const profile = pickEnum(input.profile, PROFILE_VALUES, "profile", problems) ?? DEFAULT_SERVER_CONFIG.profile;
    const ctx = clampInteger(input.ctx, CTX_MIN, CTX_MAX, "ctx", problems) ?? DEFAULT_SERVER_CONFIG.ctx;
    const idleMinutes =
        clampInteger(input.idleMinutes, 0, IDLE_MINUTES_MAX, "idleMinutes", problems) ??
        DEFAULT_SERVER_CONFIG.idleMinutes;
    const device = pickEnum(input.device, DEVICE_VALUES, "device", problems) ?? DEFAULT_SERVER_CONFIG.device;
    const gpuBackend =
        pickEnum(input.gpuBackend, GPU_BACKEND_VALUES, "gpuBackend", problems) ?? DEFAULT_SERVER_CONFIG.gpuBackend;
    const releaseTag =
        pickPatternedString(input.releaseTag, TAG_PATTERN, "releaseTag", "must be \"latest\" or a tag such as \"b12686\"", problems) ??
        DEFAULT_SERVER_CONFIG.releaseTag;

    const binaryPath = normalizeBinaryPath(input.binaryPath, problems);
    const modelPath = normalizeModelPath(input.modelPath, problems);
    const customModel = normalizeCustomModel(input.customModel, problems);

    if (profile === "custom" && !customModel) {
        problems.push("customModel is required when profile is custom");
    }

    // Built from scratch, so unknown keys in the file or the patch are dropped here and nowhere
    // else has to guard against them.
    const config: ServerConfig = { profile, ctx, idleMinutes, device, gpuBackend, releaseTag };

    if (customModel) {
        config.customModel = customModel;
    }

    if (binaryPath) {
        config.binaryPath = binaryPath;
    }

    if (modelPath) {
        config.modelPath = modelPath;
    }

    return { config, problems };
}

async function writeConfigFile(file: string, config: ServerConfig): Promise<void> {
    await mkdir(dirname(file), { recursive: true });

    // Same directory as the target, so the rename is atomic on every filesystem we run on: a
    // crash leaves either the old file or the new one, never half of either.
    const temp = `${file}.${process.pid}.tmp`;

    try {
        await writeFile(temp, `${JSON.stringify(config, null, 4)}\n`, "utf8");
        await rename(temp, file);
    } catch (error) {
        await unlink(temp).catch(() => undefined);

        throw error;
    }
}

/** Current saved config. A missing or unreadable file yields the defaults, never an error. */
export async function loadConfig(dataDir: string): Promise<ServerConfig> {
    let raw: unknown;

    try {
        raw = JSON.parse(await readFile(configPath(dataDir), "utf8"));
    } catch {
        // First run (ENOENT) or a hand-edit that broke the JSON: the defaults always work.
        return { ...DEFAULT_SERVER_CONFIG };
    }

    const record = raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};
    // Runs through the same validator as a save; a field the validator rejects falls back to its
    // default instead of making the whole plugin unstartable.
    const { config } = validate({ ...DEFAULT_SERVER_CONFIG, ...record });

    return config;
}

/** Merge `patch` over the saved config, validate, and write it atomically. Nothing is written when invalid. */
export async function saveConfig(dataDir: string, patch: unknown): Promise<ServerConfig> {
    const record = asRecord(patch, "config");
    const current = await loadConfig(dataDir);
    const merged: Record<string, unknown> = { ...current };

    for (const [key, value] of Object.entries(record)) {
        // JSON has no `undefined`; ignoring it keeps "missing key" and "explicit clear" (null or
        // an empty string) distinct.
        if (value !== undefined) {
            merged[key] = value;
        }
    }

    const { config, problems } = validate(merged);

    if (problems.length > 0) {
        throw new ConfigError(problems.join("; "));
    }

    await writeConfigFile(configPath(dataDir), config);

    return config;
}

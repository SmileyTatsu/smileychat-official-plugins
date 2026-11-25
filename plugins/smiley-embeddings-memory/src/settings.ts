/**
 * Browser-side settings: shape, defaults, normalization and the two values other
 * modules derive from them (the server config and the active model identity).
 *
 * The embedding API key lives in these settings, which the app stores as plain
 * local JSON (`api.settings`). `connections:secrets` is not used on purpose: the
 * app only exposes provider secrets to plugins that declare that permission, and
 * this key is not a chat-provider key.
 */

import type { DeviceChoice, GpuBackendChoice, ModelProfileChoice } from "./server/types";
import type { ActiveModel } from "./embedder";
import { DEFAULT_PROFILE, DEFAULT_QWEN3_TASK, DEFAULT_THRESHOLDS, LOCAL_MODELS } from "./models";
import type { PrefixProfile } from "./models";

export type EmbeddingSource = "local" | "api";

export type PluginSettings = {
    enabled: boolean;
    source: EmbeddingSource;
    /** `custom` means the user typed a Hugging Face repo + file themselves. */
    profile: ModelProfileChoice;
    customModel: {
        repo: string;
        file: string;
        pooling: "mean" | "last";
        prefixProfile: PrefixProfile;
    };
    device: DeviceChoice;
    gpuBackend: GpuBackendChoice;
    /** llama.cpp tag such as `b12686`, or `latest`. */
    releaseTag: string;
    /** Manual paths win over the downloaded files when set. */
    binaryPath: string;
    modelPath: string;
    ctx: number;
    idleMinutes: number;
    api: {
        baseUrl: string;
        apiKey: string;
        model: string;
        prefixProfile: PrefixProfile;
    };
    qwenTask: string;
    chunkSize: number;
    chunkOverlap: number;
    queryMessages: number;
    topK: number;
    retain: number;
    /** null = auto: the active model profile's calibrated default. */
    threshold: number | null;
    depth: number;
    template: string;
    maxMemoryTokens: number;
    timeoutMs: number;
    autoIndexBacklog: boolean;
    /** Set once the user has seen "your messages will be sent to {host}". */
    apiNoticeAccepted: boolean;
    /** Set once the user has seen that local mode downloads programs and models. */
    downloadNoticeAccepted: boolean;
    /** First successful test; keeps the wizard collapsed after a reload. */
    setupComplete: boolean;
};

export const SETTINGS_KEY = "settings";

export const DEFAULT_TEMPLATE = `Relevant memories of past events from this conversation. They already happened; use them as background and do not repeat them as if they were new:
{{memories}}`;

const DEFAULT_API_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_API_MODEL = "text-embedding-3-small";
const DEFAULT_RELEASE_TAG = "b12686";
const MEMORIES_MARKER = "{{memories}}";

/** Also used by retrieval.ts to render the template. */
export const MEMORIES_PLACEHOLDER = MEMORIES_MARKER;

export const DEFAULT_SETTINGS: PluginSettings = {
    enabled: true,
    source: "local",
    profile: DEFAULT_PROFILE,
    customModel: {
        repo: "",
        file: "",
        pooling: "mean",
        prefixProfile: "none",
    },
    device: "auto",
    gpuBackend: "auto",
    releaseTag: DEFAULT_RELEASE_TAG,
    binaryPath: "",
    modelPath: "",
    ctx: 4096,
    idleMinutes: 10,
    api: {
        baseUrl: DEFAULT_API_BASE_URL,
        apiKey: "",
        model: DEFAULT_API_MODEL,
        prefixProfile: "none",
    },
    qwenTask: DEFAULT_QWEN3_TASK,
    chunkSize: 400,
    chunkOverlap: 40,
    queryMessages: 2,
    topK: 3,
    threshold: null,
    retain: 5,
    depth: 2,
    template: DEFAULT_TEMPLATE,
    maxMemoryTokens: 600,
    timeoutMs: 8000,
    autoIndexBacklog: true,
    apiNoticeAccepted: false,
    downloadNoticeAccepted: false,
    setupComplete: false,
};

/**
 * Rebuilds settings from whatever was stored, dropping unknown keys.
 *
 * Stored settings come from another version of this plugin, so nothing is
 * trusted: every field falls back to its default and every number is clamped to
 * a range the rest of the code assumes (a stored `topK: 5000` must not make a
 * retrieval walk the whole index).
 */
export function normalizeSettings(raw: unknown): PluginSettings {
    const source = isRecord(raw) ? raw : {};
    const api = isRecord(source.api) ? source.api : {};
    const custom = isRecord(source.customModel) ? source.customModel : {};

    const chunkSize = integerValue(source.chunkSize, 100, 2000, DEFAULT_SETTINGS.chunkSize);
    const binaryPath = stringValue(source.binaryPath);
    const modelPath = stringValue(source.modelPath);

    return {
        enabled: booleanValue(source.enabled, DEFAULT_SETTINGS.enabled),
        source: source.source === "api" || source.source === "local" ? source.source : DEFAULT_SETTINGS.source,
        profile: normalizeProfile(source.profile),
        customModel: {
            repo: stringValue(custom.repo),
            file: stringValue(custom.file),
            pooling: custom.pooling === "last" ? "last" : "mean",
            prefixProfile: normalizePrefixProfile(custom.prefixProfile),
        },
        device: normalizeDevice(source.device),
        gpuBackend: normalizeGpuBackend(source.gpuBackend),
        // An empty tag would make the installer ask GitHub for nothing.
        releaseTag: stringValue(source.releaseTag) || DEFAULT_RELEASE_TAG,
        binaryPath,
        modelPath,
        ctx: integerValue(source.ctx, 512, 32768, DEFAULT_SETTINGS.ctx),
        idleMinutes: integerValue(source.idleMinutes, 0, 1440, DEFAULT_SETTINGS.idleMinutes),
        api: {
            // The rest of the client appends `/embeddings`, so a trailing slash
            // would produce `//embeddings` and a 404 on strict servers.
            baseUrl: stripTrailingSlash(stringValue(api.baseUrl)) || DEFAULT_API_BASE_URL,
            apiKey: stringValue(api.apiKey),
            model: stringValue(api.model) || DEFAULT_API_MODEL,
            prefixProfile: normalizePrefixProfile(api.prefixProfile),
        },
        qwenTask: stringValue(source.qwenTask) || DEFAULT_QWEN3_TASK,
        chunkSize,
        chunkOverlap: integerValue(source.chunkOverlap, 0, Math.floor(chunkSize / 2), DEFAULT_SETTINGS.chunkOverlap),
        queryMessages: integerValue(source.queryMessages, 1, 10, DEFAULT_SETTINGS.queryMessages),
        topK: integerValue(source.topK, 1, 20, DEFAULT_SETTINGS.topK),
        // A threshold is a cosine, so the only meaningful range is 0..1.
        threshold: thresholdValue(source.threshold),
        retain: integerValue(source.retain, 0, 50, DEFAULT_SETTINGS.retain),
        depth: integerValue(source.depth, 0, 50, DEFAULT_SETTINGS.depth),
        template: templateValue(source.template),
        maxMemoryTokens: integerValue(source.maxMemoryTokens, 50, 4000, DEFAULT_SETTINGS.maxMemoryTokens),
        timeoutMs: integerValue(source.timeoutMs, 1000, 60000, DEFAULT_SETTINGS.timeoutMs),
        autoIndexBacklog: booleanValue(source.autoIndexBacklog, DEFAULT_SETTINGS.autoIndexBacklog),
        apiNoticeAccepted: booleanValue(source.apiNoticeAccepted, DEFAULT_SETTINGS.apiNoticeAccepted),
        downloadNoticeAccepted: booleanValue(
            source.downloadNoticeAccepted,
            DEFAULT_SETTINGS.downloadNoticeAccepted,
        ),
        setupComplete: booleanValue(source.setupComplete, DEFAULT_SETTINGS.setupComplete),
    };
}

/**
 * The subset `POST /config` accepts. Paths are omitted (not sent empty) because
 * the server treats a present path as "use this file" and its validator rejects
 * an empty string.
 */
export function serverConfigOf(s: PluginSettings): Record<string, unknown> {
    const config: Record<string, unknown> = {
        profile: s.profile,
        ctx: s.ctx,
        idleMinutes: s.idleMinutes,
        device: s.device,
        gpuBackend: s.gpuBackend,
        releaseTag: s.releaseTag,
    };

    if (s.profile === "custom") {
        config.customModel = { repo: s.customModel.repo, file: s.customModel.file, pooling: s.customModel.pooling };
    }

    if (s.binaryPath) {
        config.binaryPath = s.binaryPath;
    }

    if (s.modelPath) {
        config.modelPath = s.modelPath;
    }

    return config;
}

/**
 * The cosine cut-off retrieval should use.
 *
 * Model-aware on purpose: EmbeddingGemma scores unrelated text around 0.61-0.69
 * while Qwen3-4B scores it at 0.26-0.48, so a single 0.3 either injects nonsense
 * (gemma) or drops real memories (qwen3). A stored number always wins; `null`
 * means "auto" and picks the profile's calibrated default.
 */
export function effectiveThreshold(s: PluginSettings, prefixProfile: PrefixProfile): number {
    const fallback = DEFAULT_THRESHOLDS[prefixProfile] ?? DEFAULT_THRESHOLDS.none;
    const value = s.threshold ?? fallback;

    // Clamped again here so a hand-built settings object cannot pass 1.5 through.
    return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : fallback;
}

/**
 * Identity of the model the embedder will use, written into the index manifest.
 *
 * `dim` stays undefined for anything not in the catalog: a custom repo or a
 * manual file can be any size, and guessing 768 there would make a correct index
 * look mismatched.
 */
export function activeModelOf(s: PluginSettings): ActiveModel {
    if (s.source === "api") {
        return {
            id: `api:${hostOf(s.api.baseUrl)}/${s.api.model}`,
            source: "api",
            prefixProfile: s.api.prefixProfile,
        };
    }

    // A manual file wins over the profile: the user pointed at a specific model,
    // but the profile still says which prefix/task that model expects.
    const prefixProfile = s.profile === "custom" ? s.customModel.prefixProfile : LOCAL_MODELS[s.profile].prefixProfile;

    if (s.modelPath) {
        return {
            id: `local-file:${baseName(s.modelPath)}`,
            source: "local",
            prefixProfile,
        };
    }

    if (s.profile === "custom") {
        return {
            id: `local:${s.customModel.repo}/${s.customModel.file}`,
            source: "local",
            prefixProfile,
        };
    }

    const spec = LOCAL_MODELS[s.profile];

    return {
        id: `local:${spec.repo}/${spec.file}`,
        source: "local",
        prefixProfile,
        dim: spec.dim,
    };
}

function templateValue(value: unknown): string {
    const template = stringValue(value);

    // The memories block is the only reason the injection exists; a template
    // without the marker would silently inject the instructions and no memories.
    return template.includes(MEMORIES_MARKER) ? template : DEFAULT_TEMPLATE;
}

function normalizeProfile(value: unknown): ModelProfileChoice {
    return value === "strong" || value === "custom" || value === "default" ? value : DEFAULT_PROFILE;
}

function normalizeDevice(value: unknown): DeviceChoice {
    return value === "gpu" || value === "cpu" || value === "auto" ? value : "auto";
}

function normalizeGpuBackend(value: unknown): GpuBackendChoice {
    return value === "cuda" || value === "vulkan" || value === "auto" ? value : "auto";
}

function normalizePrefixProfile(value: unknown): PrefixProfile {
    return value === "qwen3" || value === "gemma" || value === "none" ? value : "none";
}

/** `new URL` needs a scheme; a half-typed URL must not throw or change identity. */
function hostOf(url: string): string {
    try {
        const parsed = new URL(url);

        // `new URL("localhost:8080/v1")` parses as the protocol "localhost:" with
        // no host, which is not a URL a fetch would accept; treat it as unsafe.
        if (parsed.host) {
            return parsed.host;
        }
    } catch {
        // No scheme at all: fall through to splitting the string by hand.
    }

    return url.replace(/^[a-z]+:\/\//i, "").replace(/\/.*$/, "");
}

/** Splits on both separators: Windows paths arrive with backslashes. */
function baseName(path: string): string {
    const parts = path.split(/[\\/]/);
    return parts[parts.length - 1] || path;
}

function stripTrailingSlash(url: string): string {
    return url.replace(/\/+$/, "");
}

function integerValue(value: unknown, minimum: number, maximum: number, fallback: number): number {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return fallback;
    }

    return Math.min(maximum, Math.max(minimum, Math.round(number)));
}

/** `null` means auto: the profile's calibrated default is used at retrieval time. */
function thresholdValue(value: unknown): number | null {
    // A string or a missing value is "auto", not 0: 0 would inject everything.
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return null;
    }

    return Math.min(1, Math.max(0, value));
}

function booleanValue(value: unknown, fallback: boolean): boolean {
    return typeof value === "boolean" ? value : fallback;
}

function stringValue(value: unknown): string {
    return typeof value === "string" ? value.trim() : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

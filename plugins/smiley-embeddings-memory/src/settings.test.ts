import { describe, expect, test } from "bun:test";

import {
    activeModelOf,
    DEFAULT_SETTINGS,
    DEFAULT_TEMPLATE,
    effectiveThreshold,
    normalizeSettings,
    serverConfigOf,
    SETTINGS_KEY,
} from "./settings";

describe("normalizeSettings", () => {
    test("returns the defaults for missing or unusable input", () => {
        expect(normalizeSettings(undefined)).toEqual(DEFAULT_SETTINGS);
        expect(normalizeSettings(null)).toEqual(DEFAULT_SETTINGS);
        expect(normalizeSettings("nonsense")).toEqual(DEFAULT_SETTINGS);
        expect(normalizeSettings([])).toEqual(DEFAULT_SETTINGS);
        expect(normalizeSettings({})).toEqual(DEFAULT_SETTINGS);
    });

    test("drops keys it does not know", () => {
        const settings = normalizeSettings({ enabled: false, topK: 4, leftover: "x", __proto__key: 1 });

        expect(Object.keys(settings).sort()).toEqual(Object.keys(DEFAULT_SETTINGS).sort());
        expect((settings as unknown as Record<string, unknown>).leftover).toBeUndefined();
    });

    test("clamps every number to the range the rest of the plugin assumes", () => {
        const low = normalizeSettings({
            chunkSize: 50,
            chunkOverlap: -10,
            queryMessages: 0,
            topK: 0,
            threshold: -3,
            retain: -1,
            depth: -5,
            maxMemoryTokens: 10,
            timeoutMs: 5,
            ctx: 100,
            idleMinutes: -2,
        });

        expect(low.chunkSize).toBe(100);
        expect(low.chunkOverlap).toBe(0);
        expect(low.queryMessages).toBe(1);
        expect(low.topK).toBe(1);
        expect(low.threshold).toBe(0);
        expect(low.retain).toBe(0);
        expect(low.depth).toBe(0);
        expect(low.maxMemoryTokens).toBe(50);
        expect(low.timeoutMs).toBe(1000);
        expect(low.ctx).toBe(512);
        expect(low.idleMinutes).toBe(0);

        const high = normalizeSettings({
            chunkSize: 9000,
            chunkOverlap: 9999,
            queryMessages: 99,
            topK: 5000,
            threshold: 7,
            retain: 900,
            depth: 900,
            maxMemoryTokens: 99999,
            timeoutMs: 999999,
            ctx: 1_000_000,
            idleMinutes: 99999,
        });

        expect(high.chunkSize).toBe(2000);
        expect(high.chunkOverlap).toBe(1000);
        expect(high.queryMessages).toBe(10);
        expect(high.topK).toBe(20);
        expect(high.threshold).toBe(1);
        expect(high.retain).toBe(50);
        expect(high.depth).toBe(50);
        expect(high.maxMemoryTokens).toBe(4000);
        expect(high.timeoutMs).toBe(60000);
        expect(high.ctx).toBe(32768);
        expect(high.idleMinutes).toBe(1440);
    });

    test("keeps overlap under half the chunk size and rounds fractions", () => {
        const settings = normalizeSettings({ chunkSize: 400.7, chunkOverlap: 900 });

        expect(settings.chunkSize).toBe(401);
        expect(settings.chunkOverlap).toBe(200);
    });

    test("falls back to defaults for unknown enum values", () => {
        const settings = normalizeSettings({
            source: "quantum",
            profile: "huge",
            device: "tpu",
            gpuBackend: "metal",
            customModel: { pooling: "max", prefixProfile: "clip" },
            api: { prefixProfile: "clip" },
        });

        expect(settings.source).toBe("local");
        expect(settings.profile).toBe("default");
        expect(settings.device).toBe("auto");
        expect(settings.gpuBackend).toBe("auto");
        expect(settings.customModel.pooling).toBe("mean");
        expect(settings.customModel.prefixProfile).toBe("none");
        expect(settings.api.prefixProfile).toBe("none");
    });

    test("trims strings and strips the trailing slash from the base URL", () => {
        const settings = normalizeSettings({
            binaryPath: "  C:/llama/llama-server.exe  ",
            modelPath: " C:/models/mine.gguf ",
            releaseTag: "  latest ",
            qwenTask: "  find things  ",
            api: { baseUrl: " https://example.test/v1/// ", apiKey: " sk-abc ", model: " m " },
        });

        expect(settings.binaryPath).toBe("C:/llama/llama-server.exe");
        expect(settings.modelPath).toBe("C:/models/mine.gguf");
        expect(settings.releaseTag).toBe("latest");
        expect(settings.qwenTask).toBe("find things");
        expect(settings.api.baseUrl).toBe("https://example.test/v1");
        expect(settings.api.apiKey).toBe("sk-abc");
        expect(settings.api.model).toBe("m");
    });

    test("replaces empty required strings with their defaults", () => {
        const settings = normalizeSettings({
            releaseTag: "   ",
            qwenTask: "",
            api: { baseUrl: "", model: "" },
        });

        expect(settings.releaseTag).toBe(DEFAULT_SETTINGS.releaseTag);
        expect(settings.qwenTask).toBe(DEFAULT_SETTINGS.qwenTask);
        expect(settings.api.baseUrl).toBe(DEFAULT_SETTINGS.api.baseUrl);
        expect(settings.api.model).toBe(DEFAULT_SETTINGS.api.model);
    });

    test("only keeps a template that contains the memories marker", () => {
        expect(normalizeSettings({ template: "Just words" }).template).toBe(DEFAULT_TEMPLATE);
        expect(normalizeSettings({ template: "   " }).template).toBe(DEFAULT_TEMPLATE);

        const custom = normalizeSettings({ template: "Past:\n{{memories}}\nEnd." });
        expect(custom.template).toBe("Past:\n{{memories}}\nEnd.");
    });

    test("keeps booleans but refuses truthy stand-ins", () => {
        const settings = normalizeSettings({ enabled: "yes", autoIndexBacklog: false, apiNoticeAccepted: 1 });

        expect(settings.enabled).toBe(DEFAULT_SETTINGS.enabled);
        expect(settings.autoIndexBacklog).toBe(false);
        expect(settings.apiNoticeAccepted).toBe(DEFAULT_SETTINGS.apiNoticeAccepted);
    });

    test("tracks whether setup was completed, defaulting to false", () => {
        expect(DEFAULT_SETTINGS.setupComplete).toBe(false);
        expect(normalizeSettings({}).setupComplete).toBe(false);
        expect(normalizeSettings({ setupComplete: true }).setupComplete).toBe(true);
        expect(normalizeSettings({ setupComplete: "1" }).setupComplete).toBe(false);
    });

    test("tracks the download notice separately from the API notice", () => {
        expect(DEFAULT_SETTINGS.downloadNoticeAccepted).toBe(false);
        expect(normalizeSettings({}).downloadNoticeAccepted).toBe(false);
        expect(normalizeSettings({ downloadNoticeAccepted: true }).downloadNoticeAccepted).toBe(true);
        expect(normalizeSettings({ downloadNoticeAccepted: "yes" }).downloadNoticeAccepted).toBe(false);
    });

    test("keeps an explicit threshold, clamps it, and stores auto as null", () => {
        expect(DEFAULT_SETTINGS.threshold).toBeNull();
        expect(normalizeSettings({}).threshold).toBeNull();
        expect(normalizeSettings({ threshold: null }).threshold).toBeNull();
        expect(normalizeSettings({ threshold: 0.42 }).threshold).toBe(0.42);
        expect(normalizeSettings({ threshold: -1 }).threshold).toBe(0);
        expect(normalizeSettings({ threshold: 5 }).threshold).toBe(1);
        // A string is not a threshold; auto is safer than 0, which injects everything.
        expect(normalizeSettings({ threshold: "0.5" }).threshold).toBeNull();
    });

    test("round-trips the defaults", () => {
        expect(normalizeSettings(DEFAULT_SETTINGS)).toEqual(DEFAULT_SETTINGS);
    });

    test("uses a stable storage key", () => {
        expect(SETTINGS_KEY).toBe("settings");
    });
});

describe("serverConfigOf", () => {
    test("sends the catalog profile without a custom model", () => {
        expect(serverConfigOf(normalizeSettings({ profile: "strong", device: "cpu", gpuBackend: "vulkan" }))).toEqual({
            profile: "strong",
            ctx: 4096,
            idleMinutes: 10,
            device: "cpu",
            gpuBackend: "vulkan",
            releaseTag: "b12686",
            binaryPath: "",
            modelPath: "",
        });
    });

    test("includes the custom model only for the custom profile", () => {
        const config = serverConfigOf(
            normalizeSettings({
                profile: "custom",
                customModel: { repo: "org/model-GGUF", file: "m.gguf", pooling: "last" },
            }),
        );

        expect(config.customModel).toEqual({ repo: "org/model-GGUF", file: "m.gguf", pooling: "last" });
    });

    test("always sends both path fields, empty when unset, so clearing one reaches the server", () => {
        // Omitting an empty path would keep the value config.json already has: clearing the
        // field in the UI has to be an explicit "".
        expect(serverConfigOf(DEFAULT_SETTINGS)).toMatchObject({ binaryPath: "", modelPath: "" });

        const config = serverConfigOf(
            normalizeSettings({ binaryPath: "  C:/bin/llama-server.exe  ", modelPath: "C:/models/a.gguf" }),
        );

        expect(config.binaryPath).toBe("C:/bin/llama-server.exe");
        expect(config.modelPath).toBe("C:/models/a.gguf");
    });

    test("a path that is only whitespace counts as cleared", () => {
        expect(
            serverConfigOf(normalizeSettings({ binaryPath: "   ", modelPath: "	" })),
        ).toMatchObject({ binaryPath: "", modelPath: "" });
    });

    test("never sends the browser-only settings to the server", () => {
        const config = serverConfigOf(
            normalizeSettings({ setupComplete: true, downloadNoticeAccepted: true }),
        );

        // The module has no use for them and its validator rejects unknown keys.
        expect(config).not.toHaveProperty("setupComplete");
        expect(config).not.toHaveProperty("downloadNoticeAccepted");
        expect(config).not.toHaveProperty("threshold");
        expect(config).not.toHaveProperty("enabled");
        expect(config).not.toHaveProperty("api");
    });
});

describe("effectiveThreshold", () => {
    test("uses the calibrated default of the model's prefix profile when auto", () => {
        const auto = normalizeSettings({});

        expect(effectiveThreshold(auto, "gemma")).toBe(0.76);
        expect(effectiveThreshold(auto, "qwen3")).toBe(0.59);
        expect(effectiveThreshold(auto, "none")).toBe(0.3);
    });

    test("keeps the cut-offs above the strongest unrelated probe measured", () => {
        // Three-chat calibration (see DEFAULT_THRESHOLDS): the worst unrelated
        // probe scored 0.735 with Gemma and 0.551 with Qwen3, so a cut-off at or
        // below those numbers starts injecting chat noise in long sessions.
        expect(effectiveThreshold(DEFAULT_SETTINGS, "gemma")).toBeGreaterThan(0.735);
        expect(effectiveThreshold(DEFAULT_SETTINGS, "qwen3")).toBeGreaterThan(0.551);
    });

    test("an explicit number overrides every profile", () => {
        const explicit = normalizeSettings({ threshold: 0.45 });

        expect(effectiveThreshold(explicit, "gemma")).toBe(0.45);
        expect(effectiveThreshold(explicit, "qwen3")).toBe(0.45);
        expect(effectiveThreshold(explicit, "none")).toBe(0.45);
    });

    test("clamps values that bypassed normalization", () => {
        const high = { ...DEFAULT_SETTINGS, threshold: 1.5 };
        const broken = { ...DEFAULT_SETTINGS, threshold: Number.NaN };

        expect(effectiveThreshold(high, "qwen3")).toBe(1);
        expect(effectiveThreshold(broken, "qwen3")).toBe(0.59);
    });
});

describe("activeModelOf", () => {
    test("names a catalog model by repo and file and keeps its dimension", () => {
        expect(activeModelOf(DEFAULT_SETTINGS)).toEqual({
            id: "local:unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-UD-Q4_K_XL.gguf",
            source: "local",
            prefixProfile: "gemma",
            dim: 768,
        });

        const strong = activeModelOf(normalizeSettings({ profile: "strong" }));
        expect(strong.id).toBe("local:Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q4_K_M.gguf");
        expect(strong.dim).toBe(2560);
        expect(strong.prefixProfile).toBe("qwen3");
    });

    test("leaves the dimension unknown for a custom repo", () => {
        const model = activeModelOf(
            normalizeSettings({
                profile: "custom",
                customModel: { repo: "org/custom-GGUF", file: "custom.gguf", prefixProfile: "qwen3" },
            }),
        );

        expect(model).toEqual({
            id: "local:org/custom-GGUF/custom.gguf",
            source: "local",
            prefixProfile: "qwen3",
            dim: undefined,
        });
    });

    test("identifies a manual file by name and keeps the profile's prefixes", () => {
        const model = activeModelOf(normalizeSettings({ modelPath: "C:\\models\\mine.gguf" }));

        expect(model.id).toBe("local-file:mine.gguf");
        expect(model.dim).toBeUndefined();
        expect(model.prefixProfile).toBe("gemma");

        const strong = activeModelOf(normalizeSettings({ profile: "strong", modelPath: "/opt/models/q.gguf" }));
        expect(strong.id).toBe("local-file:q.gguf");
        expect(strong.prefixProfile).toBe("qwen3");
    });

    test("names an API model by host and model id", () => {
        const model = activeModelOf(
            normalizeSettings({
                source: "api",
                api: { baseUrl: "https://api.openai.com/v1/", model: "text-embedding-3-large", prefixProfile: "none" },
            }),
        );

        expect(model).toEqual({
            id: "api:api.openai.com/text-embedding-3-large",
            source: "api",
            prefixProfile: "none",
            dim: undefined,
        });
    });

    test("still produces a stable id for a base URL without a scheme", () => {
        const model = activeModelOf(
            normalizeSettings({ source: "api", api: { baseUrl: "localhost:8080/v1", model: "m" } }),
        );

        expect(model.id).toBe("api:localhost:8080/m");
    });
});

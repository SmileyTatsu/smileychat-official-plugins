import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ConfigError, loadConfig, saveConfig } from "./config";
import { DEFAULT_SERVER_CONFIG, type ServerConfig } from "./types";

let dir = "";

beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "smiley-config-"));
});

afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
});

async function writeConfigFile(content: unknown): Promise<void> {
    const text = typeof content === "string" ? content : JSON.stringify(content, null, 4);

    await writeFile(join(dir, "config.json"), text, "utf8");
}

async function readConfigFile(): Promise<unknown> {
    return JSON.parse(await readFile(join(dir, "config.json"), "utf8")) as unknown;
}

async function tempFileNames(): Promise<string[]> {
    return readdir(dir);
}

function absolute(name: string): string {
    return join(dir, name);
}

describe("loadConfig", () => {
    test("missing file yields the defaults", async () => {
        expect(await loadConfig(dir)).toEqual(DEFAULT_SERVER_CONFIG);
    });

    test("corrupt json yields the defaults instead of an error", async () => {
        await writeConfigFile("{ not json");

        expect(await loadConfig(dir)).toEqual(DEFAULT_SERVER_CONFIG);
    });

    test("a json value that is not an object yields the defaults", async () => {
        await writeConfigFile("[1, 2, 3]");

        expect(await loadConfig(dir)).toEqual(DEFAULT_SERVER_CONFIG);
    });

    test("unknown keys are dropped", async () => {
        await writeConfigFile({ ...DEFAULT_SERVER_CONFIG, extraFlag: "--host 0.0.0.0", binaryArgs: ["--mlock"] });

        const config = await loadConfig(dir);

        expect(config).toEqual(DEFAULT_SERVER_CONFIG);
        expect(Object.keys(config).sort()).toEqual(["ctx", "device", "gpuBackend", "idleMinutes", "profile", "releaseTag"]);
    });

    test("hand-edited bad fields fall back to their defaults, good fields survive", async () => {
        await writeConfigFile({
            profile: "hax",
            ctx: 999_999,
            idleMinutes: -3,
            device: "gpu",
            gpuBackend: "cuda",
            releaseTag: "v1.0.0",
            binaryPath: "llama-server.exe",
        });

        expect(await loadConfig(dir)).toEqual({
            ...DEFAULT_SERVER_CONFIG,
            // Numbers are clamped rather than reverted: the extremes mean "as large as possible"
            // / "never idle out" and both are usable.
            ctx: 32_768,
            idleMinutes: 0,
            device: "gpu",
            gpuBackend: "cuda",
        });
    });

    test("keeps valid paths from the file", async () => {
        await writeConfigFile({ ...DEFAULT_SERVER_CONFIG, binaryPath: absolute("llama-server.exe"), modelPath: absolute("x.gguf") });

        const config = await loadConfig(dir);

        expect(config.binaryPath).toBe(absolute("llama-server.exe"));
        expect(config.modelPath).toBe(absolute("x.gguf"));
    });
});

describe("saveConfig", () => {
    test("merges the patch over the saved config and writes it", async () => {
        const first = await saveConfig(dir, { profile: "strong", ctx: 2048, device: "gpu" });

        expect(first).toEqual({ ...DEFAULT_SERVER_CONFIG, profile: "strong", ctx: 2048, device: "gpu" });
        expect(await readConfigFile()).toEqual(first);

        const second = await saveConfig(dir, { idleMinutes: 0 });

        expect(second).toEqual({ ...first, idleMinutes: 0 });
        expect(await loadConfig(dir)).toEqual(second);
    });

    test("clamps ctx and idleMinutes into their ranges", async () => {
        expect((await saveConfig(dir, { ctx: 100 })).ctx).toBe(512);
        expect((await saveConfig(dir, { ctx: 999_999 })).ctx).toBe(32_768);
        expect((await saveConfig(dir, { idleMinutes: -5 })).idleMinutes).toBe(0);
        expect((await saveConfig(dir, { idleMinutes: 5000 })).idleMinutes).toBe(1440);
    });

    test("rounds a fractional number instead of rejecting the form", async () => {
        expect((await saveConfig(dir, { ctx: 4096.5 })).ctx).toBe(4097);
    });

    test("rejects a non numeric ctx", async () => {
        await expect(saveConfig(dir, { ctx: "4096" })).rejects.toThrow(ConfigError);
        await expect(saveConfig(dir, { ctx: null })).rejects.toThrow(/ctx must be a number/);
    });

    test("rejects an unknown enum and writes nothing", async () => {
        await saveConfig(dir, { profile: "strong" });
        const before = await readFile(join(dir, "config.json"), "utf8");

        await expect(saveConfig(dir, { profile: "hax" })).rejects.toThrow(/profile must be one of default, strong, custom/);
        await expect(saveConfig(dir, { device: "tpu" })).rejects.toThrow(/device must be one of/);
        await expect(saveConfig(dir, { gpuBackend: "metal" })).rejects.toThrow(/gpuBackend must be one of/);

        expect(await readFile(join(dir, "config.json"), "utf8")).toBe(before);
    });

    test("rejects a bad release tag", async () => {
        for (const releaseTag of ["", "b", "v1.0.0", "latest ", "b12a", "latest;rm -rf /"]) {
            await expect(saveConfig(dir, { releaseTag })).rejects.toThrow(/releaseTag/);
        }

        expect((await saveConfig(dir, { releaseTag: "latest" })).releaseTag).toBe("latest");
        expect((await saveConfig(dir, { releaseTag: "b12686" })).releaseTag).toBe("b12686");
    });

    test("rejects a patch that is not an object", async () => {
        for (const patch of [null, 42, "profile=strong", [1]]) {
            await expect(saveConfig(dir, patch)).rejects.toThrow(ConfigError);
        }
    });

    test("writes atomically: no temp file survives and the json is pretty printed", async () => {
        await saveConfig(dir, { profile: "strong" });

        expect((await tempFileNames()).filter((name) => name.endsWith(".tmp"))).toEqual([]);

        const text = await readFile(join(dir, "config.json"), "utf8");

        expect(text.endsWith("\n")).toBe(true);
        expect(text).toContain('\n    "profile": "strong"');
    });

    test("a failed validation leaves the previous file byte for byte intact", async () => {
        await saveConfig(dir, { profile: "strong", ctx: 8192 });

        const before = await readFile(join(dir, "config.json"), "utf8");

        await expect(saveConfig(dir, { ctx: 8192, modelPath: "model.gguf" })).rejects.toThrow(ConfigError);

        expect(await readFile(join(dir, "config.json"), "utf8")).toBe(before);
        expect((await tempFileNames()).filter((name) => name.endsWith(".tmp"))).toEqual([]);
    });
});

describe("path validation", () => {
    // Built inside the tests: `dir` only exists once beforeEach has run.
    const validBinaryNames = ["llama-server.exe", "llama-server", "Llama-Server.EXE"];

    test("accepts an absolute llama-server path, whatever the extension case", async () => {
        for (const binaryPath of validBinaryNames.map(absolute)) {
            expect((await saveConfig(dir, { binaryPath })).binaryPath).toBe(binaryPath);
        }
    });

    test("rejects relative, parent-traversing and non llama-server binary paths", async () => {
        const invalid = [
            "llama-server.exe",
            "./llama-server.exe",
            `${dir}/../llama-server.exe`,
            absolute("not-llama-server.exe"),
            absolute("llama.exe"),
            absolute("cmd.exe"),
            `${absolute("llama-server.exe")} --mlock`,
        ];

        for (const binaryPath of invalid) {
            await expect(saveConfig(dir, { binaryPath })).rejects.toThrow(ConfigError);
        }
    });

    test("an empty string or null clears a saved path", async () => {
        await saveConfig(dir, { binaryPath: absolute("llama-server.exe"), modelPath: absolute("m.gguf") });

        const cleared = await saveConfig(dir, { binaryPath: "", modelPath: null });

        expect(cleared.binaryPath).toBeUndefined();
        expect(cleared.modelPath).toBeUndefined();
        expect(await loadConfig(dir)).toEqual(cleared);
    });

    test("a cleared path disappears from the file instead of lingering", async () => {
        await saveConfig(dir, { binaryPath: absolute("llama-server.exe"), modelPath: absolute("m.gguf") });

        // The exact repro: the field was set, then emptied; the file must not keep the old path.
        await saveConfig(dir, { binaryPath: "" });

        const afterClear = (await readConfigFile()) as Record<string, unknown>;

        expect(afterClear).not.toHaveProperty("binaryPath");
        // Only the field that was cleared goes away.
        expect(afterClear.modelPath).toBe(absolute("m.gguf"));

        await saveConfig(dir, { binaryPath: absolute("llama-server.exe"), modelPath: null });

        const afterReset = (await readConfigFile()) as Record<string, unknown>;

        expect(afterReset.binaryPath).toBe(absolute("llama-server.exe"));
        expect(afterReset).not.toHaveProperty("modelPath");
        // Reloading must not bring either key back.
        expect((await loadConfig(dir)).modelPath).toBeUndefined();
        expect((await loadConfig(dir)).binaryPath).toBe(absolute("llama-server.exe"));
    });

    test("an absent key keeps the stored path", async () => {
        const binaryPath = absolute("llama-server.exe");

        await saveConfig(dir, { binaryPath, modelPath: absolute("m.gguf") });
        // A partial save from another tab (device change) must not wipe the manual paths.
        await saveConfig(dir, { device: "cpu" });

        const reloaded = await loadConfig(dir);

        expect(reloaded.binaryPath).toBe(binaryPath);
        expect(reloaded.modelPath).toBe(absolute("m.gguf"));
    });

    test("load does not resurrect an empty path from a hand-edited file", async () => {
        await writeConfigFile({ ...DEFAULT_SERVER_CONFIG, binaryPath: "", modelPath: "   " });

        const loaded = await loadConfig(dir);

        expect(loaded.binaryPath).toBeUndefined();
        expect(loaded.modelPath).toBeUndefined();
        expect(JSON.stringify(loaded)).not.toContain("binaryPath");
    });

    test("accepts an absolute .gguf model path and nothing else", async () => {
        expect((await saveConfig(dir, { modelPath: absolute("model.Q4_K_M.gguf") })).modelPath).toBe(
            absolute("model.Q4_K_M.gguf"),
        );

        const invalid = [
            "model.gguf",
            absolute("model.bin"),
            absolute("model.gguf.txt"),
            `${dir}/../model.gguf`,
            `${absolute("model.gguf")}" --host 0.0.0.0`,
        ];

        for (const modelPath of invalid) {
            await expect(saveConfig(dir, { modelPath })).rejects.toThrow(ConfigError);
        }
    });
});

describe("custom model validation", () => {
    const pooling = "mean" as const;

    test("is required when the profile is custom", async () => {
        await expect(saveConfig(dir, { profile: "custom" })).rejects.toThrow(/customModel is required/);
    });

    test("accepts an owner/repo and a single .gguf file name", async () => {
        const config: ServerConfig = await saveConfig(dir, {
            profile: "custom",
            customModel: { repo: "Qwen/Qwen3-Embedding-0.6B-GGUF", file: "Qwen3-Embedding-0.6B-Q8_0.gguf", pooling: "last" },
        });

        expect(config.profile).toBe("custom");
        expect(config.customModel).toEqual({
            repo: "Qwen/Qwen3-Embedding-0.6B-GGUF",
            file: "Qwen3-Embedding-0.6B-Q8_0.gguf",
            pooling: "last",
        });
    });

    test("rejects injection attempts in the repo", async () => {
        for (const repo of [
            "../etc/passwd",
            "owner/repo/../../x",
            "owner",
            "/repo",
            "owner/",
            "owner/repo name",
            "owner/repo\\x",
            "owner/repo?x=1",
            "owner//repo",
            "owner/repo\n--mlock",
        ]) {
            await expect(saveConfig(dir, { profile: "custom", customModel: { repo, file: "a.gguf", pooling } })).rejects.toThrow(
                /customModel\.repo/,
            );
        }
    });

    test("rejects a file name with slashes, a leading dot dot or another extension", async () => {
        for (const file of ["../a.gguf", "sub/a.gguf", "a\\b.gguf", "..gguf", "a.bin", "a.gguf ", "/etc/a.gguf"]) {
            await expect(
                saveConfig(dir, { profile: "custom", customModel: { repo: "owner/repo", file, pooling } }),
            ).rejects.toThrow(/customModel\.file/);
        }
    });

    test("rejects an unknown pooling", async () => {
        await expect(
            saveConfig(dir, {
                profile: "custom",
                customModel: { repo: "owner/repo", file: "a.gguf", pooling: "cls" },
            }),
        ).rejects.toThrow(/customModel\.pooling/);
    });

    test("a valid custom model survives a reload", async () => {
        const saved = await saveConfig(dir, {
            profile: "custom",
            customModel: { repo: "owner/repo", file: "a.gguf", pooling: "last" },
        });

        expect(await loadConfig(dir)).toEqual(saved);
    });
});

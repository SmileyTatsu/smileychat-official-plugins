// Manual smoke test: drives LlamaProcess against a real llama.cpp build and a real model.
//
// Not part of `bun test` — it needs a multi-GB model on disk and a GPU. Usage:
//
//   bun scripts/smoke-llama.ts <model.gguf> <bin-root> [data-dir]
//
// <bin-root> is the folder that holds the extracted releases (llama-b<tag>-bin-win-cuda-13.5-x64,
// ...-vulkan-x64, ...-cpu-x64). Every build found there is started in turn, so this also proves
// the CUDA -> Vulkan -> CPU device reporting the status panel shows.

import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { LlamaProcess } from "../src/server/llama-process";
import type { LaunchSpec, PluginServerHost } from "../src/server/types";

const [, , modelPath, binRoot, dataDirArg] = process.argv;

if (modelPath === undefined || binRoot === undefined) {
    console.error("usage: bun scripts/smoke-llama.ts <model.gguf> <bin-root> [data-dir]");
    process.exit(2);
}

const BINARY_NAME = process.platform === "win32" ? "llama-server.exe" : "llama-server";
const SENTENCES = [
    "The kettle boiled while rain tapped against the kitchen window.",
    "Steam curled off the cup and fogged the glass.",
    "The central bank raised interest rates by half a point.",
];

function realHost(dataDir: string): PluginServerHost {
    return {
        pluginId: "smiley-embeddings-memory",
        dataDir,
        logger: {
            debug: () => {},
            info: (message) => console.log(`    [host] ${message}`),
            warn: (message) => console.log(`    [host] warn: ${message}`),
            error: (message) => console.log(`    [host] error: ${message}`),
        },
        spawn(command, args = [], options = {}) {
            const child = Bun.spawn([command, ...args], {
                cwd: options.cwd,
                // The host merges the extra env over its own; the manager only adds a library path.
                env: options.env === undefined ? undefined : { ...process.env, ...options.env },
                stdout: "ignore",
                stderr: "ignore",
            });

            return {
                pid: child.pid,
                exited: child.exited,
                kill: async () => {
                    child.kill();
                    await child.exited;
                },
            };
        },
    };
}

function cosine(left: number[], right: number[]): number {
    let dot = 0;
    let leftNorm = 0;
    let rightNorm = 0;

    for (let index = 0; index < left.length; index += 1) {
        dot += left[index] * right[index];
        leftNorm += left[index] * left[index];
        rightNorm += right[index] * right[index];
    }

    return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm) || 1);
}

function isAlive(pid: number): boolean {
    try {
        process.kill(pid, 0);

        return true;
    } catch {
        return false;
    }
}

function findBinary(backend: string): string | undefined {
    // The release folder and the `cudart-*` runtime folder both contain "cuda", and only one of
    // them holds the executable.
    for (const folder of readdirSync(binRoot).filter((name) => name.includes(backend))) {
        const candidate = join(binRoot, folder, BINARY_NAME);

        if (existsSync(candidate)) {
            return candidate;
        }
    }

    return undefined;
}

const dataDir = dataDirArg ?? mkdtempSync(join(tmpdir(), "smoke-llama-"));

for (const [backend, gpuLayers, device] of [
    ["cuda", 99, "gpu"],
    ["vulkan", 99, "gpu"],
    ["cpu", 0, "cpu"],
] as [string, number, "gpu" | "cpu"][]) {
    const binary = findBinary(backend);

    if (binary === undefined) {
        console.log(`\n=== ${backend}: no build under ${binRoot}, skipped`);
        continue;
    }

    const candidate: LaunchSpec = {
        binary,
        model: modelPath,
        backend: backend as LaunchSpec["backend"],
        gpuLayers,
        pooling: "mean",
        ctx: 4096,
    };

    console.log(`\n=== ${backend}: ${basename(binary)} (-ngl ${gpuLayers})`);
    const manager = new LlamaProcess(realHost(dataDir), { startTimeoutMs: 180_000 });

    try {
        await manager.ensureReady([candidate], device, 0);

        const status = manager.status();
        console.log(
            `  status: state=${status.state} backend=${status.backend} device=${status.device ?? "-"} layers=${
                status.layersOffloaded ?? "?"
            }/${status.layersTotal ?? "?"} pid=${status.pid} port=${status.port}`,
        );

        const started = performance.now();
        const vectors = await manager.embed(SENTENCES);
        const elapsed = Math.round(performance.now() - started);
        console.log(`  embed: ${vectors.length} texts, dim ${vectors[0].length}, ${elapsed} ms (cold, includes warm-up)`);

        const again = await manager.embed(SENTENCES);
        const warm = Math.round(performance.now() - started) - elapsed;
        console.log(`  embed: warm ${warm} ms -> ${Math.round((SENTENCES.length / warm) * 1000)} emb/s`);

        console.log(`  cos(similar)   = ${cosine(vectors[0], vectors[1]).toFixed(3)}`);
        console.log(`  cos(unrelated) = ${cosine(vectors[0], vectors[2]).toFixed(3)}`);
        // llama.cpp's GPU kernels are not bit-exact across batch layouts, so the repeat is checked
        // by cosine (measured 0.99995 for the same text in a different batch) and not by equality.
        console.log(`  repeat cos     = ${cosine(vectors[0], again[0]).toFixed(5)}`);

        const pid = status.pid as number;
        await manager.stop();
        await Bun.sleep(250);
        console.log(`  stopped: process ${pid} alive=${isAlive(pid)}`);
    } catch (error) {
        console.log(`  failed: ${(error as Error).message}`);
    } finally {
        await manager.dispose();
    }
}

if (dataDirArg === undefined) {
    rmSync(dataDir, { recursive: true, force: true });
}

console.log("\ndone");

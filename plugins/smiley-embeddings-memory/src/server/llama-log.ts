// Reads which device llama-server actually used out of its own log file.
//
// llama.cpp only prints device and offload lines at `-lv 4`, so the process is started with
// `--log-file <dataDir>/llama-server.log -lv 4` and this parser is fed that file afterwards.
// The interesting part is that a GPU build *lists* the GPU even when it computes on the CPU:
// `-ngl 0` still logs `using device CUDA0` and only the offload count tells the truth.

import type { Backend } from "./releases";

export type LlamaLogInfo = { backend?: Backend; device?: string; layersOffloaded?: number; layersTotal?: number };

const GPU_DEVICE = "(CUDA\\d+|Vulkan\\d+|MTL\\d+)";

// Weights placed on a device is the strongest signal: it only appears when tensors went there.
const MODEL_BUFFER = new RegExp(`load_tensors:\\s+${GPU_DEVICE}\\s+model buffer size`);
const USING_DEVICE = new RegExp(`using device\\s+${GPU_DEVICE}`);
// The device_info block lists devices as `- NAME : ...`; the online (non-anchored) match is
// needed because llama.cpp prefixes every line with a timestamp, and the memory breakdown table
// (`| - CUDA0 (RTX ...) |`) is excluded by requiring the colon.
const LISTED_DEVICE = new RegExp(`(?:^|\\s)-\\s*${GPU_DEVICE}\\s*:`);
const OFFLOADED = /offloaded\s+(\d+)\s*\/\s*(\d+)\s+layers/g;

function backendOfDevice(device: string): Backend | undefined {
    if (device.startsWith("CUDA")) {
        return "cuda";
    }

    if (device.startsWith("Vulkan")) {
        return "vulkan";
    }

    if (device.startsWith("MTL")) {
        return "metal";
    }

    return undefined;
}

function lastMatch(text: string, pattern: RegExp): RegExpExecArray | undefined {
    const global = new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`);
    let found: RegExpExecArray | undefined;
    let match = global.exec(text);

    while (match !== null) {
        found = match;

        if (match[0].length === 0) {
            global.lastIndex += 1;
        }

        match = global.exec(text);
    }

    return found;
}

function detectDevice(text: string): string | undefined {
    const strong = lastMatch(text, MODEL_BUFFER) ?? lastMatch(text, USING_DEVICE);

    if (strong) {
        return strong[1];
    }

    // Fallback: the `device_info:` list. It is also printed by CPU-only runs, so this only
    // names the device; the offload count below still decides the backend.
    for (const line of text.split(/\r?\n/)) {
        const match = LISTED_DEVICE.exec(line);

        if (match) {
            return match[1];
        }
    }

    return undefined;
}

export function parseLlamaLog(text: string): LlamaLogInfo {
    // A restart appends to the same log file, so the last offload line is the current run.
    const offload = lastMatch(text, OFFLOADED);
    const device = detectDevice(text);
    const info: LlamaLogInfo = {};

    if (offload) {
        info.layersOffloaded = Number(offload[1]);
        info.layersTotal = Number(offload[2]);
    }

    if (device) {
        info.device = device;
    }

    if (offload && info.layersOffloaded === 0) {
        // The build knows the GPU but nothing was offloaded: this is a CPU run (`-ngl 0`), even
        // though the log names CUDA0/Vulkan0.
        info.backend = "cpu";

        return info;
    }

    const deviceBackend = device ? backendOfDevice(device) : undefined;

    if (deviceBackend) {
        info.backend = (info.layersOffloaded ?? 0) > 0 ? deviceBackend : undefined;

        return info;
    }

    if ((info.layersOffloaded ?? 0) > 0 && /Metal/i.test(text)) {
        // Metal builds sometimes print ggml_metal lines without an MTL device id.
        info.backend = "metal";
    }

    return info;
}

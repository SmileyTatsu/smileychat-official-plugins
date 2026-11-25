import { describe, expect, test } from "bun:test";

import { parseLlamaLog } from "./llama-log";

// Trimmed real output of llama-server b12686 with `-lv 4` (Windows, RTX 5060 Ti). The timestamps
// are the ones llama.cpp writes in front of every line, so the parser is exercised on the real
// shape instead of a cleaned-up one.
const CUDA_LOG = [
    "0.00.135.724 I cmn  common_param: common_params_print_info: verbosity = 4 (adjust with the `-lv N` CLI arg)",
    "0.00.135.727 I cmn  common_param: device_info:",
    "0.00.209.149 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15174 MiB free)",
    "0.00.209.164 I cmn  common_param:   - CPU     : AMD Ryzen 7 5700X 8-Core Processor              (32681 MiB, 14159 MiB free)",
    "0.00.449.485 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 5060 Ti) (0000:0b:00.0) - 15174 MiB free",
    "0.00.634.122 I load_tensors: offloading output layer to GPU",
    "0.00.634.127 I load_tensors: offloading 27 repeating layers to GPU",
    "0.00.634.128 I load_tensors: offloaded 29/29 layers to GPU",
    "0.00.634.131 I load_tensors:   CPU_Mapped model buffer size =   157.37 MiB",
    "0.00.634.132 I load_tensors:        CUDA0 model buffer size =   603.87 MiB",
    "0.01.026.777 I llama_kv_cache:      CUDA0 KV buffer size =  3584.00 MiB",
].join("\r\n");

const VULKAN_LOG = [
    "0.00.645.005 I cmn  common_param: device_info:",
    "0.00.646.305 I cmn  common_param:   - Vulkan0 : NVIDIA GeForce RTX 5060 Ti (16266 MiB, 15492 MiB free)",
    "0.00.646.314 I cmn  common_param:   - CPU     : AMD Ryzen 7 5700X 8-Core Processor              (32681 MiB, 14557 MiB free)",
    "0.00.986.672 I llama_prepare_model_devices: using device Vulkan0 (NVIDIA GeForce RTX 5060 Ti) (0000:0b:00.0) - 15490 MiB free",
    "0.01.127.156 I load_tensors: offloading output layer to GPU",
    "0.01.127.163 I load_tensors: offloaded 29/29 layers to GPU",
    "0.01.127.168 I load_tensors:      Vulkan0 model buffer size =   603.87 MiB",
].join("\n");

// Same machine with `-ngl 0`: the CUDA build is still in use, so CUDA0 is listed and even picked
// as "using device", but no layer is offloaded and the weights stay in CPU_Mapped memory.
const CPU_LOG = [
    "0.00.130.386 I cmn  common_param: device_info:",
    "0.00.196.505 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15174 MiB free)",
    "0.00.196.587 I cmn  common_param:   - CPU     : AMD Ryzen 7 5700X 8-Core Processor              (32681 MiB, 14159 MiB free)",
    "0.00.424.117 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 5060 Ti) (0000:0b:00.0) - 15174 MiB free",
    "0.00.570.203 I load_tensors: offloading 0 repeating layers to GPU",
    "0.00.570.208 I load_tensors: offloaded 0/29 layers to GPU",
    "0.00.570.212 I load_tensors:   CPU_Mapped model buffer size =   603.87 MiB",
].join("\n");

// No Metal run was recorded (the spike machine is Windows), so this is the shape llama.cpp
// prints on Apple silicon: MTL0 in the device list, MTL0 model buffers, full offload.
const METAL_LOG = [
    "0.00.030.101 I cmn  common_param: device_info:",
    "0.00.030.505 I cmn  common_param:   - MTL0    : Apple M3 Pro (10922 MiB, 8000 MiB free)",
    "0.00.030.587 I cmn  common_param:   - CPU     : Apple M3 Pro",
    "0.00.120.117 I llama_prepare_model_devices: using device MTL0 (Apple M3 Pro)",
    "0.00.400.203 I load_tensors: offloading 29 repeating layers to GPU",
    "0.00.400.208 I load_tensors: offloaded 29/29 layers to GPU",
    "0.00.400.212 I load_tensors:        MTL0 model buffer size =   603.87 MiB",
].join("\n");

describe("llama log", () => {
    test("reads backend, device and layers from a CUDA run", () => {
        expect(parseLlamaLog(CUDA_LOG)).toEqual({
            backend: "cuda",
            device: "CUDA0",
            layersOffloaded: 29,
            layersTotal: 29,
        });
    });

    test("reads a Vulkan run", () => {
        expect(parseLlamaLog(VULKAN_LOG)).toEqual({
            backend: "vulkan",
            device: "Vulkan0",
            layersOffloaded: 29,
            layersTotal: 29,
        });
    });

    test("reads a Metal run", () => {
        expect(parseLlamaLog(METAL_LOG)).toEqual({
            backend: "metal",
            device: "MTL0",
            layersOffloaded: 29,
            layersTotal: 29,
        });
    });

    test("a GPU build with 0 offloaded layers is reported as cpu", () => {
        const info = parseLlamaLog(CPU_LOG);

        expect(info.backend).toBe("cpu");
        expect(info.layersOffloaded).toBe(0);
        expect(info.layersTotal).toBe(29);
    });

    test("a log without -lv 4 lines yields an empty result", () => {
        const quiet = [
            "0.00.135.683 I srv  llama_server: initializing ...",
            "0.00.135.723 I cmn  common_param: common_params_print_info: build 12686 (82dd97047)",
            "0.00.945.576 I cmn  common_init_: added </s> logit bias = -inf",
        ].join("\n");

        expect(parseLlamaLog(quiet)).toEqual({});
        expect(parseLlamaLog("")).toEqual({});
    });

    test("uses the last offload line when the log holds several runs", () => {
        const restarted = [
            "0.00.570.208 I load_tensors: offloaded 0/29 layers to GPU",
            "0.00.634.132 I load_tensors:        CUDA0 model buffer size =   603.87 MiB",
            "1.00.634.128 I load_tensors: offloaded 29/29 layers to GPU",
        ].join("\n");

        expect(parseLlamaLog(restarted)).toEqual({
            backend: "cuda",
            device: "CUDA0",
            layersOffloaded: 29,
            layersTotal: 29,
        });
    });

    test("falls back to the device_info list when no weights line is present", () => {
        const listed = [
            "0.00.645.005 I cmn  common_param: device_info:",
            "0.00.646.305 I cmn  common_param:   - Vulkan0 : NVIDIA GeForce RTX 5060 Ti (16266 MiB, 15492 MiB free)",
            "0.00.646.314 I cmn  common_param:   - CPU     : AMD Ryzen 7 5700X 8-Core Processor",
            "0.01.127.163 I load_tensors: offloaded 29/29 layers to GPU",
        ].join("\n");

        expect(parseLlamaLog(listed)).toEqual({
            backend: "vulkan",
            device: "Vulkan0",
            layersOffloaded: 29,
            layersTotal: 29,
        });
    });

    test("does not mistake the memory breakdown table for the device list", () => {
        const breakdown = [
            "0.00.386.100 I common_memory_breakdown_print: | memory breakdown [MiB]  | total    free    self   model   context   compute    unaccounted |",
            "0.00.386.108 I common_memory_breakdown_print: |   - CUDA0 (RTX 5060 Ti) | 16310 = 15174 + (4246 =   603 +    3584 +      58) +       -3109 |",
            "0.00.570.208 I load_tensors: offloaded 0/29 layers to GPU",
        ].join("\n");

        expect(parseLlamaLog(breakdown)).toEqual({ backend: "cpu", layersOffloaded: 0, layersTotal: 29 });
    });
});

import { describe, expect, test } from "bun:test";

import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
    parseNvidiaSmi,
    parseReleases,
    pickRelease,
    planBinaries,
    type BinaryPlan,
    type GpuInfo,
    type Platform,
    type Release,
} from "./releases";

// Real payload of `GET /releases?per_page=3` for b12686, trimmed to tag_name + asset fields.
// tsconfig has no `resolveJsonModule`, so the fixture is read instead of imported.
const FIXTURE = JSON.parse(
    readFileSync(join(import.meta.dir, "fixtures", "b12686-assets.json"), "utf8"),
) as unknown;

function fixtureRelease(): Release {
    const releases = parseReleases([FIXTURE]);

    expect(releases).toHaveLength(1);

    return releases[0];
}

function names(plan: BinaryPlan): string[] {
    return plan.assets.map((asset) => asset.name);
}

const WIN_X64: Platform = { os: "win32", arch: "x64" };
const NVIDIA_13: GpuInfo = { nvidia: true, cudaMajor: 13 };

describe("releases", () => {
    test("parseReleases keeps the real asset fields and drops malformed entries", () => {
        const releases = parseReleases([
            { tag_name: "b1", assets: [{ name: "a.zip", size: 10, browser_download_url: "https://x/a.zip" }] },
            { tag_name: "", assets: [] },
            { tag_name: "b2" },
            { tag_name: "b3", assets: [{ name: "b.zip", size: "10", browser_download_url: "https://x/b.zip" }] },
            { tag_name: "b4", assets: [{ size: 1, browser_download_url: "https://x/c.zip" }] },
            { tag_name: "b5", assets: [{ name: "d.zip", size: 1 }] },
            "nope",
            null,
        ]);

        expect(releases).toEqual([
            {
                tag: "b1",
                assets: [{ name: "a.zip", size: 10, url: "https://x/a.zip", digest: undefined }],
            },
            { tag: "b3", assets: [] },
            { tag: "b4", assets: [] },
            { tag: "b5", assets: [] },
        ]);
        expect(parseReleases({ tag_name: "b1", assets: [] })).toEqual([]);
    });

    test("parseReleases keeps digests when GitHub publishes them", () => {
        const [release] = parseReleases([FIXTURE]);

        expect(release.tag).toBe("b12686");
        expect(release.assets).toHaveLength(38);
        expect(release.assets[0].digest).toMatch(/^sha256:[0-9a-f]{64}$/);
        expect(release.assets[0].url).toStartWith("https://github.com/ggml-org/llama.cpp/releases/download/b12686/");
    });

    test("pickRelease matches an exact tag and ignores the v1.0.0 release for latest", () => {
        const releases = parseReleases([
            { tag_name: "v1.0.0", assets: [] },
            { tag_name: "b12686", assets: [{ name: "a.zip", size: 1, browser_download_url: "https://x/a.zip" }] },
            { tag_name: "b12682", assets: [{ name: "b.zip", size: 1, browser_download_url: "https://x/b.zip" }] },
            { tag_name: "b99999", assets: [] },
        ]);

        expect(pickRelease(releases, "b12682")?.tag).toBe("b12682");
        expect(pickRelease(releases, "latest")?.tag).toBe("b12686");
        expect(pickRelease(releases, "b1")).toBeUndefined();
    });

    test("planBinaries prefers cuda 13 with its runtime on a CUDA 13 driver", () => {
        const plans = planBinaries(fixtureRelease(), WIN_X64, NVIDIA_13, { device: "auto", gpuBackend: "auto" });

        expect(plans.map((plan) => plan.backend)).toEqual(["cuda", "vulkan", "cpu"]);
        expect(names(plans[0])).toEqual([
            "llama-b12686-bin-win-cuda-13.5-x64.zip",
            "cudart-llama-bin-win-cuda-13.5-x64.zip",
        ]);
        expect(names(plans[1])).toEqual(["llama-b12686-bin-win-vulkan-x64.zip"]);
        expect(names(plans[2])).toEqual(["llama-b12686-bin-win-cpu-x64.zip"]);
    });

    test("planBinaries falls back to cuda 12 on a CUDA 12 driver", () => {
        const plans = planBinaries(fixtureRelease(), WIN_X64, { nvidia: true, cudaMajor: 12 }, {
            device: "auto",
            gpuBackend: "auto",
        });

        expect(plans.map((plan) => plan.backend)).toEqual(["cuda", "vulkan", "cpu"]);
        expect(names(plans[0])).toEqual([
            "llama-b12686-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        ]);
    });

    test("planBinaries skips cuda without an NVIDIA GPU and keeps cpu last", () => {
        const plans = planBinaries(fixtureRelease(), WIN_X64, { nvidia: false }, { device: "auto", gpuBackend: "auto" });

        expect(plans.map((plan) => plan.backend)).toEqual(["vulkan", "cpu"]);
    });

    test("planBinaries honours the gpuBackend preference", () => {
        const vulkan = planBinaries(fixtureRelease(), WIN_X64, NVIDIA_13, { device: "gpu", gpuBackend: "vulkan" });
        const cuda = planBinaries(fixtureRelease(), WIN_X64, NVIDIA_13, { device: "gpu", gpuBackend: "cuda" });
        const none = planBinaries(fixtureRelease(), WIN_X64, { nvidia: false }, { device: "gpu", gpuBackend: "cuda" });

        expect(vulkan.map((plan) => plan.backend)).toEqual(["vulkan"]);
        expect(cuda.map((plan) => plan.backend)).toEqual(["cuda"]);
        expect(none).toEqual([]);
    });

    test("planBinaries device cpu returns only the cpu build", () => {
        const plans = planBinaries(fixtureRelease(), WIN_X64, NVIDIA_13, { device: "cpu", gpuBackend: "auto" });

        expect(plans.map((plan) => plan.backend)).toEqual(["cpu"]);
        expect(names(plans[0])).toEqual(["llama-b12686-bin-win-cpu-x64.zip"]);
    });

    test("planBinaries covers windows arm64 (cuda 13 only) and linux", () => {
        const winArm = planBinaries(fixtureRelease(), { os: "win32", arch: "arm64" }, NVIDIA_13, {
            device: "auto",
            gpuBackend: "auto",
        });
        const linuxX64 = planBinaries(fixtureRelease(), { os: "linux", arch: "x64" }, NVIDIA_13, {
            device: "auto",
            gpuBackend: "auto",
        });
        const linuxArm = planBinaries(fixtureRelease(), { os: "linux", arch: "arm64" }, NVIDIA_13, {
            device: "auto",
            gpuBackend: "auto",
        });

        expect(winArm.map((plan) => plan.backend)).toEqual(["cuda", "vulkan", "cpu"]);
        expect(names(winArm[0])).toEqual([
            "llama-b12686-bin-win-cuda-13.5-arm64.zip",
            "cudart-llama-bin-win-cuda-13.5-arm64.zip",
        ]);
        expect(names(winArm[2])).toEqual(["llama-b12686-bin-win-cpu-arm64.zip"]);

        expect(linuxX64.map((plan) => plan.backend)).toEqual(["cuda", "vulkan", "cpu"]);
        expect(names(linuxX64[0])).toEqual([
            "llama-b12686-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "cudart-llama-b12686-bin-ubuntu-cuda-13.5-x64.tar.gz",
        ]);
        expect(names(linuxX64[2])).toEqual(["llama-b12686-bin-ubuntu-x64.tar.gz"]);

        // Linux arm64 has no GPU build in b12686: auto must still offer the cpu archive.
        expect(linuxArm.map((plan) => plan.backend)).toEqual(["cpu"]);
        expect(names(linuxArm[0])).toEqual(["llama-b12686-bin-ubuntu-arm64.tar.gz"]);
    });

    test("planBinaries treats the macOS arm64 binary as metal for gpu and cpu devices", () => {
        const gpu = planBinaries(fixtureRelease(), { os: "darwin", arch: "arm64" }, { nvidia: false }, {
            device: "gpu",
            gpuBackend: "auto",
        });
        const auto = planBinaries(fixtureRelease(), { os: "darwin", arch: "arm64" }, { nvidia: false }, {
            device: "auto",
            gpuBackend: "auto",
        });
        const cpu = planBinaries(fixtureRelease(), { os: "darwin", arch: "arm64" }, { nvidia: false }, {
            device: "cpu",
            gpuBackend: "auto",
        });
        const intel = planBinaries(fixtureRelease(), { os: "darwin", arch: "x64" }, { nvidia: false }, {
            device: "auto",
            gpuBackend: "auto",
        });

        expect(names(gpu[0])).toEqual(["llama-b12686-bin-macos-arm64.tar.gz"]);
        expect(auto.map((plan) => plan.backend)).toEqual(["metal"]);
        expect(cpu.map((plan) => plan.backend)).toEqual(["metal"]);
        expect(intel.map((plan) => plan.backend)).toEqual(["cpu"]);
        expect(names(intel[0])).toEqual(["llama-b12686-bin-macos-x64.tar.gz"]);
    });

    test("planBinaries picks the plain android build, not the snapdragon one", () => {
        const plans = planBinaries(fixtureRelease(), { os: "android", arch: "arm64" }, { nvidia: false }, {
            device: "auto",
            gpuBackend: "auto",
        });

        expect(plans.map((plan) => plan.backend)).toEqual(["cpu"]);
        expect(names(plans[0])).toEqual(["llama-b12686-bin-android-arm64.tar.gz"]);
    });

    test("planBinaries keeps the newest cuda minor when a release ships several", () => {
        const release: Release = {
            tag: "b1",
            assets: ["13.1", "13.5", "13.10"].map((version) => ({
                name: `llama-b1-bin-win-cuda-${version}-x64.zip`,
                size: 1,
                url: `https://example/${version}`,
            })),
        };
        const plans = planBinaries(release, WIN_X64, NVIDIA_13, { device: "gpu", gpuBackend: "cuda" });

        expect(names(plans[0])).toEqual(["llama-b1-bin-win-cuda-13.10-x64.zip"]);
    });

    test("parseNvidiaSmi reads the old `CUDA Version:` header", () => {
        const header = [
            "Mon Nov 24 10:00:00 2026",
            "+-----------------------------------------------------------------------------------------+",
            "| NVIDIA-SMI 551.61                 Driver Version: 551.61         CUDA Version: 12.4     |",
            "|-----------------------------------------+------------------------+----------------------|",
        ].join("\n");

        expect(parseNvidiaSmi(header)).toEqual({ nvidia: true, cudaMajor: 12 });
        expect(parseNvidiaSmi("CUDA Version: 13")).toEqual({ nvidia: true, cudaMajor: 13 });
    });

    test("parseNvidiaSmi reads the new `CUDA UMD Version:` header", () => {
        // Driver 610.x renamed the field; without this the machine fell back to Vulkan.
        const header = [
            "Tue Nov 24 10:00:00 2026",
            "+-----------------------------------------------------------------------------------------+",
            "| NVIDIA-SMI 610.88                 KMD Version: 610.88        CUDA UMD Version: 13.3     |",
            "|-----------------------------------------+------------------------+----------------------|",
        ].join("\n");

        expect(parseNvidiaSmi(header)).toEqual({ nvidia: true, cudaMajor: 13 });
    });

    test("parseNvidiaSmi reports the GPU when the header has no CUDA version", () => {
        const header = [
            "+-----------------------------------------------------------------------------------------+",
            "| NVIDIA-SMI 610.88                 KMD Version: 610.88                                   |",
            "|-----------------------------------------+------------------------+----------------------|",
        ].join("\n");
        const info = parseNvidiaSmi(header);

        expect(info.nvidia).toBe(true);
        expect(info.cudaMajor).toBeUndefined();
    });

    test("parseNvidiaSmi stays negative without the nvidia-smi banner", () => {
        expect(parseNvidiaSmi("'nvidia-smi' is not recognized as an internal or external command")).toEqual({
            nvidia: false,
        });
        expect(parseNvidiaSmi("nvidia-smi: command not found")).toEqual({ nvidia: false });
        expect(parseNvidiaSmi("segmentation fault")).toEqual({ nvidia: false });
        expect(parseNvidiaSmi("")).toEqual({ nvidia: false });
    });

    test("planBinaries offers the newest CUDA build when the driver version is unknown", () => {
        const unknown: GpuInfo = { nvidia: true };
        const auto = planBinaries(fixtureRelease(), WIN_X64, unknown, { device: "auto", gpuBackend: "auto" });
        const gpu = planBinaries(fixtureRelease(), WIN_X64, unknown, { device: "gpu", gpuBackend: "auto" });
        const linux = planBinaries(fixtureRelease(), { os: "linux", arch: "x64" }, unknown, {
            device: "auto",
            gpuBackend: "auto",
        });
        const vulkanOnly = planBinaries(fixtureRelease(), WIN_X64, unknown, { device: "gpu", gpuBackend: "vulkan" });

        expect(auto.map((plan) => plan.backend)).toEqual(["cuda", "vulkan", "cpu"]);
        expect(names(auto[0])).toEqual([
            "llama-b12686-bin-win-cuda-13.5-x64.zip",
            "cudart-llama-bin-win-cuda-13.5-x64.zip",
        ]);
        expect(gpu.map((plan) => plan.backend)).toEqual(["cuda", "vulkan"]);
        expect(names(linux[0])).toEqual([
            "llama-b12686-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "cudart-llama-b12686-bin-ubuntu-cuda-13.5-x64.tar.gz",
        ]);
        expect(vulkanOnly.map((plan) => plan.backend)).toEqual(["vulkan"]);
    });

    test("planBinaries takes the newest build a release actually ships when the version is unknown", () => {
        const release: Release = {
            tag: "b1",
            assets: ["12.4", "12.8"].map((version) => ({
                name: `llama-b1-bin-win-cuda-${version}-x64.zip`,
                size: 1,
                url: `https://example/${version}`,
            })),
        };
        const plans = planBinaries(release, WIN_X64, { nvidia: true }, { device: "gpu", gpuBackend: "cuda" });

        expect(names(plans[0])).toEqual(["llama-b1-bin-win-cuda-12.8-x64.zip"]);
    });
});

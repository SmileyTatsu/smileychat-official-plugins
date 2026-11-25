// Picks the llama.cpp release assets to download for a platform and GPU.
//
// Pure: no network, no disk. The GitHub release payload is passed in as parsed JSON so the
// selection can be unit-tested against a real b12686 asset list instead of hand-made names.

export type Backend = "cuda" | "vulkan" | "cpu" | "metal";

export type Platform = { os: "win32" | "linux" | "darwin" | "android"; arch: "x64" | "arm64" };

export type GpuInfo = { nvidia: boolean; cudaMajor?: number };

export type ReleaseAsset = { name: string; size: number; url: string; digest?: string };

export type Release = { tag: string; assets: ReleaseAsset[] };

export type BinaryPlan = { backend: Backend; assets: ReleaseAsset[] };

// llama.cpp publishes Windows builds as .zip and the Linux/macOS/Android ones as .tar.gz
// (it changed format more than once), so every pattern accepts both. The plan only returns
// the asset; the caller decides how to unpack it.
const ARCHIVE_EXT = "(?:zip|tar\\.gz)";

function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function assetPattern(tag: string, middle: string, arch: string): RegExp {
    return new RegExp(`^llama-${escapeRegExp(tag)}-bin-${middle}-${arch}\\.${ARCHIVE_EXT}$`);
}

function firstAsset(release: Release, pattern: RegExp): ReleaseAsset | undefined {
    return release.assets.find((asset) => pattern.test(asset.name));
}

function compareVersions(left: string, right: string): number {
    const leftParts = left.split(".").map(Number);
    const rightParts = right.split(".").map(Number);
    const length = Math.max(leftParts.length, rightParts.length);

    for (let index = 0; index < length; index += 1) {
        const diff = (leftParts[index] ?? 0) - (rightParts[index] ?? 0);
        if (diff !== 0) {
            return diff;
        }
    }

    return 0;
}

export function parseReleases(json: unknown): Release[] {
    if (!Array.isArray(json)) {
        return [];
    }

    const releases: Release[] = [];

    for (const entry of json) {
        if (!entry || typeof entry !== "object") {
            continue;
        }

        const raw = entry as Record<string, unknown>;
        const tag = typeof raw.tag_name === "string" ? raw.tag_name.trim() : "";

        if (!tag || !Array.isArray(raw.assets)) {
            continue;
        }

        const assets: ReleaseAsset[] = [];

        for (const item of raw.assets) {
            if (!item || typeof item !== "object") {
                continue;
            }

            const rawAsset = item as Record<string, unknown>;
            const name = typeof rawAsset.name === "string" ? rawAsset.name : "";
            const url = typeof rawAsset.browser_download_url === "string" ? rawAsset.browser_download_url : "";
            const size = rawAsset.size;

            if (!name || !url || typeof size !== "number" || !Number.isFinite(size) || size < 0) {
                continue;
            }

            const digest = typeof rawAsset.digest === "string" && rawAsset.digest ? rawAsset.digest : undefined;
            assets.push({ name, size, url, digest });
        }

        releases.push({ tag, assets });
    }

    return releases;
}

export function pickRelease(releases: Release[], tag: string): Release | undefined {
    if (tag === "latest") {
        // `/releases/latest` returns the v1.0.0 tag without binaries, and the release list is
        // not sorted by build number, so "latest" means the highest b#### that has assets.
        let best: { release: Release; build: number } | undefined;

        for (const release of releases) {
            const match = /^b(\d+)$/.exec(release.tag);

            if (!match || release.assets.length === 0) {
                continue;
            }

            const build = Number(match[1]);

            if (!best || build > best.build) {
                best = { release, build };
            }
        }

        return best?.release;
    }

    return releases.find((release) => release.tag === tag);
}

type CudaPick = { asset: ReleaseAsset; version: string };

function pickCudaBuild(
    release: Release,
    osPrefix: "win" | "ubuntu",
    arch: Platform["arch"],
    major: number,
): CudaPick | undefined {
    const pattern = assetPattern(release.tag, `${osPrefix}-cuda-(\\d+(?:\\.\\d+)*)`, arch);
    let best: CudaPick | undefined;

    for (const asset of release.assets) {
        const match = pattern.exec(asset.name);

        if (!match) {
            continue;
        }

        const version = match[1];

        if (Number.parseInt(version.split(".")[0], 10) !== major) {
            continue;
        }

        // Several minors of the same major can be published at once (13.1 and 13.5); the
        // newest one is the one built against the current toolkit.
        if (!best || compareVersions(version, best.version) > 0) {
            best = { asset, version };
        }
    }

    return best;
}

function pickCudart(
    release: Release,
    osPrefix: "win" | "ubuntu",
    arch: Platform["arch"],
    version: string,
): ReleaseAsset | undefined {
    // The tag is optional in these names: b12686 ships `cudart-llama-bin-win-cuda-13.5-x64.zip`
    // without it but `cudart-llama-b12686-bin-ubuntu-cuda-13.5-x64.tar.gz` with it.
    const tagPrefix = `(?:${escapeRegExp(release.tag)}-)?`;

    const exact = firstAsset(
        release,
        new RegExp(`^cudart-llama-${tagPrefix}bin-${osPrefix}-cuda-${escapeRegExp(version)}-${arch}\\.${ARCHIVE_EXT}$`),
    );

    if (exact) {
        return exact;
    }

    // The runtime is republished less often than the binaries: fall back to another minor of
    // the same major instead of failing a 620 MB install over a metadata mismatch.
    const pattern = new RegExp(
        `^cudart-llama-${tagPrefix}bin-${osPrefix}-cuda-${version.split(".")[0]}\\.\\d+-${arch}\\.${ARCHIVE_EXT}$`,
    );

    return firstAsset(release, pattern);
}

function cudaPlan(
    release: Release,
    osPrefix: "win" | "ubuntu",
    arch: Platform["arch"],
    major: number,
): BinaryPlan | undefined {
    const cuda = pickCudaBuild(release, osPrefix, arch, major);

    if (!cuda) {
        return undefined;
    }

    const assets = [cuda.asset];
    const runtime = pickCudart(release, osPrefix, arch, cuda.version);

    if (runtime) {
        assets.push(runtime);
    }

    return { backend: "cuda", assets };
}

// Which CUDA majors to try, in order. A 12.x driver cannot run a 13.x build, but a 13.x driver
// can still run a 12.x one, so a missing 13 build in a release is not a dead end.
function cudaMajorsFor(platform: Platform, gpu: GpuInfo): number[] {
    if (!gpu.nvidia || gpu.cudaMajor === undefined) {
        return [];
    }

    if (gpu.cudaMajor >= 13) {
        // Windows/linux arm64 CUDA builds only exist for 13.x.
        return platform.arch === "arm64" ? [13] : [13, 12];
    }

    return gpu.cudaMajor === 12 ? [12] : [];
}

function gpuPlans(
    release: Release,
    platform: Platform,
    gpu: GpuInfo,
    gpuBackend: "auto" | "cuda" | "vulkan",
): BinaryPlan[] {
    const plans: BinaryPlan[] = [];
    const wantCuda = gpuBackend !== "vulkan";
    const wantVulkan = gpuBackend !== "cuda";

    if (platform.os === "win32") {
        if (wantCuda) {
            for (const major of cudaMajorsFor(platform, gpu)) {
                const plan = cudaPlan(release, "win", platform.arch, major);

                if (plan) {
                    plans.push(plan);
                    break;
                }
            }
        }

        if (wantVulkan) {
            const asset = firstAsset(release, assetPattern(release.tag, "win-vulkan", platform.arch));

            if (asset) {
                plans.push({ backend: "vulkan", assets: [asset] });
            }
        }

        return plans;
    }

    if (platform.os === "linux") {
        // Linux arm64 ships only the plain CPU build (no CUDA/Vulkan arm64 assets in b12686).
        if (platform.arch === "arm64") {
            return [];
        }

        if (wantCuda) {
            for (const major of cudaMajorsFor(platform, gpu)) {
                const plan = cudaPlan(release, "ubuntu", platform.arch, major);

                if (plan) {
                    plans.push(plan);
                    break;
                }
            }
        }

        if (wantVulkan) {
            const asset = firstAsset(release, assetPattern(release.tag, "ubuntu-vulkan", platform.arch));

            if (asset) {
                plans.push({ backend: "vulkan", assets: [asset] });
            }
        }

        return plans;
    }

    if (platform.os === "darwin") {
        // Apple silicon always computes on Metal through the one universal binary; there is no
        // separate GPU asset, and a user forcing cuda/vulkan on macOS has nothing to download.
        if (platform.arch !== "arm64" || gpuBackend !== "auto") {
            return [];
        }

        const asset = firstAsset(release, assetPattern(release.tag, "macos", "arm64"));

        return asset ? [{ backend: "metal", assets: [asset] }] : [];
    }

    // Android has no GPU builds; Termux users configure a manual binary path instead.
    return [];
}

function cpuPlan(release: Release, platform: Platform): BinaryPlan | undefined {
    if (platform.os === "win32") {
        const asset = firstAsset(release, assetPattern(release.tag, "win-cpu", platform.arch));

        return asset ? { backend: "cpu", assets: [asset] } : undefined;
    }

    if (platform.os === "linux") {
        const asset = firstAsset(release, assetPattern(release.tag, "ubuntu", platform.arch));

        return asset ? { backend: "cpu", assets: [asset] } : undefined;
    }

    if (platform.os === "darwin") {
        const asset = firstAsset(release, assetPattern(release.tag, "macos", platform.arch));

        // The macOS arm64 binary is the Metal build; the caller passes `-ngl 0` for device cpu.
        return asset ? { backend: platform.arch === "arm64" ? "metal" : "cpu", assets: [asset] } : undefined;
    }

    if (platform.arch === "arm64") {
        const asset = firstAsset(release, assetPattern(release.tag, "android", "arm64"));

        return asset ? { backend: "cpu", assets: [asset] } : undefined;
    }

    return undefined;
}

export function planBinaries(
    release: Release,
    platform: Platform,
    gpu: GpuInfo,
    opts: { device: "auto" | "gpu" | "cpu"; gpuBackend: "auto" | "cuda" | "vulkan" },
): BinaryPlan[] {
    const cpu = cpuPlan(release, platform);

    if (opts.device === "cpu") {
        return cpu ? [cpu] : [];
    }

    const candidates = gpuPlans(release, platform, gpu, opts.gpuBackend);

    if (opts.device === "gpu") {
        // No silent CPU fallback: the caller reports that no GPU build fits and offers `cpu`.
        return candidates;
    }

    if (!cpu) {
        return candidates;
    }

    // macOS uses the same binary for Metal and for `-ngl 0`, so do not offer it twice.
    const used = new Set(candidates.map((plan) => plan.assets[0]?.name));

    return used.has(cpu.assets[0].name) ? candidates : [...candidates, cpu];
}

export function parseNvidiaSmi(stdout: string): GpuInfo {
    const match = /CUDA Version:\s*(\d+)(?:\.(\d+))?/.exec(stdout);

    if (!match) {
        // No header, no driver, or an nvidia-smi too old to report CUDA: treat as no NVIDIA GPU.
        return { nvidia: false };
    }

    return { nvidia: true, cudaMajor: Number(match[1]) };
}

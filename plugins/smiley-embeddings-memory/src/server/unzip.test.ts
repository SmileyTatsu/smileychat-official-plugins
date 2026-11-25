import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import { mkdirSync } from "node:fs";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { zipSync, type Zippable } from "fflate";

import { extractZip } from "./unzip";

let scratch = "";
let counter = 0;

beforeAll(async () => {
    scratch = await mkdtemp(join(tmpdir(), "smiley-unzip-"));
});

afterAll(async () => {
    await rm(scratch, { recursive: true, force: true });
});

function tempDir(name: string): string {
    counter += 1;

    const dir = join(scratch, `${counter}-${name}`);
    mkdirSync(dir, { recursive: true });

    return dir;
}

async function writeZip(dir: string, entries: Zippable): Promise<string> {
    const path = join(dir, "archive.zip");
    await writeFile(path, zipSync(entries));

    return path;
}

function decode(bytes: Uint8Array): string {
    return new TextDecoder().decode(bytes);
}

async function exists(path: string): Promise<boolean> {
    try {
        await stat(path);

        return true;
    } catch {
        return false;
    }
}

describe("extractZip", () => {
    test("writes nested entries and returns their relative paths", async () => {
        const dir = tempDir("nested");
        const zip = await writeZip(dir, {
            "llama-server.exe": new TextEncoder().encode("binary"),
            "runtimes/cuda/bin/cublas64_13.dll": new TextEncoder().encode("dll"),
            "runtimes/cuda/README.md": new TextEncoder().encode("readme"),
            "empty.txt": new Uint8Array(0),
        });

        const written = await extractZip(zip, join(dir, "out"));

        expect(written.sort()).toEqual([
            "empty.txt",
            "llama-server.exe",
            "runtimes/cuda/README.md",
            "runtimes/cuda/bin/cublas64_13.dll",
        ]);
        expect(decode(await readFile(join(dir, "out", "runtimes/cuda/README.md")))).toBe("readme");
        expect(decode(await readFile(join(dir, "out", "llama-server.exe")))).toBe("binary");
        expect((await stat(join(dir, "out", "empty.txt"))).size).toBe(0);
    });

    test("skips directory entries", async () => {
        const dir = tempDir("dirs");
        const zip = await writeZip(dir, {
            "bin/": new Uint8Array(0),
            "bin/tool": new TextEncoder().encode("tool"),
            "models/": new Uint8Array(0),
        });

        const written = await extractZip(zip, join(dir, "out"));

        expect(written).toEqual(["bin/tool"]);
        expect(decode(await readFile(join(dir, "out", "bin/tool")))).toBe("tool");
    });

    test("returns an empty list for an empty archive", async () => {
        const dir = tempDir("empty");
        const zip = await writeZip(dir, {});

        expect(await extractZip(zip, join(dir, "out"))).toEqual([]);
    });

    test("rejects every zip-slip shape", async () => {
        const variants = ["../evil.txt", "/abs.txt", "C:/x.txt", "a/../../b.txt", "..\\evil.txt", "a/..\\..\\evil.txt"];

        for (const name of variants) {
            const dir = tempDir("slip");
            const zip = await writeZip(dir, { [name]: new TextEncoder().encode("evil") });

            await expect(extractZip(zip, join(dir, "out"))).rejects.toThrow(/refusing/);
            expect(await exists(join(dir, "evil.txt"))).toBe(false);
            expect(await exists(join(dir, "out", "evil.txt"))).toBe(false);
        }
    });

    test("normalises backslashes on a legitimately nested entry", async () => {
        const dir = tempDir("backslash");
        const zip = await writeZip(dir, { "bin\\sub\\file.txt": new TextEncoder().encode("nested") });

        const written = await extractZip(zip, join(dir, "out"));

        expect(written).toEqual(["bin/sub/file.txt"]);
        expect(decode(await readFile(join(dir, "out", "bin/sub/file.txt")))).toBe("nested");
    });

    test("enforces the file-count cap", async () => {
        const dir = tempDir("count");
        const zip = await writeZip(dir, {
            "a.txt": new Uint8Array(1),
            "b.txt": new Uint8Array(1),
            "c.txt": new Uint8Array(1),
        });

        await expect(extractZip(zip, join(dir, "out"), { maxFiles: 2 })).rejects.toThrow(/more than 2 files/);
    });

    test("enforces the per-file cap while streaming and removes the partial file", async () => {
        const dir = tempDir("filecap");
        const big = new Uint8Array(200_000);
        big.fill(7);
        const zip = await writeZip(dir, { "big.bin": [big, { level: 0 }] });

        await expect(extractZip(zip, join(dir, "out"), { maxFileBytes: 50_000 })).rejects.toThrow(/byte limit/);
        expect(await exists(join(dir, "out", "big.bin"))).toBe(false);
    });

    test("enforces the total-size cap across entries", async () => {
        const dir = tempDir("totalcap");
        const chunk = new Uint8Array(100_000);
        chunk.fill(3);
        const zip = await writeZip(dir, {
            "one.bin": [chunk, { level: 0 }],
            "two.bin": [chunk, { level: 0 }],
        });

        await expect(extractZip(zip, join(dir, "out"), { maxTotalBytes: 150_000 })).rejects.toThrow(/archive is over/);
    });

    test("round-trips a 60 MB entry", async () => {
        const dir = tempDir("large");
        const total = 60 * 1024 * 1024;
        const big = new Uint8Array(total);

        for (let index = 0; index < 1024; index += 1) {
            big[index] = index % 256;
        }

        big.fill(42, 1024);

        // Stored (level 0) so the test measures the streaming path, not the inflate speed.
        const zip = await writeZip(dir, { "big.bin": [big, { level: 0 }] });
        const written = await extractZip(zip, join(dir, "out"));
        const extracted = await readFile(join(dir, "out", "big.bin"));

        expect(written).toEqual(["big.bin"]);
        expect(extracted.length).toBe(total);
        expect(extracted[0]).toBe(0);
        expect(extracted[1023]).toBe(1023 % 256);
        expect(extracted[1024]).toBe(42);

        const hash = new Bun.CryptoHasher("sha256").update(big).digest("hex");

        expect(new Bun.CryptoHasher("sha256").update(extracted).digest("hex")).toBe(hash);
    }, 60_000);

    test("fails cleanly when an entry cannot be written (no hang, no crash)", async () => {
        const dir = tempDir("writefail");
        const zip = await writeZip(dir, {
            "blocked.bin": new TextEncoder().encode("data"),
            "ok.txt": new TextEncoder().encode("ok"),
        });
        const out = join(dir, "out");

        // A directory occupying the entry's path makes createWriteStream fail: the error arrives
        // as an 'error' event while the feed loop is awaiting the next chunk, which is exactly the
        // case that used to hang or crash.
        mkdirSync(join(out, "blocked.bin"), { recursive: true });

        await expect(extractZip(zip, out)).rejects.toThrow(/E[A-Z]+/);

        // The process is still usable afterwards.
        const second = await writeZip(tempDir("writefail-ok"), { "fine.txt": new TextEncoder().encode("fine") });

        expect(await extractZip(second, join(dir, "out2"))).toEqual(["fine.txt"]);
    });

    test("makes the llama binaries executable on non-Windows hosts", async () => {
        const dir = tempDir("mode");

        if (process.platform === "win32") {
            // Windows has no mode bits to check; the extractor must simply not fail there.
            const zip = await writeZip(dir, { "llama-server": new TextEncoder().encode("x") });

            expect(await extractZip(zip, join(dir, "out"))).toEqual(["llama-server"]);

            return;
        }

        const zip = await writeZip(dir, { "llama-server": new TextEncoder().encode("x"), "readme.md": new Uint8Array(1) });

        await extractZip(zip, join(dir, "out"));

        expect((await stat(join(dir, "out", "llama-server"))).mode & 0o777).toBe(0o755);
        expect((await stat(join(dir, "out", "readme.md"))).mode & 0o777).not.toBe(0o755);
    });
});

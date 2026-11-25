import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import { lstatSync, mkdirSync, readFileSync, readlinkSync, statSync } from "node:fs";
import { mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { gzipSync } from "fflate";

import { extractTarGz } from "./untar";

let scratch = "";
let counter = 0;

beforeAll(async () => {
    scratch = await mkdtemp(join(tmpdir(), "smiley-untar-"));
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

function encode(text: string): Uint8Array {
    return new TextEncoder().encode(text);
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

// --- tiny tar writer ------------------------------------------------------

function writeText(buffer: Uint8Array, offset: number, length: number, value: string): void {
    const bytes = encode(value).subarray(0, length);

    buffer.set(bytes, offset);
}

function writeOctal(buffer: Uint8Array, offset: number, length: number, value: number): void {
    writeText(buffer, offset, length - 1, value.toString(8).padStart(length - 1, "0"));
    buffer[offset + length - 1] = 0;
}

// GNU base-256: high bit set on the first byte, the rest big-endian. Used for sizes that do not
// fit in the 11 octal digits.
function writeBase256(buffer: Uint8Array, offset: number, length: number, value: number): void {
    buffer[offset] = 0x80;

    let rest = value;

    for (let index = length - 1; index >= 1; index -= 1) {
        buffer[offset + index] = rest & 0xff;
        rest = Math.floor(rest / 256);
    }
}

type HeaderOptions = { mode?: number; linkname?: string; base256Size?: boolean };

function tarHeader(name: string, size: number, typeflag: string, opts: HeaderOptions = {}): Uint8Array {
    const header = new Uint8Array(512);
    const mode = opts.mode ?? (typeflag === "5" ? 0o755 : 0o644);

    writeText(header, 0, 100, name);
    writeOctal(header, 100, 8, mode);
    writeOctal(header, 108, 8, 0);
    writeOctal(header, 116, 8, 0);

    if (opts.base256Size) {
        writeBase256(header, 124, 12, size);
    } else {
        writeOctal(header, 124, 12, size);
    }

    writeOctal(header, 136, 12, 0);
    header.fill(0x20, 148, 156); // checksum field is spaces while the sum is computed
    header[156] = typeflag.charCodeAt(0);
    writeText(header, 157, 100, opts.linkname ?? "");
    writeText(header, 257, 6, "ustar");
    writeText(header, 263, 2, "00");
    writeText(header, 265, 32, "smiley");
    writeText(header, 297, 32, "smiley");

    let checksum = 0;

    for (const byte of header) {
        checksum += byte;
    }

    writeText(header, 148, 6, checksum.toString(8).padStart(6, "0"));
    header[154] = 0;
    header[155] = 0x20;

    return header;
}

function padToBlock(data: Uint8Array): Uint8Array {
    const pad = (512 - (data.length % 512)) % 512;

    if (pad === 0) {
        return data;
    }

    const out = new Uint8Array(data.length + pad);

    out.set(data);

    return out;
}

function concat(parts: Uint8Array[]): Uint8Array {
    const length = parts.reduce((sum, part) => sum + part.length, 0);
    const out = new Uint8Array(length);
    let offset = 0;

    for (const part of parts) {
        out.set(part, offset);
        offset += part.length;
    }

    return out;
}

function buildTar(parts: Uint8Array[]): Uint8Array {
    return concat([...parts, new Uint8Array(1024)]);
}

function fileEntry(name: string, data: Uint8Array, opts: HeaderOptions = {}): Uint8Array[] {
    return [tarHeader(name, data.length, "0", opts), padToBlock(data)];
}

function dirEntry(name: string): Uint8Array[] {
    return [tarHeader(name, 0, "5")];
}

function symlinkEntry(name: string, target: string): Uint8Array[] {
    return [tarHeader(name, 0, "2", { linkname: target })];
}

function hardlinkEntry(name: string, target: string): Uint8Array[] {
    return [tarHeader(name, 0, "1", { linkname: target })];
}

function paxRecord(key: string, value: string): Uint8Array {
    const body = `${key}=${value}\n`;
    let length = body.length + 2;

    for (;;) {
        const digits = String(length).length;

        if (body.length + digits + 1 === length) {
            break;
        }

        length = body.length + digits + 1;
    }

    return encode(`${length} ${body}`);
}

function paxFile(name: string, data: Uint8Array, opts: HeaderOptions = {}): Uint8Array[] {
    const record = paxRecord("path", name);

    // The real header keeps a truncated name: only the pax record carries the full one.
    return [tarHeader("PaxHeaders/smiley", record.length, "x"), padToBlock(record), ...fileEntry(name.slice(0, 100), data, opts)];
}

// GNU `L` long-name record followed by the real (truncated) header.
function gnuLongNameFile(name: string, data: Uint8Array): Uint8Array[] {
    const body = encode(`${name}\0`);

    return [tarHeader("././@LongLink", body.length, "L"), padToBlock(body), ...fileEntry(name.slice(0, 100), data)];
}

function tarGz(parts: Uint8Array[]): Uint8Array {
    return gzipSync(buildTar(parts));
}

async function writeArchive(dir: string, data: Uint8Array, name = "archive.tar.gz"): Promise<string> {
    const path = join(dir, name);

    await writeFile(path, data);

    return path;
}

describe("extractTarGz", () => {
    test("extracts regular files and nested directories", async () => {
        const dir = tempDir("basic");
        const archive = await writeArchive(
            dir,
            tarGz([
                ...dirEntry("bin/"),
                ...fileEntry("bin/llama-server", encode("binary"), { mode: 0o755 }),
                ...dirEntry("lib/"),
                ...fileEntry("lib/libllama.so.0", encode("shared")),
                ...fileEntry("empty.txt", new Uint8Array(0)),
            ]),
        );
        const out = join(dir, "out");
        const written = await extractTarGz(archive, out);

        expect(written.sort()).toEqual(["bin/llama-server", "empty.txt", "lib/libllama.so.0"]);
        expect(decode(readFileSync(join(out, "bin/llama-server")))).toBe("binary");
        expect(decode(readFileSync(join(out, "lib/libllama.so.0")))).toBe("shared");
        expect(statSync(join(out, "empty.txt")).size).toBe(0);
    });

    test("reads a name longer than 100 chars from a pax record", async () => {
        const dir = tempDir("pax");
        const name = `share/${"deep/".repeat(30)}model.gguf`;
        const archive = await writeArchive(dir, tarGz(paxFile(name, encode("pax-content"))));
        const out = join(dir, "out");
        const written = await extractTarGz(archive, out);

        expect(name.length).toBeGreaterThan(100);
        expect(written).toEqual([name]);
        expect(decode(readFileSync(join(out, name)))).toBe("pax-content");
    });

    test("reads a name longer than 100 chars from a GNU L record", async () => {
        const dir = tempDir("gnu");
        const name = `lib/${"x".repeat(140)}.so`;
        const archive = await writeArchive(dir, tarGz(gnuLongNameFile(name, encode("gnu-content"))));
        const out = join(dir, "out");
        const written = await extractTarGz(archive, out);

        expect(written).toEqual([name]);
        expect(decode(readFileSync(join(out, name)))).toBe("gnu-content");
    });

    test("accepts a base-256 encoded size", async () => {
        const dir = tempDir("base256");
        const archive = await writeArchive(dir, tarGz(fileEntry("odd.bin", encode("base256"), { base256Size: true })));
        const out = join(dir, "out");

        await extractTarGz(archive, out);
        expect(decode(readFileSync(join(out, "odd.bin")))).toBe("base256");
    });

    test("links: symlink inside, hardlink copy", async () => {
        const dir = tempDir("links");
        const archive = await writeArchive(
            dir,
            tarGz([
                ...fileEntry("lib/libllama.so.0", encode("versioned")),
                ...symlinkEntry("lib/libllama.so", "libllama.so.0"),
                ...hardlinkEntry("lib/libllama.so.0.hard", "lib/libllama.so.0"),
            ]),
        );
        const out = join(dir, "out");

        await extractTarGz(archive, out);

        const linkPath = join(out, "lib/libllama.so");

        if (process.platform === "win32") {
            // Windows needs a privilege for symlinks, so the extractor copies the target instead.
            expect(decode(readFileSync(linkPath))).toBe("versioned");
        } else {
            expect(lstatSync(linkPath).isSymbolicLink()).toBe(true);
            expect(readlinkSync(linkPath)).toBe("libllama.so.0");
        }

        expect(decode(readFileSync(join(out, "lib/libllama.so.0.hard")))).toBe("versioned");
    });

    test("resolves a chained symlink (lib.so -> lib.so.1 -> lib.so.1.0.0)", async () => {
        const dir = tempDir("chain");
        const archive = await writeArchive(
            dir,
            tarGz([
                ...fileEntry("lib/libllama.so.1.0.0", encode("chained")),
                // The link to the real file is listed after the link that points at it, which is
                // the order real llama.cpp archives use.
                ...symlinkEntry("lib/libllama.so", "libllama.so.1"),
                ...symlinkEntry("lib/libllama.so.1", "libllama.so.1.0.0"),
            ]),
        );
        const out = join(dir, "out");

        await extractTarGz(archive, out);

        expect(decode(readFileSync(join(out, "lib/libllama.so")))).toBe("chained");
    });

    test("rejects a symlink that escapes the destination", async () => {
        const dir = tempDir("slip-link");
        const archive = await writeArchive(dir, tarGz([...symlinkEntry("lib/evil.so", "../../etc/passwd")]));
        const out = join(dir, "out");

        await expect(extractTarGz(archive, out)).rejects.toThrow(/outside the destination/);
        expect(await exists(join(out, "lib/evil.so"))).toBe(false);
    });

    test("rejects an absolute symlink target", async () => {
        const dir = tempDir("abs-link");
        const archive = await writeArchive(dir, tarGz([...symlinkEntry("lib/evil.so", "/etc/passwd")]));

        await expect(extractTarGz(archive, join(dir, "out"))).rejects.toThrow(/absolute symlink target/);
    });

    test("rejects traversal and absolute entry names", async () => {
        for (const name of ["../evil.txt", "/abs.txt", "C:/x.txt", "a/../../b.txt", "..\\evil.txt"]) {
            const dir = tempDir("slip");
            const archive = await writeArchive(dir, tarGz(fileEntry(name, encode("evil"))));

            await expect(extractTarGz(archive, join(dir, "out"))).rejects.toThrow(/refusing/);
            expect(await exists(join(dir, "evil.txt"))).toBe(false);
        }
    });

    test("enforces the caps while streaming", async () => {
        const big = new Uint8Array(200_000);
        big.fill(9);

        const countDir = tempDir("cap-count");
        const countArchive = await writeArchive(
            countDir,
            tarGz([...fileEntry("a.txt", encode("a")), ...fileEntry("b.txt", encode("b")), ...fileEntry("c.txt", encode("c"))]),
        );

        await expect(extractTarGz(countArchive, join(countDir, "out"), { maxFiles: 2 })).rejects.toThrow(/more than 2 files/);

        const fileDir = tempDir("cap-file");
        const fileArchive = await writeArchive(fileDir, tarGz(fileEntry("big.bin", big)));

        await expect(extractTarGz(fileArchive, join(fileDir, "out"), { maxFileBytes: 50_000 })).rejects.toThrow(/byte limit/);
        expect(await exists(join(fileDir, "out", "big.bin"))).toBe(false);

        const totalDir = tempDir("cap-total");
        const totalArchive = await writeArchive(
            totalDir,
            tarGz([...fileEntry("one.bin", big), ...fileEntry("two.bin", big)]),
        );

        await expect(extractTarGz(totalArchive, join(totalDir, "out"), { maxTotalBytes: 150_000 })).rejects.toThrow(
            /archive is over/,
        );
    });

    test("rejects a truncated archive and removes the partial file", async () => {
        const dir = tempDir("truncated");
        const random = new Uint8Array(200_000);

        for (let index = 0; index < random.length; index += 1) {
            random[index] = (index * 7919) % 251;
        }

        const full = tarGz(fileEntry("big.bin", random));
        const archive = await writeArchive(dir, full.slice(0, Math.floor(full.length * 0.6)));

        await expect(extractTarGz(archive, join(dir, "out"))).rejects.toThrow();
        expect(await exists(join(dir, "out", "big.bin"))).toBe(false);
    });

    test("round-trips a 60 MB entry", async () => {
        const dir = tempDir("large");
        const total = 60 * 1024 * 1024;
        const big = new Uint8Array(total);

        for (let index = 0; index < 1024; index += 1) {
            big[index] = index % 256;
        }

        big.fill(17, 1024);

        const archive = await writeArchive(dir, tarGz(fileEntry("big.bin", big)));
        const out = join(dir, "out");

        expect(await extractTarGz(archive, out)).toEqual(["big.bin"]);

        const extracted = readFileSync(join(out, "big.bin"));
        const hash = new Bun.CryptoHasher("sha256").update(big).digest("hex");

        expect(extracted.length).toBe(total);
        expect(new Bun.CryptoHasher("sha256").update(extracted).digest("hex")).toBe(hash);
    }, 60_000);

    test("keeps the tar mode bits on non-Windows hosts", async () => {
        const dir = tempDir("modes");
        const archive = await writeArchive(
            dir,
            tarGz([
                ...fileEntry("llama-server", encode("x"), { mode: 0o755 }),
                ...fileEntry("readme.md", encode("r"), { mode: 0o644 }),
                ...fileEntry("loose", encode("l"), { mode: 0o777 }),
            ]),
        );
        const out = join(dir, "out");

        await extractTarGz(archive, out);

        if (process.platform === "win32") {
            return;
        }

        expect(statSync(join(out, "llama-server")).mode & 0o777).toBe(0o755);
        expect(statSync(join(out, "readme.md")).mode & 0o777).toBe(0o644);
        expect(statSync(join(out, "loose")).mode & 0o777).toBe(0o755);
    });
});

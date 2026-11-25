// Streaming `.tar.gz` extraction: llama.cpp ships every non-Windows build as a tarball
// (`llama-b12686-bin-ubuntu-x64.tar.gz`, `cudart-llama-b12686-bin-ubuntu-cuda-13.5-x64.tar.gz`),
// so the installer cannot rely on `extractZip` alone.
//
// The gzip stream is decompressed with fflate's `Gunzip` and parsed by the small ustar reader
// below: 512-byte headers, GNU `L`/`K` long names and pax `x` records, octal and base-256 sizes.
// Nothing is buffered beyond one decompression chunk plus the entry currently being written, and
// the caps are checked as the bytes arrive.

import { chmodSync, copyFileSync, createWriteStream, existsSync, mkdirSync, symlinkSync, type WriteStream } from "node:fs";
import { unlink } from "node:fs/promises";
import { finished } from "node:stream/promises";
import { dirname, resolve, sep } from "node:path";

import { Gunzip } from "fflate";

import type { ExtractZipOptions } from "./unzip";

const DEFAULT_MAX_FILES = 2000;
const DEFAULT_MAX_FILE_BYTES = 1024 ** 3;
const DEFAULT_MAX_TOTAL_BYTES = 3 * 1024 ** 3;

// pax headers and GNU long names are a few hundred bytes; anything approaching this is hostile.
const MAX_METADATA_BYTES = 1024 ** 2;

const BLOCK_SIZE = 512;
const textDecoder = new TextDecoder();

type PendingLink = {
    kind: "symlink" | "hardlink";
    path: string; // absolute path of the link inside destDir
    target: string; // raw link text, normalised to forward slashes
    resolved: string; // absolute path the target resolves to, already checked to be inside
};

type ModeEntry = { path: string; mode: number };

function isInside(root: string, target: string): boolean {
    return target === root || target.startsWith(root + sep);
}

// Same rules as the zip path (kept in sync with unzip.ts): normalise separators, then reject
// absolute, drive-qualified and `..` entries. Returns null for a directory-style name.
function safeEntryPath(name: string): string | null {
    const normalized = name.replace(/\\/g, "/");

    if (normalized.startsWith("/")) {
        throw new Error(`refusing absolute entry path "${name}"`);
    }

    if (/^[a-zA-Z]:/.test(normalized)) {
        throw new Error(`refusing drive-qualified entry path "${name}"`);
    }

    const segments = normalized.split("/");

    if (segments.includes("..")) {
        throw new Error(`refusing entry path with ".." segments "${name}"`);
    }

    if (normalized.endsWith("/")) {
        return null;
    }

    const relative = segments.filter((segment) => segment !== "" && segment !== ".").join("/");

    return relative || null;
}

function readString(bytes: Uint8Array): string {
    const end = bytes.indexOf(0);

    return textDecoder.decode(end === -1 ? bytes : bytes.subarray(0, end));
}

function isZeroBlock(block: Uint8Array): boolean {
    for (let index = 0; index < block.length; index += 1) {
        if (block[index] !== 0) {
            return false;
        }
    }

    return true;
}

// Sizes and modes are either octal text or, for values that do not fit, a base-256 integer with
// the high bit of the first byte set (GNU extension). Both appear in real archives.
function parseNumeric(field: Uint8Array): number {
    if (field[0] & 0x80) {
        if (field[0] === 0xff) {
            return -1; // negative base-256 is not meaningful for size/mode
        }

        let value = 0;

        for (let index = 1; index < field.length; index += 1) {
            value = value * 256 + field[index];

            if (!Number.isSafeInteger(value)) {
                throw new Error("archive entry size is too large");
            }
        }

        return value;
    }

    const text = readString(field).trim();

    if (!text) {
        return 0;
    }

    const value = Number.parseInt(text, 8);

    return Number.isFinite(value) ? value : -1;
}

function parsePax(body: Uint8Array): Map<string, string> {
    const entries = new Map<string, string>();
    const text = textDecoder.decode(body);
    let offset = 0;

    // Records look like `27 path=some/very/long/file/name\n`, the leading decimal being the
    // length of the whole record including its own digits.
    while (offset < text.length) {
        const space = text.indexOf(" ", offset);

        if (space === -1) {
            break;
        }

        const length = Number(text.slice(offset, space));

        if (!Number.isFinite(length) || length <= 0) {
            break;
        }

        const record = text.slice(offset, offset + length);
        const body = record.slice(space - offset + 1).replace(/\n$/, "");
        const equals = body.indexOf("=");

        if (equals > 0) {
            entries.set(body.slice(0, equals), body.slice(equals + 1));
        }

        offset += length;
    }

    return entries;
}

// See unzip.ts: an errored stream never drains, so waiting only for 'drain' can hang forever.
function waitForDrain(stream: WriteStream, failureOf: () => Error | null): Promise<void> {
    return new Promise<void>((resolve, reject) => {
        const onDrain = () => {
            cleanup();
            resolve();
        };
        const onError = (error: Error) => {
            cleanup();
            reject(error);
        };
        const onClose = () => {
            cleanup();
            reject(failureOf() ?? new Error("write stream closed before it drained"));
        };
        const cleanup = () => {
            stream.off("drain", onDrain);
            stream.off("error", onError);
            stream.off("close", onClose);
        };

        const failure = failureOf();

        if (failure) {
            reject(failure);

            return;
        }

        stream.once("drain", onDrain);
        stream.once("error", onError);
        stream.once("close", onClose);
    });
}

export async function extractTarGz(archivePath: string, destDir: string, opts: ExtractZipOptions = {}): Promise<string[]> {
    const root = resolve(destDir);
    const maxFiles = opts.maxFiles ?? DEFAULT_MAX_FILES;
    const maxFileBytes = opts.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
    const maxTotalBytes = opts.maxTotalBytes ?? DEFAULT_MAX_TOTAL_BYTES;

    mkdirSync(root, { recursive: true });

    const written: string[] = [];
    const modes: ModeEntry[] = [];
    const links: PendingLink[] = [];
    const completions: Promise<unknown>[] = [];
    const openStreams = new Set<WriteStream>();

    // Decompressed bytes that have not been consumed yet, as views into fflate's output.
    const queue: Uint8Array[] = [];
    let headOffset = 0;
    let buffered = 0;

    let phase: "header" | "data" | "metadata" | "padding" | "end" = "header";
    let remaining = 0; // bytes left in the current entry body
    let paddingLeft = 0;
    let entrySize = 0;
    let entryBytes = 0;
    let entryName = "";
    let stream: WriteStream | null = null;
    let metadataType = "";
    let paxNext = new Map<string, string>();
    let longName: string | null = null;
    let longLink: string | null = null;

    let files = 0;
    let totalBytes = 0;
    let inProgressPath: string | null = null;
    let failure: Error | null = null;
    let blocked: Promise<void> | null = null;

    function take(count: number): Uint8Array {
        const out = new Uint8Array(count);
        let filled = 0;

        while (filled < count) {
            const head = queue[0];
            const use = Math.min(head.length - headOffset, count - filled);

            out.set(head.subarray(headOffset, headOffset + use), filled);
            filled += use;
            headOffset += use;

            if (headOffset === head.length) {
                queue.shift();
                headOffset = 0;
            }
        }

        buffered -= count;

        return out;
    }

    function skip(count: number): void {
        let left = count;

        while (left > 0 && queue.length > 0) {
            const head = queue[0];
            const use = Math.min(head.length - headOffset, left);

            headOffset += use;
            left -= use;
            buffered -= use;

            if (headOffset === head.length) {
                queue.shift();
                headOffset = 0;
            }
        }
    }

    function finishEntryBody(): void {
        if (stream) {
            // No drain wait after the last chunk: end() flushes what is already queued.
            blocked = null;
            stream.end();
            completions.push(finished(stream));
            openStreams.delete(stream);
            stream = null;
            inProgressPath = null;
        }

        const pad = (BLOCK_SIZE - (entrySize % BLOCK_SIZE)) % BLOCK_SIZE;

        paddingLeft = pad;
        phase = pad > 0 ? "padding" : "header";
    }

    function writeEntryBody(count: number): number {
        const target = stream;

        if (!target) {
            return 0;
        }

        let left = count;

        while (left > 0 && queue.length > 0 && !blocked) {
            const head = queue[0];
            const use = Math.min(head.length - headOffset, left);

            // Copy: fflate reuses its output buffers, so a view could change under the stream.
            const chunk = head.slice(headOffset, headOffset + use);

            headOffset += use;
            left -= use;
            buffered -= use;

            if (headOffset === head.length) {
                queue.shift();
                headOffset = 0;
            }

            entryBytes += chunk.length;
            totalBytes += chunk.length;

            if (entryBytes > maxFileBytes) {
                throw new Error(`refusing to extract ${entryName}: file is over the ${maxFileBytes} byte limit`);
            }

            if (totalBytes > maxTotalBytes) {
                throw new Error(`refusing to extract ${entryName}: archive is over the ${maxTotalBytes} byte limit`);
            }

            if (!target.write(chunk)) {
                blocked = waitForDrain(target, () => failure);
            }
        }

        return count - left;
    }

    function addLink(kind: "symlink" | "hardlink", name: string, target: string): void {
        const relative = safeEntryPath(name);

        if (!relative) {
            throw new Error(`refusing ${kind} with an unusable path "${name}"`);
        }

        const linkText = target.replace(/\\/g, "/");

        if (!linkText) {
            throw new Error(`refusing ${kind} without a target: "${name}"`);
        }

        if (linkText.startsWith("/") || /^[a-zA-Z]:/.test(linkText)) {
            throw new Error(`refusing absolute ${kind} target "${target}" for "${name}"`);
        }

        files += 1;

        if (files > maxFiles) {
            throw new Error(`refusing to extract ${name}: archive has more than ${maxFiles} files`);
        }

        // Symlink targets resolve against the link's own directory, hardlink targets against the
        // archive root, so the containment check needs a different base for each.
        const linkPath = resolve(root, relative);
        const base = kind === "symlink" ? dirname(linkPath) : root;
        const resolved = resolve(base, linkText);

        if (!isInside(root, resolved)) {
            throw new Error(`refusing ${kind} target outside the destination: "${target}" for "${name}"`);
        }

        links.push({ kind, path: linkPath, target: linkText, resolved });
    }

    function startEntry(block: Uint8Array, typeflag: string, name: string, size: number, mode: number, linkTarget: string): void {
        if (typeflag === "5") {
            const relative = safeEntryPath(name.endsWith("/") ? name.slice(0, -1) : name);

            if (!relative) {
                throw new Error(`refusing directory entry with an unusable path "${name}"`);
            }

            const target = resolve(root, relative);

            if (!isInside(root, target)) {
                throw new Error(`refusing entry outside the destination: ${name}`);
            }

            mkdirSync(target, { recursive: true });
            modes.push({ path: target, mode });

            entrySize = size;
            phase = "data"; // the (usually zero) body is skipped

            return;
        }

        if (typeflag === "2" || typeflag === "1") {
            addLink(typeflag === "2" ? "symlink" : "hardlink", name, linkTarget);

            entrySize = size;
            phase = "data";

            return;
        }

        if (typeflag !== "0" && typeflag !== "\0" && typeflag !== "") {
            // Devices, fifos, global pax headers and anything unknown: consume and ignore.
            entrySize = size;
            phase = "data";

            return;
        }

        const relative = safeEntryPath(name);

        if (!relative) {
            throw new Error(`refusing file entry with an unusable path "${name}"`);
        }

        files += 1;

        if (files > maxFiles) {
            throw new Error(`refusing to extract ${name}: archive has more than ${maxFiles} files`);
        }

        if (size > maxFileBytes) {
            throw new Error(`refusing to extract ${name}: ${size} bytes is over the ${maxFileBytes} byte limit`);
        }

        const target = resolve(root, relative);

        if (!isInside(root, target)) {
            throw new Error(`refusing entry outside the destination: ${name}`);
        }

        mkdirSync(dirname(target), { recursive: true });

        const sink = createWriteStream(target);

        sink.on("error", (error: Error) => {
            failure ??= error;
        });

        openStreams.add(sink);
        inProgressPath = target;
        stream = sink;
        entryName = name;
        entrySize = size;
        entryBytes = 0;
        phase = "data";
        modes.push({ path: target, mode });
        written.push(relative);
    }

    function handleHeader(block: Uint8Array): void {
        if (isZeroBlock(block)) {
            // Two zero blocks end the archive; some writers pad with megabytes of zeros.
            phase = "end";

            return;
        }

        const typeflag = block[156] === 0 ? "\0" : String.fromCharCode(block[156]);
        const size = parseNumeric(block.subarray(124, 136));

        if (size < 0) {
            throw new Error("invalid entry size in tar header");
        }

        if (typeflag === "x" || typeflag === "g" || typeflag === "L" || typeflag === "K") {
            if (size > MAX_METADATA_BYTES) {
                throw new Error(`refusing ${typeflag} header with ${size} bytes of metadata`);
            }

            metadataType = typeflag;
            entrySize = size;
            remaining = size;
            phase = "metadata";

            return;
        }

        const mode = parseNumeric(block.subarray(100, 108)) & 0o777;
        const nameField = readString(block.subarray(0, 100));
        const prefixField = readString(block.subarray(345, 500));
        const isUstar = readString(block.subarray(257, 263)).startsWith("ustar");
        const paxName = paxNext.get("path");
        const paxLink = paxNext.get("linkpath");
        const name = paxName ?? longName ?? (isUstar && prefixField ? `${prefixField}/${nameField}` : nameField);
        const linkTarget = paxLink ?? longLink ?? readString(block.subarray(157, 257));

        // pax/long-name records only apply to the entry that follows them.
        paxNext = new Map();
        longName = null;
        longLink = null;

        remaining = size;
        startEntry(block, typeflag, name, size, mode, linkTarget);
    }

    function handleMetadata(body: Uint8Array): void {
        if (metadataType === "x" || metadataType === "g") {
            const records = parsePax(body);

            if (metadataType === "x") {
                paxNext = records;
            }
        } else if (metadataType === "L") {
            longName = readString(body).replace(/\0/g, "");
        } else if (metadataType === "K") {
            longLink = readString(body).replace(/\0/g, "");
        }
    }

    function pump(): void {
        for (;;) {
            if (failure || blocked) {
                return;
            }

            if (phase === "header") {
                if (buffered < BLOCK_SIZE) {
                    return;
                }

                handleHeader(take(BLOCK_SIZE));
                continue;
            }

            if (phase === "metadata") {
                if (buffered < remaining) {
                    return;
                }

                handleMetadata(take(remaining));

                const pad = (BLOCK_SIZE - (entrySize % BLOCK_SIZE)) % BLOCK_SIZE;

                paddingLeft = pad;
                phase = pad > 0 ? "padding" : "header";
                continue;
            }

            if (phase === "padding") {
                const use = Math.min(paddingLeft, buffered);

                if (use === 0) {
                    return;
                }

                skip(use);
                paddingLeft -= use;

                if (paddingLeft === 0) {
                    phase = "header";
                }

                continue;
            }

            if (phase === "data") {
                if (remaining === 0) {
                    finishEntryBody();
                    continue;
                }

                if (buffered === 0) {
                    return;
                }

                if (!stream) {
                    // Directory, link or skipped entry: consume the body without writing it.
                    const use = Math.min(remaining, buffered);

                    skip(use);
                    remaining -= use;
                    continue;
                }

                const consumed = writeEntryBody(Math.min(remaining, buffered));

                if (consumed === 0) {
                    return; // blocked on the write stream
                }

                remaining -= consumed;
                continue;
            }

            // "end": drop whatever trailing padding is left.
            skip(buffered);

            return;
        }
    }

    const gunzip = new Gunzip((chunk) => {
        if (chunk.length > 0) {
            queue.push(chunk);
            buffered += chunk.length;
        }
    });

    const reader = Bun.file(archivePath).stream().getReader();

    try {
        const pumpUntilSettled = async () => {
            for (;;) {
                pump();

                if (failure) {
                    throw failure;
                }

                if (!blocked) {
                    return;
                }

                const pending = blocked;

                blocked = null;
                await pending;
            }
        };

        for (;;) {
            const { done, value } = await reader.read();

            if (done) {
                break;
            }

            if (value && value.length > 0) {
                gunzip.push(value, false);
            }

            await pumpUntilSettled();
        }

        gunzip.push(new Uint8Array(0), true);
        await pumpUntilSettled();

        // Read through a function: the parser mutates these from a callback, so a direct check
        // would be narrowed to their values at declaration time.
        const truncated = () => phase !== "end" && (phase !== "header" || buffered > 0);

        // Anything still mid-entry or mid-header means the tarball was cut short.
        if (truncated()) {
            throw new Error("truncated archive: unexpected end inside a tar entry");
        }

        await Promise.all(completions);
    } catch (error) {
        const pending = blocked as Promise<void> | null;

        if (pending) {
            pending.catch(() => undefined);
            blocked = null;
        }

        await reader.cancel().catch(() => undefined);

        const closing = [...openStreams].map((open) => finished(open).catch(() => undefined));

        for (const open of openStreams) {
            open.destroy();
        }

        openStreams.clear();
        await Promise.all(closing);

        if (inProgressPath) {
            await unlink(inProgressPath).catch(() => undefined);
        }

        throw error instanceof Error ? error : new Error(String(error));
    }

    // Links come last: a hardlink may reference a file that appears later in the archive, and the
    // Windows symlink fallback needs the target to exist.
    const copyJobs: PendingLink[] = [];

    for (const link of links) {
        mkdirSync(dirname(link.path), { recursive: true });

        if (link.kind === "symlink" && process.platform !== "win32") {
            symlinkSync(link.target, link.path);
            continue;
        }

        copyJobs.push(link);
    }

    // Real llama.cpp archives chain their links (`libllama.so -> libllama.so.1 -> libllama.so.1.0.0`),
    // so a copy job whose target is another link only becomes possible after that one exists:
    // retry in passes instead of depending on the order inside the tar.
    let pendingJobs = copyJobs;

    for (let pass = 0; pendingJobs.length > 0 && pass < 8; pass += 1) {
        const stillMissing: PendingLink[] = [];

        for (const job of pendingJobs) {
            if (!existsSync(job.resolved)) {
                stillMissing.push(job);
                continue;
            }

            copyFileSync(job.resolved, job.path);
        }

        if (stillMissing.length === pendingJobs.length) {
            break; // no progress: the targets are genuinely absent
        }

        pendingJobs = stillMissing;
    }

    const brokenHardlink = pendingJobs.find((job) => job.kind === "hardlink");

    if (brokenHardlink) {
        throw new Error(`hardlink target "${brokenHardlink.target}" is missing from the archive`);
    }

    // Whatever symlink is left had no target in the archive; on Windows that is skipped on
    // purpose (creating links needs a privilege the app does not ask for).

    if (process.platform !== "win32") {
        for (const entry of modes) {
            // Masking keeps setuid/sticky bits and group write out of an extracted archive.
            chmodSync(entry.path, entry.mode & 0o755);
        }
    }

    return written;
}

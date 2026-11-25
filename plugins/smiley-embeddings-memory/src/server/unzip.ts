// Streaming ZIP extraction with the two protections an installer needs: archive members can
// never be written outside the destination directory, and the caps are enforced while the data
// arrives instead of after a 400 MB archive has already been expanded into memory or disk.
//
// fflate's `Unzip` stream is fed with the file's own ReadableStream, so no complete archive is
// ever held in memory; the write streams provide the backpressure by pausing the feed.

import { createWriteStream, mkdirSync, type WriteStream } from "node:fs";
import { chmod, unlink } from "node:fs/promises";
import { finished } from "node:stream/promises";
import { dirname, resolve, sep } from "node:path";

import { Unzip, UnzipInflate, type UnzipFile } from "fflate";

export type ExtractZipOptions = { maxFiles?: number; maxFileBytes?: number; maxTotalBytes?: number };

const DEFAULT_MAX_FILES = 2000;
const DEFAULT_MAX_FILE_BYTES = 1024 ** 3;
const DEFAULT_MAX_TOTAL_BYTES = 3 * 1024 ** 3;

function isInside(root: string, target: string): boolean {
    return target === root || target.startsWith(root + sep);
}

// Returns the path to write, or null for entries that only describe a directory. Throws for
// anything that would land outside `destDir`.
function safeEntryPath(name: string): string | null {
    // Archive names are arbitrary bytes and Windows-built zips use backslashes: normalise first,
    // otherwise `..\\evil` would pass the `/`-based checks on a POSIX host.
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

function isExecutableName(relative: string): boolean {
    // fflate's streaming reader does not expose the Unix mode bits stored in the central
    // directory, so fall back to llama.cpp's naming: `llama-*` binaries and the extensionless
    // helper tools are the only things that must be executable.
    const base = relative.slice(relative.lastIndexOf("/") + 1);

    return base.startsWith("llama-") || !base.includes(".");
}

// fflate hands data to us synchronously, so backpressure means waiting between pushes. The wait
// has to settle on 'error'/'close' too: an errored stream never drains again, so `once(stream,
// "drain")` alone would leave the extraction hanging forever.
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

export async function extractZip(zipPath: string, destDir: string, opts: ExtractZipOptions = {}): Promise<string[]> {
    const root = resolve(destDir);
    const maxFiles = opts.maxFiles ?? DEFAULT_MAX_FILES;
    const maxFileBytes = opts.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
    const maxTotalBytes = opts.maxTotalBytes ?? DEFAULT_MAX_TOTAL_BYTES;

    mkdirSync(root, { recursive: true });

    const written: string[] = [];
    const completions: Promise<unknown>[] = [];
    const openStreams = new Set<WriteStream>();

    let files = 0;
    let totalBytes = 0;
    let inProgressPath: string | null = null;
    let failure: Error | null = null;
    let blocked: Promise<void> | null = null;

    const unzip = new Unzip();

    unzip.register(UnzipInflate);

    unzip.onfile = (file: UnzipFile) => {
        // Thrown from inside a fflate callback, this aborts the current push and therefore the
        // whole extraction, which is what we want for a hostile archive.
        const relative = safeEntryPath(file.name);

        if (relative === null) {
            // Directory (or root) entry: nothing to write, but fflate still needs a sink or it
            // keeps the entry's bytes queued in memory.
            file.ondata = () => undefined;
            file.start();

            return;
        }

        files += 1;

        if (files > maxFiles) {
            throw new Error(`refusing to extract ${file.name}: archive has more than ${maxFiles} files`);
        }

        const target = resolve(root, relative);

        if (!isInside(root, target)) {
            throw new Error(`refusing entry outside the destination: ${file.name}`);
        }

        if (typeof file.originalSize === "number" && file.originalSize > maxFileBytes) {
            throw new Error(
                `refusing to extract ${file.name}: ${file.originalSize} bytes is over the ${maxFileBytes} byte limit`,
            );
        }

        mkdirSync(dirname(target), { recursive: true });

        const stream = createWriteStream(target);
        let bytes = 0;

        openStreams.add(stream);
        inProgressPath = target;

        stream.on("error", (error: Error) => {
            failure ??= error;
        });

        file.ondata = (error, chunk, final) => {
            if (error) {
                throw error;
            }

            if (chunk && chunk.length > 0) {
                bytes += chunk.length;
                totalBytes += chunk.length;

                if (bytes > maxFileBytes) {
                    throw new Error(`refusing to extract ${file.name}: file is over the ${maxFileBytes} byte limit`);
                }

                if (totalBytes > maxTotalBytes) {
                    throw new Error(`refusing to extract ${file.name}: archive is over the ${maxTotalBytes} byte limit`);
                }

                if (!stream.write(chunk) && !final) {
                    // Only arm the wait when more data can follow: on the final chunk end()
                    // flushes what is already queued.
                    blocked = waitForDrain(stream, () => failure);
                }
            }

            if (final) {
                // No drain wait after the last chunk: end() flushes what is already queued.
                blocked = null;
                stream.end();
                completions.push(finished(stream));
                openStreams.delete(stream);
                written.push(relative);
                inProgressPath = null;
            }
        };

        file.start();
    };

    const reader = Bun.file(zipPath).stream().getReader();

    try {
        for (;;) {
            const { done, value } = await reader.read();

            if (done) {
                break;
            }

            if (value && value.length > 0) {
                unzip.push(value, false);
            }

            if (blocked) {
                const pending = blocked;
                blocked = null;
                await pending;
            }

            if (failure) {
                throw failure;
            }
        }

        unzip.push(new Uint8Array(0), true);
        await Promise.all(completions);
    } catch (error) {
        // A drain wait armed by the last push may still reject later (the stream is about to be
        // destroyed); consume it so it can never surface as an unhandled rejection.
        const pending = blocked as Promise<void> | null;

        if (pending) {
            pending.catch(() => undefined);
            blocked = null;
        }

        await reader.cancel().catch(() => undefined);

        // Windows refuses to delete a file that still has an open handle, so the streams have to
        // be closed before the partial entry can be removed.
        const closing = [...openStreams].map((stream) => finished(stream).catch(() => undefined));

        for (const stream of openStreams) {
            stream.destroy();
        }

        openStreams.clear();
        await Promise.all(closing);

        // Only the entry that was mid-write is removed: the completed files may be from an
        // earlier install that shares the destination.
        if (inProgressPath) {
            await unlink(inProgressPath).catch(() => undefined);
        }

        throw error instanceof Error ? error : new Error(String(error));
    }

    if (process.platform !== "win32") {
        for (const relative of written) {
            if (isExecutableName(relative)) {
                await chmod(resolve(root, relative), 0o755).catch(() => undefined);
            }
        }
    }

    return written;
}

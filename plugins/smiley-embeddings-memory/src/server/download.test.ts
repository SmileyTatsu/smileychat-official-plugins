import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { DownloadError, downloadFile, isAllowedDownloadUrl } from "./download";

// A deterministic body big enough that the 250 ms progress throttle has something to do.
const BODY = new Uint8Array(300_000);

for (let index = 0; index < BODY.length; index += 1) {
    BODY[index] = (index * 31) % 251;
}

const BODY_SHA = new Bun.CryptoHasher("sha256").update(BODY).digest("hex");

type Range = { start: number; end: number };

function parseRange(header: string | null, length: number): Range | undefined {
    if (!header) {
        return undefined;
    }

    const match = /^bytes=(\d+)-(\d*)$/.exec(header.trim());

    if (!match) {
        return undefined;
    }

    const start = Number(match[1]);
    const end = match[2] ? Number(match[2]) : length - 1;

    return { start, end };
}

function partial(body: Uint8Array, range: Range): Response {
    const slice = body.slice(range.start, range.end + 1);

    return new Response(slice, {
        status: 206,
        headers: {
            "content-length": String(slice.length),
            "content-range": `bytes ${range.start}-${range.end}/${body.length}`,
        },
    });
}

let server: ReturnType<typeof Bun.serve>;
let baseUrl = "";
// Kept in an object so TypeScript does not narrow the properties to `null` after a reset.
const bookkeeping = { lastRange: null as string | null, requestCount: 0 };

// Read through a function: TypeScript keeps the narrowing from a `= null` assignment otherwise.
function lastRange(): string | null {
    return bookkeeping.lastRange;
}
let scratch = "";

beforeAll(async () => {
    scratch = await mkdtemp(join(tmpdir(), "smiley-download-"));

    server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        async fetch(request) {
            const url = new URL(request.url);
            const range = request.headers.get("range");
            bookkeeping.requestCount += 1;
            bookkeeping.lastRange = range;

            if (url.pathname === "/file") {
                const parsed = parseRange(range, BODY.length);

                if (!parsed) {
                    return new Response(BODY, {
                        headers: { "content-length": String(BODY.length), "accept-ranges": "bytes" },
                    });
                }

                if (parsed.start >= BODY.length) {
                    return new Response(null, {
                        status: 416,
                        headers: { "content-range": `bytes */${BODY.length}` },
                    });
                }

                return partial(BODY, parsed);
            }

            if (url.pathname === "/norange") {
                // Simulates a CDN that ignores Range and returns the whole file with 200.
                return new Response(BODY, { headers: { "content-length": String(BODY.length) } });
            }

            if (url.pathname === "/slow") {
                const parsed = parseRange(range, BODY.length);
                let offset = parsed?.start ?? 0;
                const chunk = 15_000;

                const stream = new ReadableStream<Uint8Array>({
                    async pull(controller) {
                        if (offset >= BODY.length) {
                            controller.close();
                            return;
                        }

                        await new Promise((resolve) => setTimeout(resolve, 25));
                        const next = Math.min(offset + chunk, BODY.length);
                        controller.enqueue(BODY.slice(offset, next));
                        offset = next;
                    },
                });

                return new Response(stream, {
                    status: parsed ? 206 : 200,
                    headers: parsed
                        ? { "content-range": `bytes ${parsed.start}-${BODY.length - 1}/${BODY.length}` }
                        : { "content-length": String(BODY.length) },
                });
            }

            if (url.pathname === "/redirect-foreign") {
                return new Response(null, { status: 302, headers: { location: "http://evil.example/file" } });
            }

            if (url.pathname === "/redirect-a") {
                return new Response(null, { status: 302, headers: { location: "/redirect-b" } });
            }

            if (url.pathname === "/redirect-b") {
                return new Response(null, { status: 307, headers: { location: `${baseUrl}/file` } });
            }

            return new Response("not found", { status: 404 });
        },
    });

    baseUrl = `http://127.0.0.1:${server.port}`;
});

afterAll(async () => {
    server.stop(true);
    await rm(scratch, { recursive: true, force: true });
});

function allowLocal(url: string): boolean {
    return url.startsWith(baseUrl);
}

let counter = 0;

function tempPath(name: string): string {
    counter += 1;

    return join(scratch, `${counter}-${name}`);
}

async function exists(path: string): Promise<boolean> {
    try {
        await stat(path);

        return true;
    } catch {
        return false;
    }
}

async function catchDownload(promise: Promise<void>): Promise<DownloadError> {
    try {
        await promise;
    } catch (error) {
        expect(error).toBeInstanceOf(DownloadError);

        return error as DownloadError;
    }

    throw new Error("expected the download to fail");
}

describe("isAllowedDownloadUrl", () => {
    test("accepts the CDN hosts the installer needs", () => {
        const allowed = [
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30",
            "https://github.com/ggml-org/llama.cpp/releases/download/b12686/llama-b12686-bin-win-cpu-x64.zip",
            "https://objects.githubusercontent.com/github-production-release-asset/x.zip",
            "https://release-assets.githubusercontent.com/x.zip",
            "https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/model.gguf",
            "https://cdn-lfs.huggingface.co/repos/x/y",
            "https://cas-bridge.xethub.hf.co/x",
            "https://cdn-lfs-us-1.hf.co/x",
        ];

        for (const url of allowed) {
            expect(isAllowedDownloadUrl(url)).toBe(true);
        }
    });

    test("rejects lookalike hosts, plain http and garbage", () => {
        const rejected = [
            "http://github.com/x.zip",
            "http://api.github.com/repos",
            "https://github.com.evil.io/x.zip",
            "https://evilgithubusercontent.com/x.zip",
            "https://githubusercontent.com/x.zip",
            "https://huggingface.co.evil.io/x.gguf",
            "https://not-hf.co/x",
            // `.hf.co` is only allowed as a suffix (subdomains), so the bare short domain is not.
            "https://hf.co/unsloth/embeddinggemma-2-GGUF",
            "https://api.github.com@evil.io/x.zip",
            "ftp://github.com/x.zip",
            "file:///C:/x.zip",
            "github.com/x.zip",
            "",
        ];

        for (const url of rejected) {
            expect(isAllowedDownloadUrl(url)).toBe(false);
        }
    });
});

describe("downloadFile", () => {
    test("downloads, verifies size and sha256, then renames into place", async () => {
        const dest = tempPath("full.bin");
        const progress: { received: number; total: number }[] = [];

        await downloadFile({
            url: `${baseUrl}/file`,
            dest,
            expectedBytes: BODY.length,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
            onProgress: (update) => progress.push(update),
        });

        expect((await readFile(dest)).equals(Buffer.from(BODY))).toBe(true);
        expect(await exists(`${dest}.part`)).toBe(false);
        expect(progress[progress.length - 1]).toEqual({ received: BODY.length, total: BODY.length });
    });

    test("replaces an existing destination file", async () => {
        const dest = tempPath("replace.bin");
        await writeFile(dest, "stale content");

        await downloadFile({
            url: `${baseUrl}/file`,
            dest,
            expectedBytes: BODY.length,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
        });

        expect((await readFile(dest)).equals(Buffer.from(BODY))).toBe(true);
    });

    test("reports a write error instead of crashing on the unhandled stream 'error' event", async () => {
        const dir = tempPath("eisdir");
        const dest = join(dir, "file.bin");

        // A directory where the .part file should go makes createWriteStream fail with EISDIR; the
        // point of the test is that the failure surfaces as a DownloadError and not as an
        // unhandled 'error' event that kills the server.
        await mkdir(join(dir, "file.bin.part"), { recursive: true });

        const error = await catchDownload(
            downloadFile({ url: `${baseUrl}/file`, dest, expectedBytes: BODY.length, allowUrl: allowLocal }),
        );

        expect(error.code).toBe("network");
        expect(error.message).toContain("could not write");
        expect(error.message).toMatch(/file\.bin\.part \(E[A-Z]+\)/);

        // Still alive: a normal download works right after the failure.
        const survivor = tempPath("after-write-error.bin");
        await downloadFile({
            url: `${baseUrl}/file`,
            dest: survivor,
            expectedBytes: BODY.length,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
        });
        expect((await readFile(survivor)).equals(Buffer.from(BODY))).toBe(true);
    });

    test("resumes a truncated .part with a Range request", async () => {
        const dest = tempPath("resume.bin");
        const alreadyThere = BODY.slice(0, 120_000);
        await writeFile(`${dest}.part`, alreadyThere);
        bookkeeping.lastRange = null;

        await downloadFile({
            url: `${baseUrl}/file`,
            dest,
            expectedBytes: BODY.length,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
        });

        expect(lastRange()).toBe("bytes=120000-");
        expect((await readFile(dest)).equals(Buffer.from(BODY))).toBe(true);
    });

    test("treats a 416 as 'already complete' and keeps the existing .part", async () => {
        const dest = tempPath("complete.bin");
        await writeFile(`${dest}.part`, BODY);
        bookkeeping.lastRange = null;

        // No expectedBytes on purpose: without a known size the downloader has to ask the
        // server, and the 416 with `bytes */N` is what tells it the .part is complete.
        await downloadFile({
            url: `${baseUrl}/file`,
            dest,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
        });

        expect(lastRange()).toBe("bytes=300000-");
        expect((await readFile(dest)).equals(Buffer.from(BODY))).toBe(true);
    });

    test("restarts from zero when the server ignores Range", async () => {
        const dest = tempPath("norange.bin");
        await writeFile(`${dest}.part`, BODY.slice(0, 5_000));
        bookkeeping.lastRange = null;

        await downloadFile({
            url: `${baseUrl}/norange`,
            dest,
            expectedBytes: BODY.length,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
        });

        expect(lastRange()).toBe("bytes=5000-");
        expect((await readFile(dest)).equals(Buffer.from(BODY))).toBe(true);
    });

    test("fails on a size mismatch and removes the .part", async () => {
        const dest = tempPath("size.bin");
        const error = await catchDownload(
            downloadFile({ url: `${baseUrl}/file`, dest, expectedBytes: BODY.length + 10, allowUrl: allowLocal }),
        );

        expect(error.code).toBe("size-mismatch");
        expect(await exists(dest)).toBe(false);
        expect(await exists(`${dest}.part`)).toBe(false);
    });

    test("fails on a hash mismatch and removes the .part", async () => {
        const dest = tempPath("hash.bin");
        const error = await catchDownload(
            downloadFile({
                url: `${baseUrl}/file`,
                dest,
                expectedBytes: BODY.length,
                sha256: "0".repeat(64),
                allowUrl: allowLocal,
            }),
        );

        expect(error.code).toBe("hash-mismatch");
        expect(await exists(`${dest}.part`)).toBe(false);
    });

    test("aborts mid-stream, keeps the .part and resumes it", async () => {
        const dest = tempPath("abort.bin");
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 100);

        const error = await catchDownload(
            downloadFile({
                url: `${baseUrl}/slow`,
                dest,
                expectedBytes: BODY.length,
                allowUrl: allowLocal,
                signal: controller.signal,
            }),
        );

        clearTimeout(timer);
        expect(error.code).toBe("aborted");

        const partialSize = (await stat(`${dest}.part`)).size;
        expect(partialSize).toBeGreaterThan(0);
        expect(await exists(dest)).toBe(false);

        await downloadFile({
            url: `${baseUrl}/slow`,
            dest,
            expectedBytes: BODY.length,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
        });

        expect((await readFile(dest)).equals(Buffer.from(BODY))).toBe(true);
    }, 20_000);

    test("refuses a redirect to a foreign host without touching the network twice", async () => {
        const dest = tempPath("redirect-foreign.bin");
        const error = await catchDownload(
            downloadFile({ url: `${baseUrl}/redirect-foreign`, dest, allowUrl: allowLocal }),
        );

        expect(error.code).toBe("host-not-allowed");
        expect(await exists(dest)).toBe(false);
        expect(await exists(`${dest}.part`)).toBe(false);
    });

    test("follows a chain of redirects on an allowed host", async () => {
        const dest = tempPath("redirect-chain.bin");

        await downloadFile({
            url: `${baseUrl}/redirect-a`,
            dest,
            expectedBytes: BODY.length,
            sha256: BODY_SHA,
            allowUrl: allowLocal,
        });

        expect((await readFile(dest)).equals(Buffer.from(BODY))).toBe(true);
    });

    test("reports an http error for 404", async () => {
        const dest = tempPath("missing.bin");
        const error = await catchDownload(downloadFile({ url: `${baseUrl}/missing`, dest, allowUrl: allowLocal }));

        expect(error.code).toBe("http");
        expect(error.message).toContain("404");
    });

    test("refuses the default host check for an http URL (tests must opt in)", async () => {
        const dest = tempPath("default.bin");
        const error = await catchDownload(downloadFile({ url: `${baseUrl}/file`, dest }));

        expect(error.code).toBe("host-not-allowed");
        expect(bookkeeping.requestCount).toBeGreaterThan(0);
    });
});

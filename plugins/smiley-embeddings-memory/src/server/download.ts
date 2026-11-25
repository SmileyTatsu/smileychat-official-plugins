// Resumable, verified file download used by the binary installer and the model downloader.
//
// Two properties matter more than speed here:
//   * the URL allow list is re-checked on every redirect hop, so a GitHub redirect can never
//     move a download to an arbitrary host;
//   * the body is streamed into `<dest>.part` and renamed only after the size and sha256 match,
//     so an interrupted or tampered download never leaves a file that looks valid.

import { createWriteStream } from "node:fs";
import { mkdir, rename, stat, unlink } from "node:fs/promises";
import { dirname } from "node:path";

export type DownloadErrorCode = "host-not-allowed" | "http" | "size-mismatch" | "hash-mismatch" | "aborted" | "network";

export class DownloadError extends Error {
    code: DownloadErrorCode;

    constructor(code: DownloadErrorCode, message: string) {
        super(message);
        this.name = "DownloadError";
        this.code = code;
    }
}

export type DownloadProgress = { received: number; total: number };

export type DownloadOptions = {
    url: string;
    dest: string;
    expectedBytes?: number;
    sha256?: string;
    signal?: AbortSignal;
    onProgress?: (progress: DownloadProgress) => void;
    fetchImpl?: typeof fetch;
    allowUrl?: (url: string) => boolean;
};

const ALLOWED_HOSTS = new Set(["api.github.com", "github.com", "huggingface.co"]);
const ALLOWED_HOST_SUFFIXES = [".githubusercontent.com", ".hf.co", ".huggingface.co"];
const MAX_REDIRECTS = 5;
const PROGRESS_INTERVAL_MS = 250;

export function isAllowedDownloadUrl(url: string): boolean {
    let parsed: URL;

    try {
        parsed = new URL(url);
    } catch {
        return false;
    }

    if (parsed.protocol !== "https:") {
        return false;
    }

    // Credentials in the URL would be forwarded to whatever host they name; nothing we download
    // needs them, and they make the visible host ambiguous.
    if (parsed.username || parsed.password) {
        return false;
    }

    const host = parsed.hostname.toLowerCase();

    if (ALLOWED_HOSTS.has(host)) {
        return true;
    }

    // Suffix matching requires the leading dot, so `evilgithubusercontent.com` (no dot before
    // the suffix) and `github.com.evil.io` (suffix is the evil domain) are both rejected.
    return ALLOWED_HOST_SUFFIXES.some((suffix) => host.endsWith(suffix));
}

async function fileSize(path: string): Promise<number> {
    try {
        const info = await stat(path);

        return info.isFile() ? info.size : 0;
    } catch {
        return 0;
    }
}

async function removeFile(path: string): Promise<void> {
    await unlink(path).catch(() => undefined);
}

function contentLength(response: Response): number | undefined {
    const value = response.headers.get("content-length");

    if (!value) {
        return undefined;
    }

    const length = Number(value);

    return Number.isFinite(length) && length >= 0 ? length : undefined;
}

type ContentRange = { start?: number; total?: number };

function parseContentRange(value: string | null): ContentRange | undefined {
    if (!value) {
        return undefined;
    }

    // `bytes 100-199/1000` for a partial answer, `bytes */1000` for a 416.
    const match = /^bytes\s+(?:(\d+)-\d+|\*)\/(\d+|\*)$/.exec(value.trim());

    if (!match) {
        return undefined;
    }

    const total = match[2] === "*" ? undefined : Number(match[2]);

    return { start: match[1] === undefined ? undefined : Number(match[1]), total };
}

function parseContentRangeTotal(value: string | null): number | undefined {
    return parseContentRange(value)?.total;
}

function isAbortError(error: unknown): boolean {
    return error instanceof Error && (error.name === "AbortError" || error.name === "TimeoutError");
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

// Node errors carry the errno in `code`; the operator needs it to tell a full disk from a file
// that an antivirus scan still holds open.
function writeErrorMessage(path: string, error: Error): string {
    const code = (error as { code?: string }).code;

    return `could not write ${path}${code ? ` (${code})` : ""}: ${error.message}`;
}

type RedirectResult = { response: Response; url: string };

async function fetchWithRedirects(
    url: string,
    headers: Record<string, string>,
    fetchImpl: typeof fetch,
    allowUrl: (url: string) => boolean,
    signal?: AbortSignal,
): Promise<RedirectResult> {
    let current = url;

    for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
        // Checked before every hop: `redirect: "manual"` means we decide where to go next, and
        // the Range header would otherwise be replayed to a host we never approved.
        if (!allowUrl(current)) {
            throw new DownloadError("host-not-allowed", `refusing to download from ${current}`);
        }

        const response = await fetchImpl(current, { method: "GET", headers, redirect: "manual", signal });

        if (response.status < 300 || response.status >= 400 || response.status === 304) {
            return { response, url: current };
        }

        const location = response.headers.get("location");
        await response.body?.cancel().catch(() => undefined);

        if (!location) {
            throw new DownloadError("http", `HTTP ${response.status} without a Location header from ${current}`);
        }

        current = new URL(location, current).toString();
    }

    throw new DownloadError("http", `too many redirects starting at ${url}`);
}

async function writeToPart(
    body: ReadableStream<Uint8Array>,
    partPath: string,
    append: boolean,
    startBytes: number,
    total: number,
    onProgress: ((progress: DownloadProgress) => void) | undefined,
    signal: AbortSignal | undefined,
): Promise<number> {
    const sink = createWriteStream(partPath, { flags: append ? "a" : "w" });
    const reader = body.getReader();
    let written = startBytes;
    let lastReport = 0;
    let failure: Error | null = null;

    // Without this listener an asynchronous write error (ENOSPC, or EPERM when Windows keeps the
    // file locked) is emitted as an 'error' event nobody handles, which takes down the whole Bun
    // process. Recording it lets the loop below reject with a DownloadError instead.
    sink.on("error", (error: Error) => {
        failure ??= error;
    });

    const throwIfFailed = () => {
        if (failure) {
            throw new DownloadError("network", writeErrorMessage(partPath, failure));
        }
    };

    // A stream that errored never drains again, so waiting for 'drain' alone can hang forever:
    // settle on 'error'/'close' too, and check for an already-recorded failure first.
    const waitFor = (event: "drain" | "finish"): Promise<void> =>
        new Promise<void>((resolve, reject) => {
            const onEvent = () => {
                cleanup();
                resolve();
            };
            const onError = (error: Error) => {
                cleanup();
                reject(new DownloadError("network", writeErrorMessage(partPath, error)));
            };
            const onClose = () => {
                cleanup();
                reject(new DownloadError("network", writeErrorMessage(partPath, failure ?? new Error("stream closed"))));
            };
            const cleanup = () => {
                sink.off(event, onEvent);
                sink.off("error", onError);
                sink.off("close", onClose);
            };

            if (failure) {
                reject(new DownloadError("network", writeErrorMessage(partPath, failure)));

                return;
            }

            sink.once(event, onEvent);
            sink.once("error", onError);
            sink.once("close", onClose);
        });

    const report = (force: boolean) => {
        if (!onProgress) {
            return;
        }

        const now = Date.now();

        if (!force && now - lastReport < PROGRESS_INTERVAL_MS) {
            return;
        }

        lastReport = now;
        onProgress({ received: written, total });
    };

    report(true);

    try {
        for (;;) {
            const { done, value } = await reader.read();
            throwIfFailed();

            if (done) {
                break;
            }

            if (signal?.aborted) {
                await reader.cancel().catch(() => undefined);
                throw new DownloadError("aborted", "download aborted");
            }

            if (!value || value.length === 0) {
                continue;
            }

            written += value.length;

            if (!sink.write(value)) {
                await waitFor("drain");
            }

            throwIfFailed();
            report(false);
        }

        // end() has no callback error channel: the 'error' event is the only signal that the
        // flush failed, hence the explicit listener in waitFor.
        const finishing = waitFor("finish");
        sink.end();
        await finishing;
    } catch (error) {
        sink.destroy();
        await reader.cancel().catch(() => undefined);
        throw error;
    }

    report(true);

    return written;
}

async function sha256File(path: string): Promise<string> {
    const hasher = new Bun.CryptoHasher("sha256");
    const reader = Bun.file(path).stream().getReader();

    for (;;) {
        const { done, value } = await reader.read();

        if (done) {
            break;
        }

        if (value) {
            hasher.update(value);
        }
    }

    return hasher.digest("hex");
}

export async function downloadFile(opts: DownloadOptions): Promise<void> {
    const allowUrl = opts.allowUrl ?? isAllowedDownloadUrl;
    const fetchImpl = opts.fetchImpl ?? fetch;
    const partPath = `${opts.dest}.part`;

    await mkdir(dirname(opts.dest), { recursive: true });

    let onDisk = await fileSize(partPath);
    let total = opts.expectedBytes ?? 0;

    try {
        if (opts.signal?.aborted) {
            throw new DownloadError("aborted", "download aborted");
        }

        // A .part that already holds the expected size needs no request at all; it still has to
        // pass the hash check further down.
        const alreadyComplete = onDisk > 0 && opts.expectedBytes !== undefined && onDisk === opts.expectedBytes;

        if (!alreadyComplete) {
            const headers: Record<string, string> = {};

            if (onDisk > 0) {
                headers.Range = `bytes=${onDisk}-`;
            }

            const { response, url } = await fetchWithRedirects(opts.url, headers, fetchImpl, allowUrl, opts.signal);
            const status = response.status;

            if (status === 416 && onDisk > 0) {
                // Nothing left to send: the .part already covers the whole file.
                total = parseContentRangeTotal(response.headers.get("content-range")) ?? opts.expectedBytes ?? onDisk;
                await response.body?.cancel().catch(() => undefined);
            } else if (status === 206) {
                const range = parseContentRange(response.headers.get("content-range"));

                if (range?.start !== onDisk) {
                    throw new DownloadError(
                        "network",
                        `server resumed at byte ${range?.start ?? "?"} instead of ${onDisk} (${url})`,
                    );
                }

                const length = contentLength(response);
                total = range?.total ?? (length === undefined ? 0 : onDisk + length);

                if (!response.body) {
                    throw new DownloadError("network", `empty response body from ${url}`);
                }

                onDisk = await writeToPart(response.body, partPath, true, onDisk, total, opts.onProgress, opts.signal);
            } else if (status === 200) {
                const length = contentLength(response);

                // The server ignored the Range header: start the file over instead of appending
                // a second copy of it.
                total = length ?? opts.expectedBytes ?? 0;

                if (!response.body) {
                    throw new DownloadError("network", `empty response body from ${url}`);
                }

                onDisk = await writeToPart(response.body, partPath, false, 0, total, opts.onProgress, opts.signal);
            } else {
                await response.body?.cancel().catch(() => undefined);
                throw new DownloadError("http", `HTTP ${status} for ${url}`);
            }
        }

        const expected = opts.expectedBytes ?? (total > 0 ? total : undefined);

        if (expected !== undefined && onDisk !== expected) {
            await removeFile(partPath);
            throw new DownloadError("size-mismatch", `expected ${expected} bytes but the file has ${onDisk}`);
        }

        if (opts.sha256) {
            const actual = await sha256File(partPath);
            const wanted = opts.sha256.trim().toLowerCase();

            if (actual !== wanted) {
                await removeFile(partPath);
                throw new DownloadError("hash-mismatch", `sha256 mismatch: expected ${wanted}, got ${actual}`);
            }
        }

        // Rename last: only a verified file ever appears under the real name.
        await rename(partPath, opts.dest);
    } catch (error) {
        if (error instanceof DownloadError) {
            throw error;
        }

        if (opts.signal?.aborted || isAbortError(error)) {
            throw new DownloadError("aborted", `download aborted; ${partPath} is kept for a resume`);
        }

        throw new DownloadError("network", `download failed: ${errorMessage(error)}`);
    }
}

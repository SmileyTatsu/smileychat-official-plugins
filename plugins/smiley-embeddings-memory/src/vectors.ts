/**
 * Vector math and the base64 float32 codec used by the index shards.
 *
 * Pure and browser-safe: encoding goes through `btoa` on Uint8Array slices, never
 * `Buffer`, so the same code runs in the Browser plugin bundle and in tests.
 */

const base64ChunkSize = 0x8000;

function bytesToBase64(bytes: Uint8Array) {
    let binary = "";

    // `btoa` takes a binary string; converting in slices keeps the call below the
    // argument limit for large (2560-d) vectors.
    for (let offset = 0; offset < bytes.length; offset += base64ChunkSize) {
        const slice = bytes.subarray(offset, offset + base64ChunkSize);
        binary += String.fromCharCode(...slice);
    }

    return btoa(binary);
}

/** Returns a unit-length copy of `v`; a zero vector stays zero. */
export function normalize(v: Float32Array): Float32Array {
    let sumSquares = 0;

    for (let i = 0; i < v.length; i += 1) sumSquares += v[i] * v[i];

    const out = new Float32Array(v.length);
    if (sumSquares === 0) return out;

    const scale = 1 / Math.sqrt(sumSquares);
    for (let i = 0; i < v.length; i += 1) out[i] = v[i] * scale;

    return out;
}

/** Dot product; also the cosine similarity when both vectors are unit length. */
export function dot(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
        throw new Error(`vector dimension mismatch: ${a.length} vs ${b.length}`);
    }

    let sum = 0;
    for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];

    return sum;
}

/** base64 of the little-endian float32 bytes (endianness is fixed, not host order). */
export function encodeVector(v: Float32Array): string {
    const bytes = new Uint8Array(v.length * 4);
    const view = new DataView(bytes.buffer);

    for (let i = 0; i < v.length; i += 1) view.setFloat32(i * 4, v[i], true);

    return bytesToBase64(bytes);
}

/** Decodes a vector written by `encodeVector`, into a fresh aligned Float32Array. */
export function decodeVector(b64: string, dim: number): Float32Array {
    let binary = "";
    try {
        binary = atob(b64);
    } catch {
        throw new Error("invalid base64 vector");
    }

    const expected = dim * 4;
    if (binary.length !== expected) {
        throw new Error(
            `vector byte length mismatch: expected ${expected} (dim ${dim}), got ${binary.length}`,
        );
    }

    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);

    const view = new DataView(bytes.buffer);
    const out = new Float32Array(dim);
    for (let i = 0; i < dim; i += 1) out[i] = view.getFloat32(i * 4, true);

    return out;
}

export type Ranked = { index: number; score: number };

export type RankOptions = {
    threshold: number;
    topK: number;
    skip?: (index: number) => boolean;
};

/**
 * Scores candidates against a query and keeps the best ones. All vectors are
 * expected to be unit length, so the dot product is already the cosine.
 */
export function rankVectors(
    query: Float32Array,
    vectors: Float32Array[],
    opts: RankOptions,
): { ranked: Ranked[]; belowThreshold: number; skipped: number; best: number } {
    const ranked: Ranked[] = [];
    let belowThreshold = 0;
    let skipped = 0;
    let best = -1;

    for (let index = 0; index < vectors.length; index += 1) {
        // Excluded candidates are not even scored, so they never distort `best`.
        if (opts.skip?.(index)) {
            skipped += 1;
            continue;
        }

        const score = dot(query, vectors[index]);
        if (score > best) best = score;

        if (score >= opts.threshold) {
            ranked.push({ index, score });
        } else {
            belowThreshold += 1;
        }
    }

    // Array#sort is stable, so equal scores keep their index order.
    ranked.sort((a, b) => b.score - a.score);

    const topK = Number.isFinite(opts.topK) ? Math.max(0, Math.floor(opts.topK)) : 0;

    return {
        ranked: ranked.slice(0, topK),
        belowThreshold,
        skipped,
        best,
    };
}

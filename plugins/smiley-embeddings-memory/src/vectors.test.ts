import { describe, expect, test } from "bun:test";

import { decodeVector, dot, encodeVector, normalize, rankVectors } from "./vectors";

function seeded(seed: number) {
    let state = seed;
    return () => {
        state = (state * 1103515245 + 12345) & 0x7fffffff;
        return state / 0x7fffffff;
    };
}

function randomVector(dim: number, seed: number) {
    const random = seeded(seed);
    const out = new Float32Array(dim);
    for (let i = 0; i < dim; i += 1) out[i] = Math.fround(random() * 2 - 1);
    return out;
}

function sameBits(a: Float32Array, b: Float32Array) {
    if (a.length !== b.length) return false;
    const left = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const right = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    for (let i = 0; i < left.length; i += 1) if (left[i] !== right[i]) return false;
    return true;
}

describe("normalize", () => {
    test("returns a unit-length copy", () => {
        const v = new Float32Array([3, 4]);
        const unit = normalize(v);

        expect(dot(unit, unit)).toBeCloseTo(1, 6);
        expect(unit[0]).toBeCloseTo(0.6, 6);
        expect(unit[1]).toBeCloseTo(0.8, 6);
    });

    test("does not mutate the input", () => {
        const v = new Float32Array([1, 2, 3]);
        const unit = normalize(v);

        expect(Array.from(v)).toEqual([1, 2, 3]);
        expect(unit).not.toBe(v);
    });

    test("keeps the zero vector zero", () => {
        const unit = normalize(new Float32Array(768));

        expect(unit.length).toBe(768);
        expect(unit.every((value) => value === 0)).toBe(true);
        expect(dot(unit, unit)).toBe(0);
    });
});

describe("dot", () => {
    test("computes the sum of products", () => {
        expect(dot(new Float32Array([1, 2, 3]), new Float32Array([4, 5, 6]))).toBe(32);
    });

    test("throws on a dimension mismatch", () => {
        expect(() => dot(new Float32Array([1, 2]), new Float32Array([1, 2, 3]))).toThrow(
            "dimension mismatch",
        );
    });
});

describe("encodeVector / decodeVector", () => {
    test("round trips bit-exactly", () => {
        for (const dim of [1, 3, 17, 768, 2560]) {
            const vector = randomVector(dim, dim);
            const decoded = decodeVector(encodeVector(vector), dim);

            expect(decoded.length).toBe(dim);
            expect(sameBits(decoded, vector)).toBe(true);
            for (let i = 0; i < dim; i += 1) expect(Object.is(decoded[i], vector[i])).toBe(true);
        }
    });

    test("returns a fresh, aligned Float32Array", () => {
        const decoded = decodeVector(encodeVector(new Float32Array([1, 2, 3])), 3);

        expect(decoded.byteOffset).toBe(0);
        expect(decoded.buffer.byteLength).toBe(12);
    });

    test("writes little-endian float32 bytes", () => {
        const bytes = new Uint8Array(
            Array.from(atob(encodeVector(new Float32Array([1]))), (c) => c.charCodeAt(0)),
        );

        expect(Array.from(bytes)).toEqual([0, 0, 0x80, 0x3f]);
    });

    test("computes the base64 size from the element count", () => {
        expect(encodeVector(new Float32Array(0))).toBe("");
        expect(encodeVector(new Float32Array(2560)).length).toBe(13656);
    });

    test("throws when the byte length does not match the dimension", () => {
        const b64 = encodeVector(new Float32Array([1, 2, 3, 4]));

        expect(() => decodeVector(b64, 3)).toThrow("byte length mismatch");
        expect(() => decodeVector(b64, 5)).toThrow("byte length mismatch");
        expect(() => decodeVector("", 1)).toThrow("byte length mismatch");
    });

    test("throws on invalid base64", () => {
        expect(() => decodeVector("not base64!!", 3)).toThrow("invalid base64");
    });
});

describe("rankVectors", () => {
    const query = new Float32Array([1, 0]);
    const vectors = [
        new Float32Array([1, 0]),
        new Float32Array([0, 1]),
        new Float32Array([0.8, 0.6]),
        new Float32Array([-1, 0]),
    ];

    test("keeps matches above the threshold sorted by score", () => {
        const result = rankVectors(query, vectors, { threshold: 0.5, topK: 5 });

        expect(result.ranked.map((entry) => entry.index)).toEqual([0, 2]);
        expect(result.ranked[0].score).toBeCloseTo(1, 6);
        expect(result.belowThreshold).toBe(2);
        expect(result.skipped).toBe(0);
        expect(result.best).toBeCloseTo(1, 6);
    });

    test("honours topK", () => {
        const result = rankVectors(query, vectors, { threshold: -1, topK: 2 });

        expect(result.ranked.length).toBe(2);
        expect(result.ranked.map((entry) => entry.index)).toEqual([0, 2]);
        expect(rankVectors(query, vectors, { threshold: -1, topK: 0 }).ranked).toEqual([]);
    });

    test("skips candidates before scoring", () => {
        const skippedIndexes: number[] = [];
        const result = rankVectors(query, vectors, {
            threshold: 0.5,
            topK: 5,
            skip(index) {
                skippedIndexes.push(index);
                return index === 0 || index === 3;
            },
        });

        expect(skippedIndexes).toEqual([0, 1, 2, 3]);
        expect(result.ranked.map((entry) => entry.index)).toEqual([2]);
        expect(result.skipped).toBe(2);
        // The excluded best vector does not count as the best score.
        expect(result.best).toBeCloseTo(0.8, 6);
        expect(result.belowThreshold).toBe(1);
    });

    test("reports best -1 when nothing was scored", () => {
        const result = rankVectors(query, vectors, {
            threshold: 0.5,
            topK: 5,
            skip: () => true,
        });

        expect(result).toEqual({ ranked: [], belowThreshold: 0, skipped: 4, best: -1 });
    });

    test("reports best -1 for an empty index", () => {
        const result = rankVectors(query, [], { threshold: 0.5, topK: 5 });

        expect(result).toEqual({ ranked: [], belowThreshold: 0, skipped: 0, best: -1 });
    });

    test("counts everything below a high threshold", () => {
        const result = rankVectors(query, vectors, { threshold: 1.5, topK: 3 });

        expect(result.ranked).toEqual([]);
        expect(result.belowThreshold).toBe(4);
        expect(result.best).toBeCloseTo(1, 6);
    });
});

import { describe, expect, test } from "bun:test";

import { chunkText, cleanMessageText, hashText } from "./chunking";

function seeded(seed: number) {
    let state = seed;
    return () => {
        state = (state * 1103515245 + 12345) & 0x7fffffff;
        return state / 0x7fffffff;
    };
}

function randomText(length: number, seed: number) {
    const random = seeded(seed);
    const words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];
    let out = "";

    while (out.length < length) {
        const word = words[Math.floor(random() * words.length)];
        const roll = random();
        out += roll > 0.9 ? "\n\n" : roll > 0.78 ? "\n" : " ";
        out += word;
    }

    return out.slice(0, length);
}

describe("chunkText", () => {
    test("never exceeds the size bound on long random text", () => {
        const text = randomText(20000, 7);
        const chunks = chunkText(text, { size: 300, overlap: 40 });

        expect(chunks.length).toBeGreaterThan(1);
        for (const chunk of chunks) {
            expect(chunk.text.length).toBeLessThanOrEqual(300);
            expect(chunk.text.trim().length).toBeGreaterThan(0);
        }
    });

    test("returns a single trimmed chunk when the text fits", () => {
        const chunks = chunkText("  short message  ", { size: 100, overlap: 10 });

        expect(chunks).toEqual([{ text: "short message", start: 2 }]);
    });

    test("still returns trimmed chunks when size equals the text length", () => {
        const chunks = chunkText("exactly fits here", { size: 17, overlap: 4 });

        expect(chunks).toEqual([{ text: "exactly fits here", start: 0 }]);
    });

    test("returns nothing for empty or whitespace-only input", () => {
        expect(chunkText("", { size: 50, overlap: 0 })).toEqual([]);
        expect(chunkText("   \n\n  ", { size: 50, overlap: 0 })).toEqual([]);
        expect(chunkText("   \n\n  ".repeat(30), { size: 10, overlap: 2 })).toEqual([]);
    });

    test("prefers a paragraph break over a line break and a space", () => {
        const text = `${"a".repeat(60)}\n\n${"b".repeat(60)}\n${"c".repeat(20)} ${"d".repeat(20)}`;
        const chunks = chunkText(text, { size: 70, overlap: 0 });

        expect(chunks[0].text).toBe("a".repeat(60));
        expect(chunks[0].text).not.toContain("b");
    });

    test("prefers a line break when no paragraph break is available", () => {
        const text = `${"a".repeat(60)}\n${"b".repeat(60)} ${"c".repeat(20)}`;
        const chunks = chunkText(text, { size: 70, overlap: 0 });

        expect(chunks[0].text).toBe("a".repeat(60));
    });

    test("prefers a space when no line break is available", () => {
        const text = `${"a".repeat(60)} ${"b".repeat(60)}`;
        const chunks = chunkText(text, { size: 70, overlap: 0 });

        expect(chunks[0].text).toBe("a".repeat(60));
        expect(chunks[1].text).toBe("b".repeat(60));
    });

    test("hard-cuts a long unbroken string and terminates", () => {
        const chunks = chunkText("x".repeat(10000), { size: 200, overlap: 50 });

        expect(chunks.length).toBe(50);
        for (const chunk of chunks) expect(chunk.text.length).toBe(200);
        expect(chunks[chunks.length - 1].start).toBeLessThan(10000);
    });

    test("terminates when overlap is larger than the size", () => {
        const chunks = chunkText("word ".repeat(2000), { size: 20, overlap: 500 });

        expect(chunks.length).toBeGreaterThan(1);
        for (const chunk of chunks) expect(chunk.text.length).toBeLessThanOrEqual(20);
    });

    test("terminates with a size of one", () => {
        const chunks = chunkText("abcdef", { size: 1, overlap: 0 });

        expect(chunks.map((chunk) => chunk.text).join("")).toBe("abcdef");
    });

    test("repeats roughly `overlap` characters at the start of the next chunk", () => {
        const text = "one two three four five six seven eight nine ten eleven twelve";
        const chunks = chunkText(text, { size: 30, overlap: 15 });

        expect(chunks.length).toBeGreaterThan(1);
        for (let i = 1; i < chunks.length; i += 1) {
            const previousEnd = chunks[i - 1].start + chunks[i - 1].text.length;
            expect(chunks[i].start).toBeLessThan(previousEnd);
            expect(chunks[i].start).toBeGreaterThanOrEqual(previousEnd - 15);
        }
        // The overlap is snapped to a word start, so no word is cut in half.
        for (const chunk of chunks) expect(chunk.text).not.toMatch(/^\w+\b \w*$/);
    });

    test("does not overlap when overlap is zero", () => {
        const chunks = chunkText("word ".repeat(100), { size: 40, overlap: 0 });

        for (let i = 1; i < chunks.length; i += 1) {
            const previousEnd = chunks[i - 1].start + chunks[i - 1].text.length;
            expect(chunks[i].start).toBeGreaterThanOrEqual(previousEnd);
        }
    });

    test("keeps offsets pointing into the input text", () => {
        const text = randomText(5000, 11);

        for (const chunk of chunkText(text, { size: 250, overlap: 30 })) {
            const slice = text.slice(chunk.start, chunk.start + chunk.text.length);
            expect(slice).toBe(chunk.text);
        }
    });

    test("never splits a surrogate pair", () => {
        const text = "😀".repeat(500);
        const chunks = chunkText(text, { size: 7, overlap: 0 });

        expect(chunks.length).toBeGreaterThan(1);
        for (const chunk of chunks) {
            expect(chunk.text).not.toMatch(/^[\uDC00-\uDFFF]/);
            expect(chunk.text).not.toMatch(/[\uD800-\uDBFF]$/);
            expect(chunk.text.length % 2).toBe(0);
        }
        expect(chunks.map((chunk) => chunk.text).join("")).toBe(text);
    });

    test("terminates on emoji even with a size of one", () => {
        const text = "😀".repeat(50);
        const chunks = chunkText(text, { size: 1, overlap: 0 });

        expect(chunks.length).toBe(100);
        expect(chunks.map((chunk) => chunk.text).join("")).toBe(text);
    });

    test("keeps mixed unicode text intact", () => {
        const text = "héllo wörld 🎉 ".repeat(100);
        const chunks = chunkText(text, { size: 40, overlap: 8 });

        for (const chunk of chunks) {
            expect(chunk.text.length).toBeLessThanOrEqual(40);
            expect(chunk.text).not.toMatch(/^[\uDC00-\uDFFF]/);
            expect(chunk.text).not.toMatch(/[\uD800-\uDBFF]$/);
            expect(Number.isNaN(chunk.text.charCodeAt(0))).toBe(false);
        }
    });

    test("falls back to a default size for invalid options", () => {
        const text = "word ".repeat(500);
        const chunks = chunkText(text, { size: Number.NaN, overlap: Number.NaN });

        expect(chunks.length).toBeGreaterThan(1);
        for (const chunk of chunks) expect(chunk.text.length).toBeLessThanOrEqual(800);
    });
});

describe("cleanMessageText", () => {
    test("turns msg bubbles into their content separated by a blank line", () => {
        const raw = '<msg>Hey!</msg><msg delay="1.5s">Did you hear?</msg>';

        expect(cleanMessageText(raw)).toBe("Hey!\n\nDid you hear?");
    });

    test("keeps the content of a single bubble with attributes", () => {
        expect(cleanMessageText('<msg delay="500ms">only one</msg>')).toBe("only one");
    });

    test("strips photo markers", () => {
        const raw = "Look {{photo}} and {{photo[0]}} and {{photo:id=abc-123}} here";

        expect(cleanMessageText(raw)).toBe("Look  and  and  here");
    });

    test("does not strip unrelated photo-looking text", () => {
        expect(cleanMessageText("{{photo_caption}} stays")).toBe("{{photo_caption}} stays");
    });

    test("survives unclosed and odd tags", () => {
        expect(cleanMessageText('<msg delay="1s">unclosed text')).toBe("unclosed text");
        expect(cleanMessageText("orphan </msg> closer")).toBe("orphan  closer");
        expect(cleanMessageText("<msg></msg>")).toBe("");
    });

    test("collapses runs of blank lines and trims", () => {
        const raw = "\n\n\nfirst\n\n\n\n\nsecond\n\n\n";

        expect(cleanMessageText(raw)).toBe("first\n\nsecond");
    });

    test("is a no-op on plain text and on empty input", () => {
        expect(cleanMessageText("plain message")).toBe("plain message");
        expect(cleanMessageText("")).toBe("");
    });
});

describe("hashText", () => {
    test("is stable across calls and known to be a 14-char hex string", () => {
        const hash = hashText("hello world");

        expect(hash).toBe(hashText("hello world"));
        expect(hash).toMatch(/^[0-9a-f]{14}$/);
    });

    test("differs for different text, including tiny edits", () => {
        expect(hashText("hello world")).not.toBe(hashText("hello worlD"));
        expect(hashText("a")).not.toBe(hashText("ab"));
        expect(hashText("")).not.toBe(hashText(" "));
    });

    test("handles long and unicode text", () => {
        const text = "😀 héllo ".repeat(5000);

        expect(hashText(text)).toBe(hashText(text));
        expect(hashText(text)).not.toBe(hashText(`${text} `));
    });
});

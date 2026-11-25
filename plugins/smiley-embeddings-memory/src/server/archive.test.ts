import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { gzipSync, zipSync } from "fflate";

import { extractArchive } from "./archive";

let scratch = "";
let counter = 0;

beforeAll(async () => {
    scratch = await mkdtemp(join(tmpdir(), "smiley-archive-"));
});

afterAll(async () => {
    await rm(scratch, { recursive: true, force: true });
});

async function file(name: string, data: Uint8Array): Promise<string> {
    counter += 1;

    const path = join(scratch, `${counter}-${name}`);

    await writeFile(path, data);

    return path;
}

describe("extractArchive", () => {
    test("routes .zip to the zip extractor", async () => {
        const archive = await file("build.zip", zipSync({ "a.txt": new TextEncoder().encode("zip") }));
        const out = join(scratch, "zip-out");

        expect(await extractArchive(archive, out)).toEqual(["a.txt"]);
        expect(new TextDecoder().decode(await readFile(join(out, "a.txt")))).toBe("zip");
    });

    test("routes .tar.gz and .tgz to the tar extractor", async () => {
        // Two zero blocks: a valid, empty tar archive.
        const emptyTar = new Uint8Array(1024);

        const tarGz = await file("build.tar.gz", gzipSync(emptyTar));
        const tgz = await file("build.tgz", gzipSync(emptyTar));

        expect(await extractArchive(tarGz, join(scratch, "tar-out"))).toEqual([]);
        expect(await extractArchive(tgz, join(scratch, "tgz-out"))).toEqual([]);
    });

    test("matches extensions case-insensitively", async () => {
        const zip = await file("BUILD.ZIP", zipSync({ "b.txt": new TextEncoder().encode("case") }));
        const tar = await file("BUILD.TAR.GZ", gzipSync(new Uint8Array(1024)));

        expect(await extractArchive(zip, join(scratch, "upper-zip"))).toEqual(["b.txt"]);
        expect(await extractArchive(tar, join(scratch, "upper-tar"))).toEqual([]);
    });

    test("rejects an unsupported archive type", async () => {
        for (const name of ["build.7z", "build.tar", "build", "build.tar.bz2"]) {
            const archive = await file(name, new Uint8Array(8));

            await expect(extractArchive(archive, join(scratch, "nope"))).rejects.toThrow(/unsupported archive type/);
        }
    });
});

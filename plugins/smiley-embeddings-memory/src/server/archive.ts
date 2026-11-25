// Picks the extractor by file extension: llama.cpp publishes Windows builds as .zip and every
// other platform as .tar.gz, and the downloader keeps the asset's own name.

import { extractTarGz } from "./untar";
import { extractZip, type ExtractZipOptions } from "./unzip";

export type ArchiveOptions = ExtractZipOptions;

export async function extractArchive(
    archivePath: string,
    destDir: string,
    opts: ArchiveOptions = {},
): Promise<string[]> {
    const lower = archivePath.toLowerCase();

    if (lower.endsWith(".zip")) {
        return extractZip(archivePath, destDir, opts);
    }

    if (lower.endsWith(".tar.gz") || lower.endsWith(".tgz")) {
        return extractTarGz(archivePath, destDir, opts);
    }

    throw new Error(`unsupported archive type: ${archivePath} (expected .zip, .tar.gz or .tgz)`);
}

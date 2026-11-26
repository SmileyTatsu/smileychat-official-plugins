/**
 * Builds `<id>-<version>.zip` for a plugin folder.
 *
 * The ZIP is what the registry points at: a root `plugin.json`, the built entry,
 * and any styles. Sources, lockfiles and the types copy stay out, so the artifact
 * only contains what the app loads.
 */
import { mkdir, rm } from "node:fs/promises";
import { join, resolve } from "node:path";

import AdmZip from "adm-zip";

const pluginDir = resolve(process.argv[2] ?? ".");
const manifest = (await Bun.file(join(pluginDir, "plugin.json")).json()) as {
    id?: string;
    version?: string;
    main?: string;
    styles?: string[];
};

if (!manifest.id || !manifest.version || !manifest.main) {
    console.error(`${pluginDir}/plugin.json needs id, version and main.`);
    process.exit(1);
}

const distDir = join(pluginDir, "dist");
const releaseDir = join(pluginDir, "release");
const zipPath = join(releaseDir, `${manifest.id}-${manifest.version}.zip`);

// Refuse to ship an artifact whose entry file was never built: the installer
// validates `main`, so a missing file turns into a confusing install error.
if (!(await Bun.file(join(distDir, "index.js")).exists())) {
    console.error(`Build ${manifest.id} first (bun run build): dist/index.js is missing.`);
    process.exit(1);
}

await rm(releaseDir, { recursive: true, force: true });
await mkdir(releaseDir, { recursive: true });

const zip = new AdmZip();
zip.addLocalFile(join(pluginDir, "plugin.json"), "", "plugin.json");
zip.addLocalFolder(distDir, "dist");

for (const style of manifest.styles ?? []) {
    if (!style.startsWith("dist/")) {
        console.error(`Only dist/ styles are shipped, found "${style}".`);
        process.exit(1);
    }
}

zip.writeZip(zipPath);

console.log(`Packed ${zipPath}`);
console.log(`Artifact name: ${manifest.id}-${manifest.version}.zip`);

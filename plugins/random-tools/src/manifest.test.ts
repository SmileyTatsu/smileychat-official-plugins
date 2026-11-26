import { describe, expect, test } from "bun:test";

import { randomToolsManifest } from "./manifest";

const manifestFile = (await Bun.file(new URL("../plugin.json", import.meta.url)).json()) as {
    id?: string;
    name?: string;
    version?: string;
    description?: string;
    main?: string;
    permissions?: string[];
    apiVersion?: number;
    category?: string;
};

/**
 * `plugin.json` is what the installer validates and what carries enable state;
 * `src/manifest.ts` is what the app reads at runtime. The repo README warns that
 * both must be bumped together, so the copy is checked instead of trusted.
 */
describe("plugin.json matches the runtime manifest", () => {
    test("identity, version and entry point agree", () => {
        expect(manifestFile.id).toBe(randomToolsManifest.id);
        expect(manifestFile.name).toBe(randomToolsManifest.name);
        expect(manifestFile.version).toBe(randomToolsManifest.version);
        expect(manifestFile.description).toBe(randomToolsManifest.description);
        expect(manifestFile.main).toBe(randomToolsManifest.main);
        expect(manifestFile.category).toBe(randomToolsManifest.category);
        expect(manifestFile.apiVersion).toBe(randomToolsManifest.apiVersion);
        expect(manifestFile.permissions).toEqual(randomToolsManifest.permissions);
    });

    test("declares only the tool permission", () => {
        expect(randomToolsManifest.permissions).toEqual(["tools:register"]);
    });
});

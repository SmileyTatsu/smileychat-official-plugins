import type { PluginManifest } from "../types/smileychat-plugin-api";

export const randomToolsManifest: PluginManifest = {
    id: "smiley-random-tools",
    name: "Smiley Random Tools",
    version: "1.0.0",
    description:
        "Gives the model fair randomness: weighted picks, dice rolls, shuffles and random numbers.",
    main: "dist/index.js",
    // Every tool is local arithmetic. No app state, no storage, no network.
    permissions: ["tools:register"],
    apiVersion: 1,
    category: "tools",
};

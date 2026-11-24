import type { PluginManifest } from "../types/smileychat-plugin-api";

export const postProcessingManifest: PluginManifest = {
    id: "smiley-post-processing",
    name: "Smiley Post Processing",
    version: "1.0.0",
    description:
        "Runs a pipeline of AI passes that rewrite each reply before it is saved.",
    main: "dist/index.js",
    permissions: [
        "state:read",
        "actions",
        "model:generate",
        "ui:settings",
        "ui:modals",
        "ui:styles",
        "ui:message-actions",
        "chat:output",
        "presets:macros",
        "presets:compile",
        "storage"
],
    apiVersion: 1,
    category: "input-output",
};

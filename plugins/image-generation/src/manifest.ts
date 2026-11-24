import type { PluginManifest } from "../types/smileychat-plugin-api";

export const imageGenerationManifest: PluginManifest = {
    id: "smiley-image-generation",
    name: "Smiley Image Generation",
    version: "1.0.0",
    description:
        "Generates NovelAI images from chat context and attaches them to messages.",
    main: "dist/index.js",
    permissions: [
        "state:read",
        "model:generate",
        "ui:settings",
        "ui:message-actions",
        "ui:modals",
        "ui:styles",
        "actions",
        "storage",
        "tools:register",
        "commands:register",
        "connections:secrets",
        "presets:compile"
],
    apiVersion: 1,
    category: "input-output",
};

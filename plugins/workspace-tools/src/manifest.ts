import type { PluginManifest } from "../types/smileychat-plugin-api";

export const workspaceToolsManifest: PluginManifest = {
    id: "smiley-workspace-tools",
    name: "Workspace AI Tools",
    version: "1.0.0",
    description:
        "Exposes workspace management actions (Lorebooks, characters, personas, chats) directly to AI models.",
    main: "dist/index.js",
    // No `network:fetch`: every call goes through api.actions, so asking for
    // outbound access would be a permission the plugin never uses.
    permissions: ["tools:register", "actions", "state:read"],
    apiVersion: 1,
    category: "tools",
};

import type { SmileyPluginApi } from "../types/smileychat-plugin-api";

import { randomToolsManifest } from "./manifest";
import { createRandomTools } from "./tools";

export { randomToolsManifest };

export function activate(api: SmileyPluginApi) {
    const disposers = createRandomTools().map((tool) => api.tools.registerTool(tool));

    api.logger.info("Random tools registered", { tools: disposers.length });

    return () => disposers.forEach((dispose) => dispose());
}

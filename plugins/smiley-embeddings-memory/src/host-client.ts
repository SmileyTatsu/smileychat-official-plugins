/**
 * Typed JSON access to the plugin's own server module through `api.host.fetch`.
 *
 * The app refuses the call (or the module answers 404/503) until the user
 * approved "Run programs on your computer", and a plain `TypeError` from fetch
 * would tell them nothing. Every failure here therefore becomes one Error that
 * either carries the server's own message or the approval hint.
 */

import type { HostClient, ServerStatus } from "./controller";
import type { HostFetch } from "./embedder";
import type { ProcessStatus, ServerConfig } from "./server/types";

/** Shown whenever the host refuses the call or the server module is not running. */
export const APPROVAL_HINT =
    "Approve the plugin in Options > Plugins > Approvals (Run programs on your computer) and make sure it is enabled";

type JsonRecord = Record<string, unknown>;

export function createHostClient(hostFetch: HostFetch): HostClient {
    /**
     * One request, with the host's own rejections (no approval, no server
     * module, offline app) folded into the approval hint. `api.host.fetch`
     * rejects before any HTTP happens in those cases, so there is no status
     * code to report.
     */
    async function send(path: string, init?: RequestInit): Promise<Response> {
        try {
            return await hostFetch(path, init);
        } catch (error) {
            throw hostUnavailable(error);
        }
    }

    async function requestJson(path: string, init?: RequestInit): Promise<JsonRecord> {
        const response = await send(path, init);
        const body = await readJson(response);

        if (!response.ok) {
            throw errorFromBody(response.status, body);
        }

        return body;
    }

    function post(path: string, payload?: unknown): Promise<JsonRecord> {
        const init: RequestInit = { method: "POST" };

        if (payload !== undefined) {
            init.headers = { "content-type": "application/json" };
            init.body = JSON.stringify(payload);
        }

        return requestJson(path, init);
    }

    return {
        async status() {
            const body = await requestJson("/status");

            // A status that is not an object would make every caller read
            // `undefined.state`; failing here names the real problem instead.
            if (!isRecord(body.process)) {
                throw new Error("The plugin's server module did not answer with a status.");
            }

            return body as unknown as ServerStatus;
        },

        async saveConfig(config) {
            const body = await post("/config", config);
            const saved = isRecord(body.config) ? body.config : body;

            // `POST /config` echoes what it stored; without a profile the
            // response is an error body that already passed the `ok` check.
            if (typeof saved.profile !== "string") {
                throw new Error("The plugin's server module did not return the saved configuration.");
            }

            return saved as unknown as ServerConfig;
        },

        async start() {
            const body = await post("/start");
            const process = isRecord(body.process) ? body.process : body;

            if (typeof process.state !== "string") {
                throw new Error("The plugin's server module did not return the process status.");
            }

            return process as unknown as ProcessStatus;
        },

        async stop() {
            await post("/stop");
        },

        async installBinary() {
            await post("/binary/install");
        },

        async cancelBinary() {
            await post("/binary/cancel");
        },

        async removeBinary() {
            await requestJson("/binary", { method: "DELETE" });
        },

        async downloadModel() {
            await post("/models/download");
        },

        async cancelModel() {
            await post("/models/cancel");
        },

        async removeModels() {
            await requestJson("/models", { method: "DELETE" });
        },
    };
}

/** The host refused the call: approval missing, module not started, app offline. */
function hostUnavailable(error: unknown): Error {
    const detail = error instanceof Error && error.message ? ` (${error.message})` : "";

    return new Error(`Could not reach the plugin's server module${detail}. ${APPROVAL_HINT}`);
}

/**
 * `{ error: { message } }` is the module's error shape; anything else (a proxy
 * page, an empty 502) becomes a status line instead of a JSON parse error.
 */
function errorFromBody(status: number, body: JsonRecord): Error {
    const error = isRecord(body.error) ? body.error : null;
    const message = error && typeof error.message === "string" ? error.message.trim() : "";

    if (message) {
        return new Error(message);
    }

    // 404/503 is what the app itself answers when the module is missing or
    // failed to boot, so the approval hint is the useful next step.
    if (status === 404 || status === 503) {
        return new Error(`The plugin's server module is not running (HTTP ${status}). ${APPROVAL_HINT}`);
    }

    return new Error(`The plugin's server module failed with HTTP ${status}.`);
}

async function readJson(response: Response): Promise<JsonRecord> {
    try {
        const body = await response.json();

        return isRecord(body) ? body : {};
    } catch {
        // A 500 HTML page or an empty body is common; the status code still
        // tells the caller what happened.
        return {};
    }
}

function isRecord(value: unknown): value is JsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

import { describe, expect, test } from "bun:test";

import type { HostFetch } from "./embedder";
import { APPROVAL_HINT, createHostClient } from "./host-client";

type Call = { path: string; init?: RequestInit };

function fakeHost(respond: (call: Call) => Response | Promise<Response>) {
    const calls: Call[] = [];
    const hostFetch: HostFetch = async (path, init) => {
        const call = { path, init };
        calls.push(call);
        return respond(call);
    };

    return { calls, hostFetch };
}

function json(body: unknown, status = 200) {
    return new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
    });
}

function empty(status: number) {
    return new Response(null, { status });
}

describe("createHostClient", () => {
    test("status() reads GET /status", async () => {
        const host = fakeHost(() =>
            json({ process: { state: "ready", pid: 10 }, installer: {}, config: { profile: "default" } }),
        );
        const client = createHostClient(host.hostFetch);
        const status = await client.status();

        expect(host.calls[0].path).toBe("/status");
        expect(host.calls[0].init).toBeUndefined();
        expect(status.process.state).toBe("ready");
    });

    test("status() rejects when the module answers with something else", async () => {
        const host = fakeHost(() => json({ message: "hello" }));
        const client = createHostClient(host.hostFetch);

        await expect(client.status()).rejects.toThrow(/did not answer with a status/);
    });

    test("start() unwraps the process from the response body", async () => {
        const host = fakeHost(() => json({ process: { state: "ready", port: 8080 } }));
        const client = createHostClient(host.hostFetch);
        const process = await client.start();

        expect(host.calls[0].path).toBe("/start");
        expect(host.calls[0].init?.method).toBe("POST");
        expect(process.port).toBe(8080);
    });

    test("saveConfig() posts the JSON body and returns the saved config", async () => {
        const host = fakeHost(() => json({ config: { profile: "strong", ctx: 4096 } }));
        const client = createHostClient(host.hostFetch);
        const saved = await client.saveConfig({ profile: "strong", ctx: 4096 });

        expect(host.calls[0].path).toBe("/config");
        expect(host.calls[0].init?.method).toBe("POST");
        expect(host.calls[0].init?.headers).toEqual({ "content-type": "application/json" });
        expect(JSON.parse(String(host.calls[0].init?.body))).toEqual({ profile: "strong", ctx: 4096 });
        expect(saved.profile).toBe("strong");
    });

    test("saveConfig() accepts a server that echoes the config itself", async () => {
        const host = fakeHost(() => json({ profile: "default", ctx: 2048, idleMinutes: 5, device: "auto", gpuBackend: "auto", releaseTag: "b12686" }));
        const client = createHostClient(host.hostFetch);
        const saved = await client.saveConfig({ profile: "default" });

        expect(saved.ctx).toBe(2048);
    });

    test("every mutation uses the documented path and method", async () => {
        const host = fakeHost(() => empty(204));
        const client = createHostClient(host.hostFetch);

        await client.stop();
        await client.installBinary();
        await client.cancelBinary();
        await client.removeBinary();
        await client.downloadModel();
        await client.cancelModel();
        await client.removeModels();

        expect(host.calls.map((call) => [call.init?.method, call.path])).toEqual([
            ["POST", "/stop"],
            ["POST", "/binary/install"],
            ["POST", "/binary/cancel"],
            ["DELETE", "/binary"],
            ["POST", "/models/download"],
            ["POST", "/models/cancel"],
            ["DELETE", "/models"],
        ]);
    });

    test("the server's own error message is the rejection", async () => {
        const host = fakeHost(() => json({ error: { message: "llama-server exited with code 1" } }, 409));
        const client = createHostClient(host.hostFetch);

        await expect(client.start()).rejects.toThrow("llama-server exited with code 1");
    });

    test("a 503 without a body asks for the approval", async () => {
        const host = fakeHost(() => empty(503));
        const client = createHostClient(host.hostFetch);

        await expect(client.status()).rejects.toThrow(APPROVAL_HINT);
    });

    test("a rejected host call becomes one Error carrying the approval hint", async () => {
        const hostFetch: HostFetch = async () => {
            throw new Error("needs your approval to run programs");
        };
        const client = createHostClient(hostFetch);
        const error = (await client.stop().catch((reason: unknown) => reason)) as Error;

        expect(error).toBeInstanceOf(Error);
        expect(error.message).toContain("needs your approval to run programs");
        // The hint is the actionable half and must appear exactly once.
        expect(error.message.split(APPROVAL_HINT)).toHaveLength(2);
    });

    test("an error status without a JSON body still names the status", async () => {
        const host = fakeHost(() => new Response("<html>oops</html>", { status: 500 }));
        const client = createHostClient(host.hostFetch);

        await expect(client.installBinary()).rejects.toThrow(/HTTP 500/);
    });
});

import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";

import { createEmbedder, EmbeddingError, testEmbedding } from "./embedder";
import type { HostFetch } from "./embedder";
import { normalizeSettings } from "./settings";
import { normalize } from "./vectors";

type ApiMode = "ok" | "auth" | "not-found" | "rate-limit-once" | "server" | "garbage" | "bad-dims" | "no-indexes";

type Scenario = {
    mode: ApiMode;
    shuffle: boolean;
    requests: { auth: string | null; body: { model: string; input: string[]; encoding_format: string } }[];
    attempts: number;
};

const scenario: Scenario = { mode: "ok", shuffle: false, requests: [], attempts: 0 };

let server: ReturnType<typeof Bun.serve>;
let baseUrl = "";

function embeddingFor(index: number): number[] {
    return [1, index + 1];
}

beforeAll(() => {
    server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        async fetch(request) {
            const url = new URL(request.url);

            if (!url.pathname.endsWith("/embeddings")) {
                return new Response("not found", { status: 404 });
            }

            const body = (await request.json()) as Scenario["requests"][number]["body"];
            scenario.requests.push({ auth: request.headers.get("authorization"), body });
            scenario.attempts += 1;

            if (scenario.mode === "auth") {
                return Response.json({ error: { message: "invalid api key" } }, { status: 401 });
            }

            if (scenario.mode === "not-found") {
                return new Response("no such route", { status: 404 });
            }

            if (scenario.mode === "rate-limit-once" && scenario.attempts === 1) {
                return Response.json({ error: { message: "slow down" } }, { status: 429 });
            }

            if (scenario.mode === "server") {
                return new Response("boom", { status: 500 });
            }

            if (scenario.mode === "garbage") {
                return new Response("<!doctype html><html>hi</html>", { status: 200 });
            }

            if (scenario.mode === "bad-dims") {
                return Response.json({
                    data: [
                        { object: "embedding", index: 0, embedding: [1, 2] },
                        { object: "embedding", index: 1, embedding: [1, 2, 3] },
                    ],
                });
            }

            if (scenario.mode === "no-indexes") {
                return Response.json({ data: [{ object: "embedding", embedding: [1, 2] }] });
            }

            const data = body.input.map((_text, index) => ({
                object: "embedding",
                index,
                embedding: embeddingFor(index),
            }));

            // The real API promises no ordering; the client must use `index`.
            return Response.json({ object: "list", data: scenario.shuffle ? data.reverse() : data });
        },
    });

    baseUrl = `http://127.0.0.1:${server.port}/v1`;
});

afterAll(() => {
    server.stop(true);
});

beforeEach(() => {
    scenario.mode = "ok";
    scenario.shuffle = false;
    scenario.requests = [];
    scenario.attempts = 0;
});

function apiSettings(overrides: { apiKey?: string; model?: string; prefixProfile?: string; base?: string } = {}) {
    return normalizeSettings({
        source: "api",
        api: {
            baseUrl: overrides.base ?? baseUrl,
            model: overrides.model ?? "test-model",
            apiKey: overrides.apiKey ?? "",
            prefixProfile: overrides.prefixProfile ?? "none",
        },
    });
}

function apiEmbedder(
    overrides: { apiKey?: string; model?: string; prefixProfile?: string; base?: string } = {},
) {
    const delays: number[] = [];
    const embedder = createEmbedder(apiSettings(overrides), {
        // Never used by the API source; the local path must not touch it either.
        hostFetch: async () => new Response("", { status: 500 }),
        delay: async (ms) => {
            delays.push(ms);
        },
    });

    return { embedder, delays };
}

async function rejection(promise: Promise<unknown>): Promise<EmbeddingError> {
    try {
        await promise;
    } catch (error) {
        expect(error).toBeInstanceOf(EmbeddingError);
        return error as EmbeddingError;
    }

    throw new Error("expected the promise to reject");
}

describe("embedder (API source)", () => {
    test("returns vectors in input order even when the response is shuffled", async () => {
        scenario.shuffle = true;
        const { embedder } = apiEmbedder();

        const vectors = await embedder.embedDocuments(["a", "b", "c"]);

        expect(vectors.length).toBe(3);
        expect(Array.from(vectors[0])).toEqual(Array.from(normalize(new Float32Array([1, 1]))));
        expect(Array.from(vectors[1])).toEqual(Array.from(normalize(new Float32Array([1, 2]))));
        expect(Array.from(vectors[2])).toEqual(Array.from(normalize(new Float32Array([1, 3]))));
    });

    test("sends the model, the inputs and the float encoding format", async () => {
        const { embedder } = apiEmbedder({ model: "text-embedding-3-small" });

        await embedder.embedDocuments(["hello"]);

        expect(scenario.requests.length).toBe(1);
        expect(scenario.requests[0].body).toEqual({
            model: "text-embedding-3-small",
            input: ["hello"],
            encoding_format: "float",
        });
    });

    test("splits 40 inputs into batches of 16, 16 and 8", async () => {
        const { embedder } = apiEmbedder();
        const texts = Array.from({ length: 40 }, (_value, index) => `text ${index}`);

        const vectors = await embedder.embedDocuments(texts);

        expect(scenario.requests.map((request) => request.body.input.length)).toEqual([16, 16, 8]);
        expect(vectors.length).toBe(40);
    });

    test("sends the Authorization header only when a key is set", async () => {
        await apiEmbedder().embedder.embedDocuments(["x"]);
        await apiEmbedder({ apiKey: "sk-test" }).embedder.embedDocuments(["x"]);

        expect(scenario.requests[0].auth).toBeNull();
        expect(scenario.requests[1].auth).toBe("Bearer sk-test");
    });

    test("applies the prefix profile to documents and queries", async () => {
        await apiEmbedder({ prefixProfile: "gemma" }).embedder.embedDocuments(["hello"]);
        await apiEmbedder({ prefixProfile: "qwen3" }).embedder.embedQueries(["hello"]);

        expect(scenario.requests[0].body.input[0]).toBe("title: none | text: hello");
        expect(scenario.requests[1].body.input[0]).toBe(
            "Instruct: Given a chat message, retrieve relevant past events from the same conversation\nQuery:hello",
        );
    });

    test("sends no prefix when the profile is none", async () => {
        await apiEmbedder({ prefixProfile: "none" }).embedder.embedQueries(["hello"]);

        expect(scenario.requests[0].body.input[0]).toBe("hello");
    });

    test("makes no request for an empty list", async () => {
        const { embedder } = apiEmbedder();

        expect(await embedder.embedDocuments([])).toEqual([]);
        expect(scenario.attempts).toBe(0);
    });

    test("maps a rejected key to auth", async () => {
        scenario.mode = "auth";
        const { embedder } = apiEmbedder({ apiKey: "wrong" });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("auth");
        expect(error.hint).toContain("API key");
        // A wrong key is not retried: it would fail the same way three times.
        expect(scenario.attempts).toBe(1);
    });

    test("maps a missing endpoint to not-found", async () => {
        scenario.mode = "not-found";
        const { embedder } = apiEmbedder();

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("not-found");
        expect(error.hint).toContain("/embeddings");
    });

    test("retries a 429 once and then succeeds", async () => {
        scenario.mode = "rate-limit-once";
        const { embedder, delays } = apiEmbedder();

        const vectors = await embedder.embedDocuments(["x"]);

        expect(vectors.length).toBe(1);
        expect(scenario.attempts).toBe(2);
        expect(delays).toEqual([400]);
    });

    test("gives up after three attempts on 5xx", async () => {
        scenario.mode = "server";
        const { embedder, delays } = apiEmbedder();

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("server");
        expect(error.message).toContain("500");
        expect(scenario.attempts).toBe(3);
        expect(delays).toEqual([400, 800]);
    });

    test("maps a body that is not JSON to bad-response", async () => {
        scenario.mode = "garbage";
        const { embedder } = apiEmbedder();

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("bad-response");
        expect(error.hint).toContain("/v1");
    });

    test("rejects vectors of different sizes", async () => {
        scenario.mode = "bad-dims";
        const { embedder } = apiEmbedder();

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("bad-response");
        expect(error.message).toContain("different sizes");
    });

    test("rejects entries without a usable index", async () => {
        scenario.mode = "no-indexes";
        const { embedder } = apiEmbedder();

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("bad-response");
        expect(error.message).toContain("indexes");
    });

    test("maps a refused connection to unreachable and mentions CORS", async () => {
        const { embedder, delays } = apiEmbedder({ base: "http://127.0.0.1:1/v1" });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("unreachable");
        expect(error.hint).toContain("CORS");
        // Network failures are transient, so they are retried and then reported.
        expect(delays).toEqual([400, 800]);
    });

    test("maps an aborted signal to aborted without calling the endpoint", async () => {
        const { embedder } = apiEmbedder();
        const controller = new AbortController();
        controller.abort();

        const error = await rejection(embedder.embedDocuments(["x"], controller.signal));

        expect(error.code).toBe("aborted");
        expect(scenario.attempts).toBe(0);
    });
});

type LocalCall = { path: string; body: { texts: string[] } };

function localHarness(options: {
    profile?: "default" | "strong";
    respond?: (call: LocalCall) => Response | Promise<Response>;
    rejectWith?: Error;
} = {}) {
    const calls: LocalCall[] = [];
    const hostFetch: HostFetch = async (path, init) => {
        const body = JSON.parse(String(init?.body ?? "{}")) as { texts: string[] };
        calls.push({ path, body });

        if (options.rejectWith) {
            throw options.rejectWith;
        }

        if (options.respond) {
            return options.respond({ path, body });
        }

        return Response.json({ vectors: body.texts.map((_text, index) => [3, 4 + index]) });
    };

    const settings = normalizeSettings({ source: "local", profile: options.profile ?? "default" });

    return { embedder: createEmbedder(settings, { hostFetch }), calls };
}

function localErrorResponse(code: string, message: string, status = 500): Response {
    return Response.json({ error: { code, message } }, { status });
}

describe("embedder (local source)", () => {
    test("posts formatted documents to /embed and normalises the vectors", async () => {
        const { embedder, calls } = localHarness();

        const vectors = await embedder.embedDocuments(["hello"]);

        expect(calls).toEqual([{ path: "/embed", body: { texts: ["title: none | text: hello"] } }]);
        expect(Array.from(vectors[0])).toEqual(Array.from(normalize(new Float32Array([3, 4]))));
    });

    test("batches 40 texts into three /embed calls", async () => {
        const { embedder, calls } = localHarness();
        const texts = Array.from({ length: 40 }, (_value, index) => `t${index}`);

        const vectors = await embedder.embedDocuments(texts);

        expect(calls.map((call) => call.body.texts.length)).toEqual([16, 16, 8]);
        expect(vectors.length).toBe(40);
    });

    test("uses the harness profile's prefix profile", async () => {
        const { embedder, calls } = localHarness();

        expect(embedder.model.prefixProfile).toBe("gemma");
        expect(embedder.model.dim).toBe(768);

        await embedder.embedDocuments(["doc"]);
        expect(calls[0].body.texts[0]).toBe("title: none | text: doc");
    });

    test("applies the Qwen3 prefixes exactly as the model card does", async () => {
        const { embedder, calls } = localHarness({ profile: "strong" });

        await embedder.embedDocuments(["doc"]);
        await embedder.embedQueries(["q"]);

        // Qwen3 wants no document prefix and no <|endoftext|>; the server adds EOS.
        expect(calls[0].body.texts[0]).toBe("doc");
        expect(calls[1].body.texts[0]).toBe(
            "Instruct: Given a chat message, retrieve relevant past events from the same conversation\nQuery:q",
        );
    });

    test("applies the Gemma query prefix", async () => {
        const { embedder, calls } = localHarness();

        await embedder.embedQueries(["q"]);

        expect(calls[0].body.texts[0]).toBe("task: search result | query: q");
    });

    test("maps not-installed and points at Install", async () => {
        const { embedder } = localHarness({
            respond: () => localErrorResponse("not-installed", "no model file", 409),
        });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("not-installed");
        expect(error.hint).toContain("Install");
    });

    test("maps starting-failed to not-ready and keeps the server message", async () => {
        const { embedder } = localHarness({
            respond: () => localErrorResponse("starting-failed", "exit code 1"),
        });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("not-ready");
        expect(error.message).toContain("exit code 1");
    });

    test("maps not-ready and keeps the server message", async () => {
        const { embedder } = localHarness({
            respond: () => localErrorResponse("not-ready", "loading weights", 503),
        });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("not-ready");
        expect(error.message).toContain("loading weights");
    });

    test("maps too-long, busy and bad-request", async () => {
        const tooLong = await rejection(
            localHarness({ respond: () => localErrorResponse("too-long", "context") }).embedder.embedDocuments(["x"]),
        );
        expect(tooLong.code).toBe("too-long");
        expect(tooLong.hint).toContain("chunk size");

        const busy = await rejection(
            localHarness({ respond: () => localErrorResponse("busy", "already embedding") }).embedder.embedDocuments([
                "x",
            ]),
        );
        expect(busy.code).toBe("server");

        const badRequest = await rejection(
            localHarness({ respond: () => localErrorResponse("bad-request", "texts missing") }).embedder.embedDocuments(
                ["x"],
            ),
        );
        expect(badRequest.code).toBe("bad-response");
    });

    test("maps an unknown error code to server", async () => {
        const { embedder } = localHarness({
            respond: () => localErrorResponse("weird", "no idea"),
        });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("server");
        expect(error.message).toContain("no idea");
    });

    test("treats a 503 from the app itself as the module not running", async () => {
        const { embedder } = localHarness({ respond: () => new Response("plugin server failed", { status: 503 }) });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("not-ready");
        expect(error.hint).toContain("Approvals");
    });

    test("turns a rejected hostFetch into not-ready with the approvals hint", async () => {
        const { embedder } = localHarness({ rejectWith: new Error("plugin server is not approved") });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("not-ready");
        expect(error.hint).toContain("server:process");
        expect(error.hint).toContain("Approvals");
    });

    test("rejects a response without vectors", async () => {
        const { embedder } = localHarness({ respond: () => Response.json({ nope: true }) });

        const error = await rejection(embedder.embedDocuments(["x"]));

        expect(error.code).toBe("bad-response");
    });

    test("rejects inconsistent vector dimensions from the local server", async () => {
        const { embedder } = localHarness({ respond: () => Response.json({ vectors: [[1, 2], [1, 2, 3]] }) });

        const error = await rejection(embedder.embedDocuments(["two inputs"]));

        expect(error.code).toBe("bad-response");
        expect(error.message).toContain("different sizes");
    });

    test("does not call the host for an empty list", async () => {
        const { embedder, calls } = localHarness();

        expect(await embedder.embedQueries([])).toEqual([]);
        expect(calls.length).toBe(0);
    });

    test("maps an aborted signal to aborted without calling the host", async () => {
        const { embedder, calls } = localHarness();
        const controller = new AbortController();
        controller.abort();

        const error = await rejection(embedder.embedDocuments(["x"], controller.signal));

        expect(error.code).toBe("aborted");
        expect(calls.length).toBe(0);
    });
});

describe("testEmbedding", () => {
    test("reports the dimension and a latency for the probe sentence", async () => {
        const { embedder, calls } = localHarness();

        const result = await testEmbedding(embedder);

        expect(result.dim).toBe(2);
        expect(result.latencyMs).toBeGreaterThanOrEqual(0);
        expect(calls[0].body.texts[0]).toBe("title: none | text: The quick brown fox jumps over the lazy dog.");
    });

    test("fails with bad-response when the endpoint returns nothing", async () => {
        const { embedder } = localHarness({ respond: () => Response.json({ vectors: [] }) });

        const error = await rejection(testEmbedding(embedder));

        expect(error.code).toBe("bad-response");
    });
});

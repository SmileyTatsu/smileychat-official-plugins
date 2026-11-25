import { describe, expect, test } from "bun:test";

import { emptyRetrievalResult } from "./retrieval";
import { createTrace } from "./trace";
import type { LastInjection } from "./trace";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function injectionOf(chatId: string): LastInjection {
    return {
        at: "2026-01-01T00:00:00.000Z",
        chatId,
        result: emptyRetrievalResult("below-threshold", "no match: all below threshold (best 0.10)"),
        dryRun: false,
    };
}

describe("createTrace", () => {
    test("keeps the newest entries of a bounded ring", () => {
        const trace = createTrace(3);

        for (let i = 1; i <= 5; i += 1) {
            trace.log("info", "retrieval", `line ${i}`);
        }

        const logs = trace.logs();

        expect(logs.map((entry) => entry.message)).toEqual(["line 3", "line 4", "line 5"]);
        expect(logs[0].level).toBe("info");
        expect(logs[0].scope).toBe("retrieval");
        expect(new Date(logs[0].at).toISOString()).toBe(logs[0].at);
    });

    test("defaults to a 200 entry buffer", () => {
        const trace = createTrace();

        for (let i = 0; i < 250; i += 1) {
            trace.log("debug", "index", `entry ${i}`);
        }

        expect(trace.logs().length).toBe(200);
        expect(trace.logs()[199].message).toBe("entry 249");
        expect(trace.logs()[0].message).toBe("entry 50");
    });

    test("keeps the detail only when one is given", () => {
        const trace = createTrace();

        trace.log("warn", "embed", "with detail", { code: "auth" });
        trace.log("warn", "embed", "without detail");

        const [withDetail, withoutDetail] = trace.logs();

        expect(withDetail.detail).toEqual({ code: "auth" });
        expect(Object.prototype.hasOwnProperty.call(withoutDetail, "detail")).toBe(false);
    });

    test("batches notifications for a burst of logs", async () => {
        const trace = createTrace();
        let calls = 0;

        trace.onChange(() => {
            calls += 1;
        });

        trace.log("info", "index", "one");
        trace.log("info", "index", "two");
        trace.log("info", "index", "three");

        expect(calls).toBe(0);

        await tick();

        expect(calls).toBe(1);
    });

    test("swallows listener errors and still notifies the others", async () => {
        const trace = createTrace();
        const seen: string[] = [];

        trace.onChange(() => {
            throw new Error("broken panel");
        });
        trace.onChange(() => {
            seen.push("second");
        });

        trace.log("error", "server", "boom");
        await tick();

        expect(seen).toEqual(["second"]);
    });

    test("stops notifying after unsubscribe", async () => {
        const trace = createTrace();
        let calls = 0;

        const unsubscribe = trace.onChange(() => {
            calls += 1;
        });

        trace.log("info", "ui", "first");
        await tick();
        unsubscribe();
        trace.log("info", "ui", "second");
        await tick();

        expect(calls).toBe(1);
    });

    test("remembers the last injection and notifies on it", async () => {
        const trace = createTrace();
        let calls = 0;

        trace.onChange(() => {
            calls += 1;
        });

        expect(trace.lastInjection()).toBeNull();

        const injection = injectionOf("chat-1");
        trace.setLastInjection(injection);
        await tick();

        expect(trace.lastInjection()).toBe(injection);
        expect(calls).toBe(1);
    });

    test("clears the log but keeps the last injection", async () => {
        const trace = createTrace();
        const injection = injectionOf("chat-1");

        trace.log("info", "retrieval", "before");
        trace.setLastInjection(injection);
        trace.clear();

        expect(trace.logs()).toEqual([]);
        expect(trace.lastInjection()).toBe(injection);
    });

    test("returns copies so callers cannot mutate the buffer", () => {
        const trace = createTrace();

        trace.log("info", "retrieval", "one");
        trace.logs().length = 0;

        expect(trace.logs().length).toBe(1);
    });
});

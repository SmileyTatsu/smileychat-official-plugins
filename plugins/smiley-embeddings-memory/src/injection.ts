/**
 * Turns a retrieval result into a prompt injection, and bounds retrieval with a
 * deadline the prompt build can survive.
 *
 * The app awaits prompt injectors and treats a rejection as a failed
 * generation, so nothing here may throw: a timeout becomes an empty result with
 * a reason, exactly like every other "no match" outcome.
 */

import type { PromptInjection } from "../types/smileychat-plugin-api";

import { emptyRetrievalResult } from "./retrieval";
import type { RetrievalResult } from "./retrieval";
import type { PluginSettings } from "./settings";

/**
 * `priority: 10` keeps these below the summarizer's summary (`20`) when the app
 * has to trim the budget: memories are background, the summary is the spine.
 */
export function buildInjection(result: RetrievalResult, settings: PluginSettings): PromptInjection | null {
    if (result.memories.length === 0 || !result.injectedText) {
        return null;
    }

    return {
        id: "smiley-embeddings-memory",
        source: "plugin",
        role: "system",
        content: result.injectedText,
        anchor: "at-depth",
        depth: settings.depth,
        order: 0,
        priority: 10,
        // Counted (unlike an ignore-budget note) so the app's own trimming can
        // drop these before it drops the conversation itself.
        tokenBudgetBehavior: "counted",
        metadata: {
            memories: result.memories.length,
        },
    };
}

/**
 * Runs `task` with an abort signal that fires at `ms`, resolving with a
 * `timeout` result instead of rejecting.
 *
 * A slow remote embedding endpoint would otherwise delay every single send, so
 * the caller always gets an answer; the in-flight request is aborted at the
 * deadline so its socket does not stay open.
 */
export async function runWithTimeout(
    task: (signal: AbortSignal) => Promise<RetrievalResult>,
    ms: number,
): Promise<RetrievalResult> {
    const controller = new AbortController();
    const timeoutLine = `no match: timeout after ${ms} ms`;

    let timer: ReturnType<typeof setTimeout> | undefined;

    const work = (async () => {
        try {
            return await task(controller.signal);
        } catch (error) {
            // `retrieveMemories` does not throw, but a future caller might; the
            // prompt build must not see the rejection either way.
            const detail = error instanceof Error ? error.message : String(error);

            return emptyRetrievalResult(
                "embedding-error",
                `no match: embedding error: ${detail}`,
                detail,
            );
        }
    })();

    const deadline = new Promise<RetrievalResult>((resolve) => {
        timer = setTimeout(() => {
            controller.abort();
            resolve(emptyRetrievalResult("timeout", timeoutLine, `after ${ms} ms`));
        }, ms);
    });

    try {
        return await Promise.race([work, deadline]);
    } finally {
        if (timer !== undefined) {
            clearTimeout(timer);
        }
    }
}

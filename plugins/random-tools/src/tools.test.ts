import { describe, expect, test } from "bun:test";

import type { PluginTool } from "../types/smileychat-plugin-api";
import { createRandomTools } from "./tools";

const tools: Record<string, PluginTool> = Object.fromEntries(
    createRandomTools().map((tool) => [tool.name, tool]),
);

async function run(name: string, args: Record<string, unknown>) {
    const tool = tools[name];

    if (!tool) {
        throw new Error(`No tool named ${name}`);
    }

    return (await tool.run(args, {} as never)) as string;
}

function errorFrom(name: string, args: Record<string, unknown>) {
    try {
        const result = tools[name]?.run(args, {} as never);

        if (result instanceof Promise) {
            throw new Error("expected a synchronous throw");
        }

        throw new Error(`expected ${name} to reject the arguments`);
    } catch (error) {
        return error instanceof Error ? error.message : String(error);
    }
}

describe("random_pick", () => {
    test("returns one indexed option by default", async () => {
        const output = await run("random_pick", {
            options: ["The storm arrives", "A stranger knocks", "The lamp dies"],
        });

        expect(output.split("\n")).toHaveLength(2);
        expect(output).toMatch(/^random_pick: 1 of 3\n1\. index [0-2]: The |^random_pick: 1 of 3\n1\. index [0-2]: A /);
    });

    test("returns distinct indexes when asked for more", async () => {
        const output = await run("random_pick", {
            options: ["a", "b", "c", "d"],
            count: 3,
        });
        const indexes = [...output.matchAll(/index (\d)/g)].map((match) => Number(match[1]));

        expect(indexes).toHaveLength(3);
        expect(new Set(indexes).size).toBe(3);
    });

    test("marks weighted picks and reports the weight", async () => {
        const output = await run("random_pick", {
            options: ["a", "b"],
            weights: [1, 3],
        });

        expect(output).toContain("(weighted)");
        expect(output).toMatch(/\(weight [13]\): [ab]$/);
    });

    test("caps count at the number of options", async () => {
        const output = await run("random_pick", {
            options: ["a", "b"],
            count: 50,
        });

        expect(output).toContain("2 of 2");
    });

    test("rejects input the schema would not produce", () => {
        expect(errorFrom("random_pick", { options: ["only one"] })).toMatch(/at least 2/);
        expect(errorFrom("random_pick", { options: [] })).toMatch(/non-empty array/);
        expect(errorFrom("random_pick", { options: ["a", "  "] })).toMatch(/non-empty strings/);
        expect(errorFrom("random_pick", { options: "a,b" })).toMatch(/non-empty array/);
        expect(errorFrom("random_pick", { options: ["a", "b"], weights: [1] })).toMatch(
            /one number per option/,
        );
        expect(errorFrom("random_pick", { options: ["a", "b"], count: 0 })).toMatch(
            /between 1 and 100/,
        );
        expect(errorFrom("random_pick", { options: ["a", "b"], count: 1.5 })).toMatch(
            /between 1 and 100/,
        );
    });
});

describe("roll_dice", () => {
    test("formats a single roll with dice and modifier", async () => {
        const output = await run("roll_dice", { dice: "2d6+3" });

        expect(output).toMatch(/^roll_dice: 2d6\+3 -> \[[1-6], [1-6]\] \+ 3 = (5|6|7|8|9|10|11|12|13|14|15)$/);
    });

    test("omits a zero modifier", async () => {
        expect(await run("roll_dice", { dice: "d20" })).toMatch(/^roll_dice: 1d20 -> \[[0-9]+\]/);
    });

    test("lists every roll and a total when asked for several", async () => {
        const output = await run("roll_dice", { dice: "1d6", rolls: 3 });
        const lines = output.split("\n");

        expect(lines[0]).toBe("roll_dice: 1d6 (3 rolls)");
        expect(lines).toHaveLength(5);
        expect(lines[4]).toStartWith("Total: ");
    });

    test("rejects bad notation with the expected shape", () => {
        expect(errorFrom("roll_dice", { dice: "2d6*2" })).toMatch(/NdS notation/);
        expect(errorFrom("roll_dice", {})).toMatch(/dice is required/);
        expect(errorFrom("roll_dice", { dice: "1d6", rolls: 0 })).toMatch(/between 1 and 20/);
    });
});

describe("shuffle_list", () => {
    test("returns every item once", async () => {
        const output = await run("shuffle_list", { items: ["a", "b", "c"] });
        const lines = output.split("\n");

        expect(lines[0]).toBe("shuffle_list: 3 items");
        expect([...lines.slice(1)].map((line) => line.split(" ")[1])).toEqual(
            expect.arrayContaining(["a", "b", "c"]),
        );
        expect(lines).toHaveLength(4);
    });

    test("rejects a list that cannot be shuffled", () => {
        expect(errorFrom("shuffle_list", { items: ["only"] })).toMatch(/at least 2/);
        expect(errorFrom("shuffle_list", { items: [] })).toMatch(/non-empty array/);
    });
});

describe("random_number", () => {
    test("reports the value and the inclusive range", async () => {
        const output = await run("random_number", { min: 4, max: 4 });

        expect(output).toBe("random_number: 4 (inclusive 4..4)");
    });

    test("stays inside the range", async () => {
        for (let attempt = 0; attempt < 100; attempt += 1) {
            const value = Number(/random_number: (-?\d+)/.exec(await run("random_number", { min: -3, max: 3 }))?.[1]);

            expect(value).toBeGreaterThanOrEqual(-3);
            expect(value).toBeLessThanOrEqual(3);
        }
    });

    test("rejects missing or backwards bounds", () => {
        expect(errorFrom("random_number", { min: 1 })).toMatch(/min and max are required/);
        expect(errorFrom("random_number", { min: 5, max: 1 })).toMatch(/must not be greater/);
    });
});

describe("tool definitions", () => {
    test("every tool is described and named for the model", () => {
        const list = createRandomTools();

        expect(list.map((tool) => tool.name)).toEqual([
            "random_pick",
            "roll_dice",
            "shuffle_list",
            "random_number",
        ]);

        for (const tool of list) {
            expect(tool.description.length).toBeGreaterThan(40);
            expect(tool.parameters).toMatchObject({ type: "object" });
        }
    });

    test("every tool shares one chat toggle", () => {
        // Chat Details enables tools per group, so a tool without a group would
        // need its own toggle and could be switched on while its siblings stay
        // off. One group, one label, four tools.
        const groups = createRandomTools().map((tool) => tool.toolGroup);

        for (const group of groups) {
            expect(group).toEqual({ id: "random", label: "Randomness" });
        }
    });
});

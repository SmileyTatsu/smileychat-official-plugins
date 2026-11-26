import type { PluginTool } from "../types/smileychat-plugin-api";

import {
    RandomInputError,
    parseDiceSpec,
    randomInteger,
    rollDice,
    shuffleItems,
    weightedPickIndexes,
} from "./random";

/**
 * One chat-scoped group keeps the four tools together in Chat Details, where the
 * user enables them per chat. The label says "Randomness" because the host
 * composes it into "Enable <label> tools for this chat" and into the panel row.
 */
const randomGroup = { id: "random", label: "Randomness" };

const maxOptions = 100;
const maxShuffleItems = 200;
const maxRolls = 20;

const objectParameters = (
    properties: Record<string, unknown>,
    required: string[],
) => ({
    type: "object",
    properties,
    required,
    additionalProperties: false,
});

function requireStringArray(args: Record<string, unknown>, key: string, field: string) {
    const value = args[key];

    if (!Array.isArray(value) || value.length === 0) {
        throw new RandomInputError(`${field} must be a non-empty array of strings.`);
    }

    const items = value.map((entry) => (typeof entry === "string" ? entry.trim() : ""));

    if (items.some((entry) => !entry)) {
        throw new RandomInputError(`${field} entries must be non-empty strings.`);
    }

    return items;
}

function optionalInteger(
    args: Record<string, unknown>,
    key: string,
    field: string,
    fallback: number,
    minimum: number,
    maximum: number,
) {
    const value = args[key];

    if (value === undefined || value === null) {
        return fallback;
    }

    if (!Number.isInteger(value) || (value as number) < minimum || (value as number) > maximum) {
        throw new RandomInputError(
            `${field} must be an integer between ${minimum} and ${maximum} (got ${String(value)}).`,
        );
    }

    return value as number;
}

function optionalWeights(args: Record<string, unknown>, optionCount: number) {
    const value = args.weights;

    if (value === undefined || value === null) {
        return undefined;
    }

    if (!Array.isArray(value) || value.length !== optionCount) {
        throw new RandomInputError(
            `weights must be an array with one number per option (${optionCount}), got ${Array.isArray(value) ? value.length : typeof value}.`,
        );
    }

    return value as number[];
}

function formatWeight(weights: number[] | undefined, index: number) {
    return weights ? ` (weight ${weights[index]})` : "";
}

/**
 * `+3` / `-2` / `""`, with optional spaces for the readable breakdown
 * (`[4, 6] + 3 = 13`) instead of the compact notation (`2d6+3`).
 */
function signedModifier(modifier: number, spaced: boolean) {
    if (modifier === 0) {
        return "";
    }

    const sign = modifier > 0 ? "+" : "-";
    const magnitude = Math.abs(modifier);

    return spaced ? ` ${sign} ${magnitude}` : `${sign}${magnitude}`;
}

export function createRandomTools(): PluginTool[] {
    return [
        {
            name: "random_pick",
            toolGroup: randomGroup,
            displayName: "Random pick",
            description:
                "Picks one or more of the given options at random, fairly. Use this instead of " +
                "choosing yourself whenever the outcome should be random: name every candidate " +
                "option first, then call this tool once with the whole list. Optional weights " +
                "favour some options without removing the others. Calling it again re-rolls.",
            parameters: objectParameters(
                {
                    options: {
                        type: "array",
                        items: { type: "string" },
                        minItems: 2,
                        maxItems: maxOptions,
                        description: "Candidate options, each a self-contained description.",
                    },
                    weights: {
                        type: "array",
                        items: { type: "number" },
                        description:
                            "Optional relative weights, one per option. Higher is more likely; every weight must be greater than 0.",
                    },
                    count: {
                        type: "integer",
                        minimum: 1,
                        maximum: maxOptions,
                        description:
                            "How many distinct options to return. Defaults to 1; never more than the number of options.",
                    },
                },
                ["options"],
            ),
            run: (args) => {
                const options = requireStringArray(args, "options", "options");

                if (options.length < 2) {
                    throw new RandomInputError(
                        "options needs at least 2 entries. For a single candidate there is nothing to randomise.",
                    );
                }

                if (options.length > maxOptions) {
                    throw new RandomInputError(
                        `options accepts at most ${maxOptions} entries (got ${options.length}).`,
                    );
                }

                const weights = optionalWeights(args, options.length);
                // Over-asking is clamped, not rejected: a model that asks for
                // "3 of these 2" wants all of them, and an error costs it a
                // whole extra round trip to discover that.
                const count = optionalInteger(args, "count", "count", 1, 1, maxOptions);
                const picked = weightedPickIndexes(
                    weights ?? options.map(() => 1),
                    count,
                );
                const header = `random_pick: ${picked.length} of ${options.length}${
                    weights ? " (weighted)" : ""
                }`;
                const lines = picked.map((index, position) => {
                    const weight = formatWeight(weights, index);

                    return `${position + 1}. index ${index}${weight}: ${options[index]}`;
                });

                return [header, ...lines].join("\n");
            },
        },
        {
            name: "roll_dice",
            toolGroup: randomGroup,
            displayName: "Roll dice",
            description:
                "Rolls dice with standard NdS notation, for example 1d20, 2d6 or 2d6+3. Use it " +
                "when the shape of the randomness matters (checks, attacks, damage, loot, " +
                "initiative), not for a fair choice between named options.",
            parameters: objectParameters(
                {
                    dice: {
                        type: "string",
                        description:
                            'Dice in NdS notation with an optional flat modifier: "1d20", "2d6", "2d6+3", "1d8-1".',
                    },
                    rolls: {
                        type: "integer",
                        minimum: 1,
                        maximum: maxRolls,
                        description: `How many separate rolls to make. Defaults to 1, at most ${maxRolls}.`,
                    },
                },
                ["dice"],
            ),
            run: (args) => {
                const spec = parseDiceSpec(args.dice);
                const rolls = optionalInteger(args, "rolls", "rolls", 1, 1, maxRolls);
                const notation = `${spec.count}d${spec.sides}${signedModifier(spec.modifier, false)}`;
                const results = Array.from({ length: rolls }, () => rollDice(spec));
                const describe = ({ rolls: values, total }: (typeof results)[number]) =>
                    `[${values.join(", ")}]${signedModifier(spec.modifier, true)} = ${total}`;

                if (results.length === 1) {
                    return `roll_dice: ${notation} -> ${describe(results[0])}`;
                }

                const lines = results.map(
                    (result, index) => `${index + 1}. ${describe(result)}`,
                );
                const total = results.reduce((sum, result) => sum + result.total, 0);
                const header = `roll_dice: ${notation} (${results.length} rolls)`;

                return [header, ...lines, `Total: ${total}`].join("\n");
            },
        },
        {
            name: "shuffle_list",
            toolGroup: randomGroup,
            displayName: "Shuffle list",
            description:
                "Returns the given items in a random order. Use it to randomise turn order, " +
                "seating, a queue of events, or any list whose order should not be predictable.",
            parameters: objectParameters(
                {
                    items: {
                        type: "array",
                        items: { type: "string" },
                        minItems: 2,
                        maxItems: maxShuffleItems,
                        description: "Items to reorder, each a self-contained description.",
                    },
                },
                ["items"],
            ),
            run: (args) => {
                const items = requireStringArray(args, "items", "items");

                if (items.length < 2) {
                    throw new RandomInputError(
                        "items needs at least 2 entries. Nothing to shuffle otherwise.",
                    );
                }

                if (items.length > maxShuffleItems) {
                    throw new RandomInputError(
                        `items accepts at most ${maxShuffleItems} entries (got ${items.length}).`,
                    );
                }

                const shuffled = shuffleItems(items);
                const lines = shuffled.map((item, index) => `${index + 1}. ${item}`);

                return [`shuffle_list: ${shuffled.length} items`, ...lines].join("\n");
            },
        },
        {
            name: "random_number",
            toolGroup: randomGroup,
            displayName: "Random number",
            description:
                "Returns one random integer inside an inclusive range, evenly distributed. Use it " +
                "for a plain number the scene needs: how many hours pass, a percentage, a day of " +
                "the month.",
            parameters: objectParameters(
                {
                    min: {
                        type: "integer",
                        description: "Inclusive lower bound.",
                    },
                    max: {
                        type: "integer",
                        description: "Inclusive upper bound.",
                    },
                },
                ["min", "max"],
            ),
            run: (args) => {
                const min = args.min;
                const max = args.max;

                if (typeof min !== "number" || typeof max !== "number") {
                    throw new RandomInputError("min and max are required integers.");
                }

                const value = randomInteger(min, max);

                return `random_number: ${value} (inclusive ${min}..${max})`;
            },
        },
    ];
}

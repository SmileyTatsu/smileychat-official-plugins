/**
 * Randomness used by the tools.
 *
 * A model asked to "pick one" is not a fair coin: sampling from the same
 * distribution that generated the candidates favours the first or the most
 * plausible option. These helpers give it an outside source of entropy instead,
 * and keep validation in one place so every tool rejects bad input the same way.
 *
 * Everything is a pure function of an injected `RandomInt`, which is what makes
 * the tools testable without mocking the platform CSPRNG.
 */

export class RandomInputError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "RandomInputError";
    }
}

/** Uniform integer in `[0, maxExclusive)`. */
export type RandomInt = (maxExclusive: number) => number;

const uint32Range = 0x100000000;

/**
 * Uniform integer from the platform CSPRNG.
 *
 * Rejection sampling instead of `%`: a modulo biases low values whenever the
 * range does not divide 2^32, which is exactly the unfairness these tools exist
 * to avoid.
 */
export const secureRandomInt: RandomInt = (maxExclusive) => {
    if (!Number.isInteger(maxExclusive) || maxExclusive <= 0) {
        throw new RandomInputError("maxExclusive must be a positive integer.");
    }

    const limit = Math.floor(uint32Range / maxExclusive) * maxExclusive;
    const buffer = new Uint32Array(1);

    for (;;) {
        crypto.getRandomValues(buffer);

        if (buffer[0] < limit) {
            return buffer[0] % maxExclusive;
        }
    }
};

function unitRandom(randomInt: RandomInt) {
    return randomInt(uint32Range) / uint32Range;
}

export function validateWeights(weights: readonly number[]) {
    if (!weights.length) {
        throw new RandomInputError("weights must not be empty.");
    }

    weights.forEach((weight, index) => {
        if (typeof weight !== "number" || !Number.isFinite(weight) || weight <= 0) {
            throw new RandomInputError(
                `weights[${index}] must be a number greater than 0 (got ${String(weight)}).`,
            );
        }
    });
}

/**
 * Picks `count` distinct indexes, weighted and without replacement.
 *
 * Without replacement because asking twice for the same entry ("pick 3 of these
 * 3") is never what a caller means; picking with replacement is available by
 * calling the tool again.
 */
export function weightedPickIndexes(
    weights: readonly number[],
    count: number,
    randomInt: RandomInt = secureRandomInt,
): number[] {
    validateWeights(weights);

    if (!Number.isInteger(count) || count < 1) {
        throw new RandomInputError(`count must be a positive integer (got ${String(count)}).`);
    }

    const wanted = Math.min(count, weights.length);
    const pool = weights.map((weight, index) => ({ index, weight }));
    const selected: number[] = [];

    while (selected.length < wanted && pool.length > 0) {
        const total = pool.reduce((sum, entry) => sum + entry.weight, 0);
        let target = unitRandom(randomInt) * total;
        let position = pool.length - 1;

        for (let candidate = 0; candidate < pool.length; candidate += 1) {
            target -= pool[candidate].weight;

            if (target < 0) {
                position = candidate;
                break;
            }
        }

        selected.push(pool[position].index);
        pool.splice(position, 1);
    }

    return selected;
}

/** Fisher-Yates over a copy, so callers never see their input mutated. */
export function shuffleItems<T>(
    items: readonly T[],
    randomInt: RandomInt = secureRandomInt,
): T[] {
    const shuffled = [...items];

    for (let index = shuffled.length - 1; index > 0; index -= 1) {
        const swapWith = randomInt(index + 1);
        [shuffled[index], shuffled[swapWith]] = [shuffled[swapWith], shuffled[index]];
    }

    return shuffled;
}

export type DiceSpec = {
    /** Number of dice, 1..100. */
    count: number;
    /** Faces per die, 2..1000. */
    sides: number;
    /** Flat modifier added to the sum, -1000..1000. */
    modifier: number;
};

const dicePattern = /^\s*(?:(\d{1,3})\s*)?d\s*(\d{1,4})\s*(?:([+-])\s*(\d{1,4}))?\s*$/i;

const maxDice = 100;
const maxSides = 1000;
const maxModifier = 1000;

export function parseDiceSpec(input: unknown): DiceSpec {
    if (typeof input !== "string" || !input.trim()) {
        throw new RandomInputError(
            'dice is required, in NdS form: "1d20", "2d6", "2d6+3".',
        );
    }

    const match = dicePattern.exec(input);

    if (!match) {
        throw new RandomInputError(
            `dice "${input}" is not NdS notation. Use forms like "1d20", "2d6" or "2d6+3".`,
        );
    }

    const count = match[1] ? Number(match[1]) : 1;
    const sides = Number(match[2]);
    const modifier = match[4] ? Number(match[4]) * (match[3] === "-" ? -1 : 1) : 0;

    if (count < 1 || count > maxDice) {
        throw new RandomInputError(`dice count must be between 1 and ${maxDice} (got ${count}).`);
    }

    if (sides < 2 || sides > maxSides) {
        throw new RandomInputError(`die sides must be between 2 and ${maxSides} (got ${sides}).`);
    }

    if (Math.abs(modifier) > maxModifier) {
        throw new RandomInputError(
            `dice modifier must be between -${maxModifier} and ${maxModifier} (got ${modifier}).`,
        );
    }

    return { count, sides, modifier };
}

export type DiceRoll = {
    rolls: number[];
    total: number;
};

export function rollDice(spec: DiceSpec, randomInt: RandomInt = secureRandomInt): DiceRoll {
    const rolls: number[] = [];

    for (let index = 0; index < spec.count; index += 1) {
        rolls.push(randomInt(spec.sides) + 1);
    }

    const total = rolls.reduce((sum, roll) => sum + roll, 0) + spec.modifier;

    return { rolls, total };
}

const maxAbsoluteNumber = 1_000_000_000;

export function randomInteger(
    min: number,
    max: number,
    randomInt: RandomInt = secureRandomInt,
): number {
    for (const [label, value] of [
        ["min", min],
        ["max", max],
    ] as const) {
        if (!Number.isInteger(value) || Math.abs(value) > maxAbsoluteNumber) {
            throw new RandomInputError(
                `${label} must be an integer between -${maxAbsoluteNumber} and ${maxAbsoluteNumber} (got ${String(value)}).`,
            );
        }
    }

    if (min > max) {
        throw new RandomInputError(`min (${min}) must not be greater than max (${max}).`);
    }

    return min + randomInt(max - min + 1);
}

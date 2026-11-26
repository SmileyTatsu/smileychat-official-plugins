import { describe, expect, test } from "bun:test";

import {
    RandomInputError,
    parseDiceSpec,
    randomInteger,
    rollDice,
    secureRandomInt,
    shuffleItems,
    weightedPickIndexes,
    type RandomInt,
} from "./random";

const lowest: RandomInt = () => 0;
const highest: RandomInt = (maxExclusive) => maxExclusive - 1;

/** Always yields `value` for a source whose exclusive max is `of`. */
const fixed = (value: number, of: number): RandomInt => () => {
    expect(of).toBeGreaterThan(value);
    return value;
};

describe("secureRandomInt", () => {
    test("stays inside the requested range", () => {
        for (let attempt = 0; attempt < 500; attempt += 1) {
            const value = secureRandomInt(7);
            expect(Number.isInteger(value)).toBe(true);
            expect(value).toBeGreaterThanOrEqual(0);
            expect(value).toBeLessThan(7);
        }
    });

    test("a range of one is always zero", () => {
        expect(secureRandomInt(1)).toBe(0);
    });

    test("rejects a non-positive or fractional range", () => {
        expect(() => secureRandomInt(0)).toThrow(RandomInputError);
        expect(() => secureRandomInt(-3)).toThrow(RandomInputError);
        expect(() => secureRandomInt(2.5)).toThrow(RandomInputError);
    });
});

describe("weightedPickIndexes", () => {
    test("returns distinct indexes", () => {
        const picked = weightedPickIndexes([1, 1, 1, 1], 4);

        expect([...picked].sort()).toEqual([0, 1, 2, 3]);
    });

    test("never returns more entries than options", () => {
        expect(weightedPickIndexes([1, 2], 10)).toHaveLength(2);
    });

    test("a zero source always takes the first candidate", () => {
        expect(weightedPickIndexes([1, 5, 5], 1, lowest)).toEqual([0]);
        expect(weightedPickIndexes([1, 5, 5], 2, lowest)).toEqual([0, 1]);
    });

    test("the top of the range reaches the last candidate", () => {
        expect(weightedPickIndexes([1, 1, 1], 1, highest)).toEqual([2]);
    });

    test("weights move the draw towards the heavier option", () => {
        // 90 of 100 slices belong to index 1, so a draw at 0.05 lands on index 0
        // and a draw at 0.5 cannot.
        expect(weightedPickIndexes([10, 90], 1, fixed(0.05 * 0x100000000, 0x100000000))).toEqual([0]);
        expect(weightedPickIndexes([10, 90], 1, fixed(0.5 * 0x100000000, 0x100000000))).toEqual([1]);
    });

    test("rejects invalid weights and counts", () => {
        expect(() => weightedPickIndexes([], 1)).toThrow(RandomInputError);
        expect(() => weightedPickIndexes([1, 0], 1)).toThrow(/weights\[1\]/);
        expect(() => weightedPickIndexes([1, -2], 1)).toThrow(RandomInputError);
        expect(() => weightedPickIndexes([1, Number.NaN], 1)).toThrow(RandomInputError);
        expect(() => weightedPickIndexes([1, 1], 0)).toThrow(RandomInputError);
        expect(() => weightedPickIndexes([1, 1], 1.5)).toThrow(RandomInputError);
    });
});

describe("shuffleItems", () => {
    test("is deterministic for a zero source", () => {
        expect(shuffleItems(["a", "b", "c"], lowest)).toEqual(["b", "c", "a"]);
    });

    test("keeps every item and never mutates the input", () => {
        const input = ["a", "b", "c", "d"];
        const shuffled = shuffleItems(input);

        expect([...shuffled].sort()).toEqual(["a", "b", "c", "d"]);
        expect(input).toEqual(["a", "b", "c", "d"]);
    });

    test("handles empty and single-item lists", () => {
        expect(shuffleItems([], lowest)).toEqual([]);
        expect(shuffleItems(["only"], lowest)).toEqual(["only"]);
    });
});

describe("parseDiceSpec", () => {
    test("accepts the documented forms", () => {
        expect(parseDiceSpec("d20")).toEqual({ count: 1, sides: 20, modifier: 0 });
        expect(parseDiceSpec("2d6")).toEqual({ count: 2, sides: 6, modifier: 0 });
        expect(parseDiceSpec("2d6+3")).toEqual({ count: 2, sides: 6, modifier: 3 });
        expect(parseDiceSpec(" 1 d8 - 2 ")).toEqual({ count: 1, sides: 8, modifier: -2 });
        expect(parseDiceSpec("D100")).toEqual({ count: 1, sides: 100, modifier: 0 });
    });

    test("rejects everything else with an actionable message", () => {
        for (const input of ["", "  ", "abc", "2d6*2", "0d6", "d1", "101d6", "2d1001", "2d6+1001"]) {
            expect(() => parseDiceSpec(input)).toThrow(RandomInputError);
        }

        expect(() => parseDiceSpec("nope")).toThrow(/NdS notation/);
        expect(() => parseDiceSpec(undefined)).toThrow(/dice is required/);
    });
});

describe("rollDice", () => {
    test("adds the modifier to the sum", () => {
        expect(rollDice({ count: 2, sides: 6, modifier: 3 }, lowest)).toEqual({
            rolls: [1, 1],
            total: 5,
        });
        expect(rollDice({ count: 2, sides: 6, modifier: 3 }, highest)).toEqual({
            rolls: [6, 6],
            total: 15,
        });
    });

    test("rolls once per die", () => {
        const result = rollDice({ count: 4, sides: 8, modifier: 0 });

        expect(result.rolls).toHaveLength(4);
        for (const roll of result.rolls) {
            expect(roll).toBeGreaterThanOrEqual(1);
            expect(roll).toBeLessThanOrEqual(8);
        }
    });
});

describe("randomInteger", () => {
    test("is inclusive on both ends", () => {
        expect(randomInteger(3, 3, lowest)).toBe(3);
        expect(randomInteger(0, 9, lowest)).toBe(0);
        expect(randomInteger(0, 9, highest)).toBe(9);
    });

    test("stays inside the range", () => {
        for (let attempt = 0; attempt < 200; attempt += 1) {
            const value = randomInteger(-5, 5);
            expect(value).toBeGreaterThanOrEqual(-5);
            expect(value).toBeLessThanOrEqual(5);
        }
    });

    test("rejects a backwards or non-integer range", () => {
        expect(() => randomInteger(10, 1)).toThrow(/must not be greater/);
        expect(() => randomInteger(0, 1.5)).toThrow(RandomInputError);
        expect(() => randomInteger(Number.NaN, 3)).toThrow(RandomInputError);
    });
});

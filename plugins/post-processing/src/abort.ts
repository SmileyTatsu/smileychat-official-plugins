/**
 * Abort detection, local copy.
 *
 * Providers reject with either a DOMException or an Error carrying the same
 * name, and both count as "the user stopped this generation".
 */
export function isAbortError(error: unknown) {
    return (
        (error instanceof DOMException && error.name === "AbortError") ||
        (error instanceof Error && error.name === "AbortError")
    );
}

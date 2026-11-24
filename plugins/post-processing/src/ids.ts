/**
 * Id generator for pipeline passes.
 *
 * Same shape as the app's (`pass-<uuid>`): ids end up in saved settings, so they
 * must not change when a pipeline moves between the bundle and the Store.
 */
export function createId(prefix: string) {
    const unique =
        typeof crypto !== "undefined" && "randomUUID" in crypto
            ? crypto.randomUUID()
            : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

    return `${prefix}-${unique}`;
}

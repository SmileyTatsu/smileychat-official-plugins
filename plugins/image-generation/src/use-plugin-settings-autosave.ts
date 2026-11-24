import { useEffect, useRef, useState } from "preact/hooks";

import type { SmileyPluginApi } from "../types/smileychat-plugin-api";

export type RequestState = "idle" | "loading" | "success" | "error";

export type UsePluginSettingsAutosaveOptions<T = unknown> = {
    api: SmileyPluginApi;
    key?: string;
    settings: T;
    onSettingsChange?: (settings: T) => void;
    validate?: (settings: T) => void | string | Promise<void | string>;
};

/** Structural equality for settings objects, so autosave only fires on change. */
export function arePluginSettingsEqual(a: unknown, b: unknown): boolean {
    if (a === b) {
        return true;
    }

    if (typeof a !== "object" || a === null || typeof b !== "object" || b === null) {
        return false;
    }

    if (Array.isArray(a) || Array.isArray(b)) {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
            return false;
        }

        return a.every((item, index) => arePluginSettingsEqual(item, b[index]));
    }

    const aRecord = a as Record<string, unknown>;
    const bRecord = b as Record<string, unknown>;
    const aKeys = Object.keys(aRecord);

    if (aKeys.length !== Object.keys(bRecord).length) {
        return false;
    }

    for (const key of aKeys) {
        if (!Object.prototype.hasOwnProperty.call(bRecord, key)) {
            return false;
        }

        if (!arePluginSettingsEqual(aRecord[key], bRecord[key])) {
            return false;
        }
    }

    return true;
}

/**
 * Debounced autosave for a plugin's own settings.
 *
 * Store plugins only have `api.settings`, so this version persists through it
 * instead of the app's internal settings registry. The behaviour that matters is
 * unchanged: changes are validated, coalesced while a save is in flight, and a
 * programmatic update from elsewhere becomes the new baseline instead of being
 * written back.
 */
export function usePluginSettingsAutosave<T>({
    api,
    settings,
    onSettingsChange,
    validate,
}: UsePluginSettingsAutosaveOptions<T>) {
    const autosaveTimerRef = useRef<number | undefined>(undefined);
    const successTimerRef = useRef<number | undefined>(undefined);
    const lastSavedRef = useRef<T>(settings);
    const latestRef = useRef<T>(settings);
    const mountedRef = useRef(true);
    const onSettingsChangeRef = useRef(onSettingsChange);
    const queuedSaveRef = useRef<T | undefined>(undefined);
    const saveInFlightRef = useRef(false);
    const [requestState, setRequestState] = useState<RequestState>("idle");
    const [statusMessage, setStatusMessage] = useState("");

    useEffect(() => {
        onSettingsChangeRef.current = onSettingsChange;
    }, [onSettingsChange]);

    useEffect(() => {
        latestRef.current = settings;
    }, [settings]);

    // A value coming from the settings subscription is already persisted, so it
    // becomes the baseline; writing it back would loop.
    useEffect(
        () =>
            api.settings.subscribe((saved) => {
                lastSavedRef.current = saved as T;
            }),
        [api],
    );

    useEffect(() => {
        if (requestState !== "success") {
            return;
        }

        successTimerRef.current = window.setTimeout(() => {
            setRequestState("idle");
            setStatusMessage("");
        }, 2500);

        return () => {
            if (successTimerRef.current) {
                window.clearTimeout(successTimerRef.current);
            }
        };
    }, [requestState]);

    useEffect(
        () => () => {
            mountedRef.current = false;

            if (autosaveTimerRef.current) {
                window.clearTimeout(autosaveTimerRef.current);
            }

            if (successTimerRef.current) {
                window.clearTimeout(successTimerRef.current);
            }

            const latest = latestRef.current;

            if (!arePluginSettingsEqual(latest, lastSavedRef.current)) {
                void saveSettings(latest, false);
            }
        },
        [],
    );

    useEffect(() => {
        if (arePluginSettingsEqual(settings, lastSavedRef.current)) {
            return;
        }

        setRequestState("loading");
        setStatusMessage("Saving settings...");

        if (autosaveTimerRef.current) {
            window.clearTimeout(autosaveTimerRef.current);
        }

        let active = true;

        void Promise.resolve()
            .then(() => validate?.(settings))
            .then((validationError) => {
                if (!active) return;

                if (typeof validationError === "string" && validationError.trim()) {
                    setRequestState("error");
                    setStatusMessage(validationError);
                    return;
                }

                autosaveTimerRef.current = window.setTimeout(() => {
                    void saveSettings(settings);
                }, 700);
            })
            .catch((error) => {
                if (!active) return;

                setRequestState("error");
                setStatusMessage(
                    error instanceof Error && error.message
                        ? error.message
                        : "Invalid settings value.",
                );
            });

        return () => {
            active = false;

            if (autosaveTimerRef.current) {
                window.clearTimeout(autosaveTimerRef.current);
            }
        };
    }, [settings, validate]);

    async function saveSettings(
        nextSettings = latestRef.current,
        updateUi = true,
    ) {
        queuedSaveRef.current = nextSettings;

        if (saveInFlightRef.current) {
            if (updateUi && mountedRef.current) {
                setRequestState("loading");
                setStatusMessage("Saving settings...");
            }
            return;
        }

        saveInFlightRef.current = true;

        if (updateUi && mountedRef.current) {
            setRequestState("loading");
        }

        let queuedAfterFailure: T | undefined;

        try {
            while (queuedSaveRef.current !== undefined) {
                const settingsToSave = queuedSaveRef.current;
                queuedSaveRef.current = undefined;

                if (validate) {
                    const validationError = await validate(settingsToSave);

                    if (
                        typeof validationError === "string" &&
                        validationError.trim()
                    ) {
                        throw new Error(validationError);
                    }
                }

                const saved = (await api.settings.set(
                    settingsToSave as Record<string, unknown>,
                )) as T;

                lastSavedRef.current = settingsToSave;

                if (
                    updateUi &&
                    mountedRef.current &&
                    arePluginSettingsEqual(latestRef.current, settingsToSave)
                ) {
                    latestRef.current = saved;
                    lastSavedRef.current = saved;
                    onSettingsChangeRef.current?.(saved);
                }
            }

            if (
                updateUi &&
                mountedRef.current &&
                arePluginSettingsEqual(latestRef.current, lastSavedRef.current)
            ) {
                setStatusMessage("Settings saved.");
                setRequestState("success");
            }
        } catch (error) {
            queuedAfterFailure = queuedSaveRef.current;

            if (mountedRef.current) {
                setStatusMessage(
                    error instanceof Error && error.message
                        ? error.message
                        : "Failed to save settings.",
                );
                setRequestState("error");
            }
        } finally {
            saveInFlightRef.current = false;

            if (queuedAfterFailure !== undefined) {
                void saveSettings(queuedAfterFailure, updateUi);
            }
        }
    }

    return {
        requestState,
        saveSettings,
        setRequestState,
        setStatusMessage,
        statusMessage,
    };
}

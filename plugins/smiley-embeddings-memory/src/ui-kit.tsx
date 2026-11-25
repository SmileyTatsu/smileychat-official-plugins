/**
 * Small shared building blocks for the memory plugin's settings panel and modal.
 *
 * Everything here is presentation only: no plugin logic, no direct host calls. The
 * look follows the sibling plugin panels (`chat-summarizer`): same surfaces, same
 * muted palette, same field/toggle rules, so this plugin reads as part of the app
 * instead of a bolted-on window. Every class is `sem-` prefixed (see styles.css).
 */

import type { ComponentChildren, JSX } from "preact";
import { useEffect, useId, useRef, useState } from "preact/hooks";

import type { IndexState, MemoryController } from "./controller";
import type { ProcessState } from "./server/types";

export type Tone = "neutral" | "ok" | "warn" | "error";

export function cx(...parts: Array<string | false | null | undefined>): string {
    return parts.filter(Boolean).join(" ");
}

/** Re-renders the calling component whenever the controller reports a change. */
export function useController(controller: MemoryController): void {
    const [, setTick] = useState(0);

    useEffect(() => {
        const unsubscribe = controller.subscribe(() => setTick((tick) => tick + 1));

        return unsubscribe;
    }, [controller]);
}

export type Polled<T> = {
    value: T | null;
    error: Error | null;
    /** Re-reads now instead of waiting for the next tick. */
    reload: () => void;
};

/**
 * Polls `load` while `active`; keeps the last good value when a read fails so a
 * single hiccup does not blank the status block. `reload()` after an action.
 */
export function usePoll<T>(load: () => Promise<T>, intervalMs: number, active: boolean): Polled<T> {
    const [state, setState] = useState<{ value: T | null; error: Error | null }>({
        value: null,
        error: null,
    });
    const [nonce, setNonce] = useState(0);
    const loadRef = useRef(load);
    loadRef.current = load;

    useEffect(() => {
        if (!active) {
            return;
        }

        let alive = true;
        let timer: ReturnType<typeof setTimeout> | undefined;

        const tick = async () => {
            try {
                const value = await loadRef.current();

                if (alive) {
                    setState({ value, error: null });
                }
            } catch (error) {
                if (alive) {
                    setState((current) => ({ value: current.value, error: asError(error) }));
                }
            }

            if (alive) {
                timer = setTimeout(() => void tick(), Math.max(200, intervalMs));
            }
        };

        void tick();

        return () => {
            alive = false;

            if (timer !== undefined) {
                clearTimeout(timer);
            }
        };
    }, [active, intervalMs, nonce]);

    return { ...state, reload: () => setNonce((next) => next + 1) };
}

export function asError(error: unknown): Error {
    return error instanceof Error ? error : new Error(String(error));
}

export type ErrorParts = { message: string; hint: string };

/**
 * `Embedder` failures carry a `hint` (what to do next). The panel shows both and
 * never a bare status code, so unwrap whatever shape arrived.
 */
export function errorParts(error: unknown): ErrorParts {
    if (error && typeof error === "object") {
        const candidate = error as { message?: unknown; hint?: unknown };
        const message = typeof candidate.message === "string" ? candidate.message : String(error);
        const hint = typeof candidate.hint === "string" ? candidate.hint : "";

        return { message, hint };
    }

    return { message: String(error), hint: "" };
}

/* ------------------------------------------------------------------ formatting */

/** SI units, matching how model sizes are quoted on the cards (176 MB, 2.5 GB). */
export function formatBytes(bytes: number | undefined): string {
    if (!Number.isFinite(bytes ?? NaN) || (bytes ?? 0) <= 0) {
        return "0 B";
    }

    const size = bytes as number;

    // Non-breaking space: "620 MB" must not split across two lines in a progress label.
    const gap = "\u00a0";

    if (size < 1000) {
        return `${Math.round(size)}${gap}B`;
    }

    if (size < 1000 * 1000) {
        return `${Math.round(size / 1000)}${gap}kB`;
    }

    if (size < 1000 * 1000 * 1000) {
        return `${Math.round(size / (1000 * 1000))}${gap}MB`;
    }

    return `${(size / (1000 * 1000 * 1000)).toFixed(1)}${gap}GB`;
}

/** Local wall-clock time; ISO timestamps from the controller stay machine-readable. */
export function formatClock(iso: string | undefined): string {
    if (!iso) {
        return "";
    }

    const date = new Date(iso);

    if (Number.isNaN(date.getTime())) {
        return iso;
    }

    return date.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

export function describeIndexState(state: IndexState): { label: string; tone: Tone } {
    switch (state) {
        case "up-to-date":
            return { label: "Up to date", tone: "ok" };
        case "stale":
            return { label: "Stale", tone: "warn" };
        case "indexing":
            return { label: "Indexing", tone: "neutral" };
        case "model-mismatch":
            return { label: "Other model", tone: "warn" };
        case "needs-consent":
            return { label: "Needs consent", tone: "warn" };
        case "error":
            return { label: "Error", tone: "error" };
        case "empty":
            return { label: "Not indexed", tone: "neutral" };
        case "disabled":
            return { label: "Off", tone: "neutral" };
        case "no-chat":
        default:
            return { label: "No chat", tone: "neutral" };
    }
}

export function describeProcessState(state: ProcessState): { label: string; tone: Tone } {
    switch (state) {
        case "ready":
            return { label: "Ready", tone: "ok" };
        case "starting":
            return { label: "Starting", tone: "warn" };
        case "error":
            return { label: "Error", tone: "error" };
        case "stopped":
        default:
            return { label: "Stopped", tone: "neutral" };
    }
}

/* ------------------------------------------------------------------- components */

export function Section({
    title,
    description,
    actions,
    children,
    className,
}: {
    title: string;
    description?: string;
    actions?: ComponentChildren;
    children: ComponentChildren;
    className?: string;
}) {
    return (
        <section className={cx("sem-section", className)}>
            <header className="sem-section-head">
                <div className="sem-section-titles">
                    <h5>{title}</h5>
                    {description ? <small>{description}</small> : null}
                </div>
                {actions ? <div className="sem-section-actions">{actions}</div> : null}
            </header>
            <div className="sem-section-body">{children}</div>
        </section>
    );
}

export function Field({
    label,
    help,
    id,
    children,
    wide,
}: {
    label: string;
    help?: string;
    id?: string;
    children: ComponentChildren;
    wide?: boolean;
}) {
    return (
        <div className={cx("sem-field", wide && "sem-field-wide")}>
            <label className="sem-field-label" for={id}>
                {label}
            </label>
            {children}
            {help ? <small className="sem-field-help">{help}</small> : null}
        </div>
    );
}

export function Button({
    variant = "default",
    wide,
    className,
    ...rest
}: {
    variant?: "default" | "primary" | "danger" | "ghost";
    wide?: boolean;
} & JSX.ButtonHTMLAttributes<HTMLButtonElement>) {
    return (
        <button
            type={rest.type ?? "button"}
            {...rest}
            className={cx(
                "sem-button",
                `sem-button-${variant}`,
                wide && "sem-button-wide",
                typeof className === "string" ? className : undefined,
            )}
        />
    );
}

export function StatusChip({
    tone = "neutral",
    children,
    dot = true,
}: {
    tone?: Tone;
    children: ComponentChildren;
    dot?: boolean;
}) {
    return (
        <span className={cx("sem-chip", `sem-chip-${tone}`)}>
            {dot ? <span className="sem-chip-dot" aria-hidden="true" /> : null}
            {children}
        </span>
    );
}

export function ProgressBar({
    value,
    max,
    label,
    detail,
}: {
    value: number;
    max: number;
    label?: string;
    detail?: string;
}) {
    const known = Number.isFinite(max) && max > 0;
    const percent = known ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;

    return (
        <div className="sem-progress">
            <div
                className={cx("sem-progress-track", !known && "sem-progress-indeterminate")}
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={known ? Math.round(max) : undefined}
                aria-valuenow={known ? Math.round(value) : undefined}
            >
                <span style={{ width: known ? `${percent}%` : "35%" }} />
            </div>
            {label || detail ? (
                <div className="sem-progress-meta">
                    {label ? <span>{label}</span> : null}
                    {detail ? <small>{detail}</small> : null}
                </div>
            ) : null}
        </div>
    );
}

export function Note({ tone = "info", children }: { tone?: "info" | "warn"; children: ComponentChildren }) {
    return <div className={cx("sem-note", `sem-note-${tone}`)}>{children}</div>;
}

export function ErrorBox({ message, hint }: { message: string; hint?: string }) {
    return (
        <div className="sem-error" role="alert">
            <strong>{message}</strong>
            {hint ? <span>{hint}</span> : null}
        </div>
    );
}

export function Metric({
    label,
    value,
    tone,
}: {
    label: string;
    value: ComponentChildren;
    tone?: Tone;
}) {
    return (
        <div className="sem-metric">
            <span>{label}</span>
            <strong className={tone ? `sem-tone-${tone}` : undefined}>{value}</strong>
        </div>
    );
}

export function Empty({ children }: { children: ComponentChildren }) {
    return <div className="sem-empty">{children}</div>;
}

export function Segmented<T extends string>({
    options,
    value,
    onChange,
    ariaLabel,
    className,
}: {
    options: Array<{ value: T; label: string }>;
    value: T;
    onChange: (next: T) => void;
    ariaLabel: string;
    className?: string;
}) {
    return (
        <div className={cx("sem-segmented", className)} role="group" aria-label={ariaLabel}>
            {options.map((option) => (
                <button
                    key={option.value}
                    type="button"
                    aria-pressed={option.value === value}
                    className={cx("sem-segment", option.value === value && "is-active")}
                    onClick={() => onChange(option.value)}
                >
                    {option.label}
                </button>
            ))}
        </div>
    );
}

export function Switch({
    label,
    help,
    checked,
    disabled,
    onChange,
}: {
    label: string;
    help?: string;
    checked: boolean;
    disabled?: boolean;
    onChange: (checked: boolean) => void;
}) {
    return (
        <label className={cx("sem-toggle", disabled && "is-disabled")}>
            <span className="sem-toggle-text">
                <span className="sem-toggle-label">{label}</span>
                {help ? <small>{help}</small> : null}
            </span>
            <input
                checked={checked}
                disabled={disabled}
                type="checkbox"
                onChange={(event) => onChange(event.currentTarget.checked)}
            />
        </label>
    );
}

/** Two-step destructive action: no blocking browser confirm, no accidental purge. */
export function ConfirmButton({
    label,
    confirmLabel,
    onConfirm,
    disabled,
    variant = "default",
    reason,
}: {
    label: string;
    confirmLabel: string;
    onConfirm: () => void;
    disabled?: boolean;
    variant?: "default" | "primary" | "danger";
    reason?: string;
}) {
    const [armed, setArmed] = useState(false);
    const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

    useEffect(
        () => () => {
            if (timer.current !== undefined) {
                clearTimeout(timer.current);
            }
        },
        [],
    );

    const disarm = () => {
        setArmed(false);

        if (timer.current !== undefined) {
            clearTimeout(timer.current);
            timer.current = undefined;
        }
    };

    return (
        <Button
            variant={armed ? "danger" : variant}
            disabled={disabled}
            title={reason}
            onClick={() => {
                if (!armed) {
                    setArmed(true);
                    timer.current = setTimeout(disarm, 5000);
                    return;
                }

                disarm();
                onConfirm();
            }}
        >
            {armed ? confirmLabel : label}
        </Button>
    );
}

/* ------------------------------------------------------------------- text inputs */

/** Shared commit-on-blur/Enter behaviour: typing never fights normalisation. */
function useCommittedText(value: string, onCommit: (next: string) => void, trim = true) {
    const [text, setText] = useState(value);
    const [editing, setEditing] = useState(false);

    useEffect(() => {
        if (!editing) {
            setText(value);
        }
    }, [value, editing]);

    const commit = () => {
        setEditing(false);
        const next = trim ? text.trim() : text;
        setText(next);

        if (next !== value) {
            onCommit(next);
        }
    };

    return { text, setText, setEditing, commit };
}

export function TextField({
    label,
    help,
    value,
    placeholder,
    onCommit,
    disabled,
    mono,
    wide,
}: {
    label: string;
    help?: string;
    value: string;
    placeholder?: string;
    onCommit: (next: string) => void;
    disabled?: boolean;
    mono?: boolean;
    wide?: boolean;
}) {
    const id = useId();
    const { text, setText, setEditing, commit } = useCommittedText(value, onCommit);

    return (
        <Field label={label} help={help} id={id} wide={wide}>
            <input
                id={id}
                className={cx(mono && "sem-mono")}
                disabled={disabled}
                placeholder={placeholder}
                spellcheck={false}
                type="text"
                value={text}
                onBlur={commit}
                // A few environments never deliver blur; change is the fallback.
                onChange={commit}
                onFocus={() => setEditing(true)}
                onInput={(event) => setText(event.currentTarget.value)}
                onKeyDown={(event) => {
                    if (event.key === "Enter") {
                        event.preventDefault();
                        event.currentTarget.blur();
                    }
                }}
            />
        </Field>
    );
}

export function PasswordField({
    label,
    help,
    value,
    placeholder,
    onCommit,
    disabled,
}: {
    label: string;
    help?: string;
    value: string;
    placeholder?: string;
    onCommit: (next: string) => void;
    disabled?: boolean;
}) {
    const id = useId();
    const [visible, setVisible] = useState(false);
    const { text, setText, setEditing, commit } = useCommittedText(value, onCommit);

    return (
        <Field label={label} help={help} id={id}>
            <div className="sem-input-with-action">
                <input
                    id={id}
                    autoComplete="off"
                    className="sem-mono"
                    disabled={disabled}
                    placeholder={placeholder}
                    spellcheck={false}
                    type={visible ? "text" : "password"}
                    value={text}
                    onBlur={commit}
                    // A few environments never deliver blur; change is the fallback.
                    onChange={commit}
                    onFocus={() => setEditing(true)}
                    onInput={(event) => setText(event.currentTarget.value)}
                    onKeyDown={(event) => {
                        if (event.key === "Enter") {
                            event.preventDefault();
                            event.currentTarget.blur();
                        }
                    }}
                />
                <button
                    type="button"
                    className="sem-input-action"
                    aria-label={visible ? "Hide API key" : "Show API key"}
                    aria-pressed={visible}
                    title={visible ? "Hide API key" : "Show API key"}
                    disabled={disabled}
                    onClick={() => setVisible((next) => !next)}
                >
                    {visible ? "Hide" : "Show"}
                </button>
            </div>
        </Field>
    );
}

export function TextAreaField({
    label,
    help,
    value,
    rows = 4,
    placeholder,
    onCommit,
    disabled,
}: {
    label: string;
    help?: string;
    value: string;
    rows?: number;
    placeholder?: string;
    onCommit: (next: string) => void;
    disabled?: boolean;
}) {
    const id = useId();
    const { text, setText, setEditing, commit } = useCommittedText(value, onCommit, false);

    return (
        <Field label={label} help={help} id={id} wide>
            <textarea
                id={id}
                className="sem-mono"
                disabled={disabled}
                placeholder={placeholder}
                rows={rows}
                value={text}
                onBlur={commit}
                // A few environments never deliver blur; change is the fallback.
                onChange={commit}
                onFocus={() => setEditing(true)}
                onInput={(event) => setText(event.currentTarget.value)}
                onKeyDown={(event) => {
                    // Enter inserts a newline here; Ctrl+Enter commits.
                    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                        event.preventDefault();
                        event.currentTarget.blur();
                    }
                }}
            />
        </Field>
    );
}

export function NumberField({
    label,
    help,
    value,
    min,
    max,
    step = 1,
    integer = true,
    onCommit,
    disabled,
    suffix,
}: {
    label: string;
    help?: string;
    value: number;
    min?: number;
    max?: number;
    step?: number;
    integer?: boolean;
    onCommit: (next: number) => void;
    disabled?: boolean;
    suffix?: string;
}) {
    const id = useId();
    const [text, setText] = useState(String(value));
    const [editing, setEditing] = useState(false);

    useEffect(() => {
        if (!editing) {
            setText(String(value));
        }
    }, [value, editing]);

    const commit = () => {
        setEditing(false);
        const parsed = Number(text.replace(",", "."));

        if (!Number.isFinite(parsed)) {
            setText(String(value));
            return;
        }

        let next = integer ? Math.round(parsed) : parsed;

        if (min !== undefined) {
            next = Math.max(min, next);
        }

        if (max !== undefined) {
            next = Math.min(max, next);
        }

        setText(String(next));

        if (next !== value) {
            onCommit(next);
        }
    };

    return (
        <Field label={label} help={help} id={id}>
            <div className={cx("sem-input-with-unit", suffix && "has-unit")}>
                <input
                    id={id}
                    disabled={disabled}
                    inputMode={integer ? "numeric" : "decimal"}
                    max={max}
                    min={min}
                    step={step}
                    type="number"
                    value={text}
                    onBlur={commit}
                    // A few environments never deliver blur; change is the fallback.
                    onChange={commit}
                    onFocus={() => setEditing(true)}
                    onInput={(event) => setText(event.currentTarget.value)}
                    onKeyDown={(event) => {
                        if (event.key === "Enter") {
                            event.preventDefault();
                            event.currentTarget.blur();
                        }
                    }}
                />
                {suffix ? <span className="sem-unit">{suffix}</span> : null}
            </div>
        </Field>
    );
}

export function SelectField<T extends string>({
    label,
    help,
    value,
    options,
    onChange,
    disabled,
}: {
    label: string;
    help?: string;
    value: T;
    options: Array<{ value: T; label: string }>;
    onChange: (next: T) => void;
    disabled?: boolean;
}) {
    const id = useId();

    return (
        <Field label={label} help={help} id={id}>
            <select
                id={id}
                disabled={disabled}
                value={value}
                onChange={(event) => onChange(event.currentTarget.value as T)}
            >
                {options.map((option) => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
        </Field>
    );
}

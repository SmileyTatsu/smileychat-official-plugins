/**
 * The plugin's viewer modal: what the last turn injected, a dry run against a
 * draft, the index/server state, and the trace log.
 *
 * This is the "why did nothing get injected" surface, so every tab answers with
 * the numbers the retrieval code actually produced instead of a summary of them.
 */

import { useEffect, useRef, useState } from "preact/hooks";

import type { IndexerProgress, MemoryController } from "./controller";
import type { Memory, NoMatchReason, RetrievalResult } from "./retrieval";
import type { LastInjection, TraceEntry, TraceLevel } from "./trace";
import {
    Button,
    ConfirmButton,
    Empty,
    ErrorBox,
    Metric,
    Note,
    ProgressBar,
    Section,
    Segmented,
    StatusChip,
    cx,
    describeIndexState,
    describeProcessState,
    errorParts,
    formatBytes,
    formatClock,
    useController,
    usePoll,
    type ErrorParts,
    type Tone,
} from "./ui-kit";

type Tab = "last" | "dry" | "status" | "logs";

const TABS: Array<{ value: Tab; label: string }> = [
    { value: "last", label: "Last injection" },
    { value: "dry", label: "Dry run" },
    { value: "status", label: "Status" },
    { value: "logs", label: "Logs" },
];

const REASON_TEXT: Record<NoMatchReason, string> = {
    disabled: "Memory is off, so nothing is retrieved.",
    "empty-index": "This chat has no index yet. Run Index this chat in settings.",
    "model-mismatch": "The index was built with another model. Re-index to use the current one.",
    "no-queries": "There was no text to search with.",
    "all-in-recent-window": "Every match is inside the recent messages that are already in the prompt.",
    "below-threshold": "The best match is below the similarity threshold in settings.",
    "token-cap": "The memories do not fit within the memory token cap.",
    timeout: "The embedder did not answer before the timeout.",
    "embedding-error": "The embedder failed, so nothing could be compared.",
};

const LEVEL_TONE: Record<TraceLevel, Tone> = {
    debug: "neutral",
    info: "neutral",
    warn: "warn",
    error: "error",
};

const LEVELS: TraceLevel[] = ["debug", "info", "warn", "error"];

export function MemoryModal({
    controller,
    onClose,
}: {
    controller: MemoryController;
    onClose: () => void;
}) {
    useController(controller);

    const [tab, setTab] = useState<Tab>("last");

    return (
        <section className="sem-modal">
            <Segmented
                className="sem-tabs"
                ariaLabel="Memory panels"
                value={tab}
                onChange={setTab}
                options={TABS}
            />

            <div className="sem-tabpanel">
                {tab === "last" ? <LastInjectionTab controller={controller} /> : null}
                {tab === "dry" ? <DryRunTab controller={controller} /> : null}
                {tab === "status" ? <StatusTab controller={controller} /> : null}
                {tab === "logs" ? <LogsTab controller={controller} /> : null}
            </div>

            <footer className="sem-modal-foot">
                <span className="sem-modal-hint">
                    {controller.activeChatId() ? "Chat memory" : "No chat is open"}
                </span>
                <Button onClick={onClose}>Close</Button>
            </footer>
        </section>
    );
}

/* ----------------------------------------------------------------- last injection */

function LastInjectionTab({ controller }: { controller: MemoryController }) {
    const injection = controller.trace.lastInjection();

    if (!injection) {
        return (
            <Empty>
                No injection yet. With memory on, the memories that match the newest messages are added
                to each prompt.
            </Empty>
        );
    }

    return <InjectionView injection={injection} />;
}

function InjectionView({ injection }: { injection: LastInjection }) {
    return (
        <div className="sem-injection">
            <div className="sem-injection-meta">
                <span title={injection.at}>{formatClock(injection.at)}</span>
                {injection.dryRun ? <StatusChip tone="warn">Dry run, nothing sent</StatusChip> : null}
                <StatusChip tone={injection.result.memories.length ? "ok" : "neutral"} dot={false}>
                    {injection.result.memories.length} memories
                </StatusChip>
            </div>

            <RetrievalView result={injection.result} />

            <p className="sem-footnote">These are the exact numbers from the last retrieval, after the cap.</p>
        </div>
    );
}

/* --------------------------------------------------------------------- dry run */

function DryRunTab({ controller }: { controller: MemoryController }) {
    const [draft, setDraft] = useState(() => controller.currentDraft() ?? "");
    const [result, setResult] = useState<RetrievalResult | null>(null);
    const [failure, setFailure] = useState<ErrorParts | null>(null);
    const [running, setRunning] = useState(false);
    const [ranAt, setRanAt] = useState<string | null>(null);

    const fromComposer = controller.currentDraft() !== null;

    async function run() {
        if (running) {
            return;
        }

        setRunning(true);
        setFailure(null);

        try {
            setResult(await controller.dryRun(draft));
            setRanAt(new Date().toISOString());
        } catch (error) {
            setFailure(errorParts(error));
        } finally {
            setRunning(false);
        }
    }

    return (
        <div className="sem-dryrun">
            <label className="sem-draft">
                <span className="sem-field-label">Message to test</span>
                <textarea
                    rows={4}
                    value={draft}
                    placeholder="Type what you are about to send..."
                    disabled={running}
                    onInput={(event) => setDraft(event.currentTarget.value)}
                />
                <small className="sem-field-help">
                    {fromComposer
                        ? "Prefilled from the composer draft; editing here does not change it."
                        : "No composer text was available, so this box is empty."}
                </small>
            </label>

            <div className="sem-button-row">
                <Button variant="primary" disabled={running || !draft.trim()} onClick={() => void run()}>
                    {running ? "Searching…" : "Run"}
                </Button>
                <small className="sem-reason">Dry run only: nothing is sent and the last injection is unchanged.</small>
            </div>

            {failure ? (
                <ErrorBox
                    message={failure.message}
                    hint={failure.hint}
                    onDismiss={() => setFailure(null)}
                />
            ) : null}

            {result ? (
                <div className="sem-result">
                    <div className="sem-injection-meta">
                        <span>{ranAt ? formatClock(ranAt) : ""}</span>
                        <StatusChip tone="warn">Dry run, not sent</StatusChip>
                    </div>
                    <RetrievalView result={result} />
                </div>
            ) : null}
        </div>
    );
}

/* ---------------------------------------------------------------------- status */

function StatusTab({ controller }: { controller: MemoryController }) {
    const indexPoll = usePoll(() => controller.indexStatus(), 2500, true);
    const serverPoll = usePoll(() => controller.host.status(), 4000, true);
    const [busy, setBusy] = useState(false);
    const [failure, setFailure] = useState<ErrorParts | null>(null);
    // Both polled errors can be on screen at once, and each is still true after the
    // other is dismissed, so dismissal is a list. A message that changes shows again.
    const [dismissed, setDismissed] = useState<string[]>([]);
    const hideError = (message: string) =>
        setDismissed((previous) => (previous.includes(message) ? previous : [...previous, message]));

    const index = indexPoll.value;
    const progress: IndexerProgress = controller.progress();
    const settings = controller.settings();
    const indexChip = index ? describeIndexState(index.state) : null;
    const process = serverPoll.value?.process;
    const processChip = process ? describeProcessState(process.state) : null;
    const installer = serverPoll.value?.installer;

    async function run(action: () => Promise<void>) {
        if (busy) {
            return;
        }

        setBusy(true);
        setFailure(null);
        setDismissed([]);

        try {
            await action();
        } catch (error) {
            setFailure(errorParts(error));
        } finally {
            setBusy(false);
            indexPoll.reload();
            serverPoll.reload();
        }
    }

    const canIndex =
        settings.enabled &&
        !progress.running &&
        !busy &&
        (settings.source !== "api" || settings.apiNoticeAccepted) &&
        (index?.messagesTotal ?? 0) > 0;

    return (
        <div className="sem-status-tab">
            <Section
                title="Index"
                description={index?.detail ?? undefined}
                actions={indexChip ? <StatusChip tone={indexChip.tone}>{indexChip.label}</StatusChip> : null}
            >
                <div className="sem-metrics sem-metrics-4">
                    <Metric label="Chunks" value={index?.chunks ?? 0} />
                    <Metric label="Indexed" value={index?.messagesIndexed ?? 0} />
                    <Metric label="In chat" value={index?.messagesTotal ?? 0} />
                    <Metric
                        label="Pending"
                        value={index?.messagesPending ?? 0}
                        tone={(index?.messagesPending ?? 0) > 0 ? "warn" : undefined}
                    />
                </div>

                {index?.indexModelId && index.indexModelId !== index.activeModelId ? (
                    <Note tone="warn">Index built with {index.indexModelId}, active model is {index.activeModelId}.</Note>
                ) : null}

                {progress.running ? (
                    <ProgressBar
                        value={progress.done}
                        max={progress.total}
                        label={`${progress.done} of ${progress.total} messages · ${progress.chunks} chunks`}
                        detail={progress.cancelling ? "Cancelling…" : undefined}
                    />
                ) : null}

                <div className="sem-button-row">
                    <Button
                        variant="primary"
                        disabled={!canIndex}
                        title={canIndex ? undefined : "Needs an open chat with messages, memory on, and no running job."}
                        onClick={() => void run(() => controller.indexAll())}
                    >
                        Index all
                    </Button>
                    <ConfirmButton
                        label="Re-index"
                        confirmLabel="Rebuild the index?"
                        disabled={busy || progress.running || !index || index.chunks === 0}
                        onConfirm={() => void run(() => controller.reindex())}
                    />
                    <ConfirmButton
                        label="Purge"
                        confirmLabel="Delete the index?"
                        disabled={busy || progress.running || !index || index.chunks === 0}
                        onConfirm={() => void run(() => controller.purge())}
                    />
                    {progress.running ? (
                        <Button onClick={() => controller.cancelIndexing()}>Cancel</Button>
                    ) : null}
                </div>

                {index?.lastError && !dismissed.includes(index.lastError) ? (
                    <ErrorBox
                        message={index.lastError}
                        onDismiss={() => hideError(index.lastError!)}
                    />
                ) : null}
                {failure ? (
                    <ErrorBox
                        message={failure.message}
                        hint={failure.hint}
                        onDismiss={() => setFailure(null)}
                    />
                ) : null}
            </Section>

            <Section title="Local server" description="Only the local source runs a program on this computer.">
                {serverPoll.error ? (
                    <Note tone="warn">Server tools are not reachable: {serverPoll.error.message}</Note>
                ) : null}

                <div className="sem-metrics">
                    <Metric
                        label="Process"
                        value={processChip?.label ?? "—"}
                        tone={processChip?.tone}
                    />
                    <Metric
                        label="Device"
                        value={
                            process?.device ??
                            process?.backend ??
                            "—"
                        }
                    />
                    <Metric
                        label="GPU layers"
                        value={
                            process?.layersTotal !== undefined
                                ? `${process.layersOffloaded ?? 0}/${process.layersTotal}`
                                : "—"
                        }
                    />
                    <Metric
                        label="Binary"
                        value={installer?.binary.installed ? formatBytes(installer.binary.bytesOnDisk) : "not installed"}
                    />
                    <Metric
                        label="Model"
                        value={installer?.model.installed ? formatBytes(installer.model.bytesOnDisk) : "not installed"}
                    />
                    <Metric
                        label="Port"
                        value={process?.port ? `${process.port}` : "—"}
                        tone={processChip?.tone}
                    />
                </div>

                {process?.lastError && !dismissed.includes(process.lastError) ? (
                    <ErrorBox
                        message={process.lastError}
                        onDismiss={() => hideError(process.lastError!)}
                    />
                ) : null}
            </Section>
        </div>
    );
}

/* ------------------------------------------------------------------------ logs */

function LogsTab({ controller }: { controller: MemoryController }) {
    const [, setTraceTick] = useState(0);
    const [levels, setLevels] = useState<Record<TraceLevel, boolean>>({
        debug: true,
        info: true,
        warn: true,
        error: true,
    });

    useEffect(() => controller.trace.onChange(() => setTraceTick((tick) => tick + 1)), [controller]);

    // Newest first: the reason for the last turn is what the user came for.
    const entries: TraceEntry[] = controller.trace.logs().slice().reverse();
    const shown = entries.filter((entry) => levels[entry.level]);

    return (
        <div className="sem-logs">
            <div className="sem-log-toolbar">
                <div className="sem-log-filters">
                    {LEVELS.map((level) => {
                        const active = levels[level];

                        return (
                            <button
                                key={level}
                                type="button"
                                aria-pressed={active}
                                className={cx("sem-level", `sem-level-${level}`, active && "is-active")}
                                title={active ? `Hide ${level} entries` : `Show ${level} entries`}
                                onClick={() => setLevels((current) => ({ ...current, [level]: !current[level] }))}
                            >
                                {level}
                            </button>
                        );
                    })}
                </div>
                <Button variant="ghost" disabled={entries.length === 0} onClick={() => controller.trace.clear()}>
                    Clear
                </Button>
            </div>

            {shown.length === 0 ? (
                <Empty>
                    {entries.length === 0
                        ? "No log entries yet. Indexing and retrieval write here as they run."
                        : "No entries match the filter."}
                </Empty>
            ) : (
                <ol className="sem-log-list">
                    {shown.map((entry, position) => (
                        <li key={`${entry.at}-${position}`} className="sem-log-row">
                            <div className="sem-log-head">
                                <span className="sem-log-time sem-mono">{formatClock(entry.at)}</span>
                                <StatusChip tone={LEVEL_TONE[entry.level]} dot={false}>
                                    {entry.level}
                                </StatusChip>
                                <span className="sem-log-scope sem-mono">{entry.scope}</span>
                            </div>
                            <p className="sem-log-message">{entry.message}</p>
                        </li>
                    ))}
                </ol>
            )}
        </div>
    );
}

/* --------------------------------------------------------------- retrieval view */

function RetrievalView({ result }: { result: RetrievalResult }) {
    return (
        <div className="sem-retrieval">
            <div className="sem-injection-line sem-mono">{result.line}</div>

            {result.reason ? (
                <Note tone="warn">
                    {REASON_TEXT[result.reason]}
                    {result.detail ? <span className="sem-detail"> ({result.detail})</span> : null}
                </Note>
            ) : null}

            {result.memories.length > 0 ? (
                <>
                    {/* The summary lists scores best-first, these cards follow the chat. */}
                    <span className="sem-label-sm">Shown in chat order (oldest first)</span>
                    <ol className="sem-memories">
                        {result.memories.map((memory) => (
                            <MemoryRow key={memory.chunkId} memory={memory} />
                        ))}
                    </ol>
                </>
            ) : null}

            <div className="sem-metrics">
                <Metric label="Candidates" value={result.candidates} />
                <Metric label="Skipped recent" value={result.skippedRecent} />
                <Metric label="Below threshold" value={result.belowThreshold} />
                <Metric label="Best score" value={result.best.toFixed(2)} />
                <Metric label="Tokens" value={result.tokens} />
                <Metric
                    label="Dropped by cap"
                    value={result.droppedByTokenCap}
                    tone={result.droppedByTokenCap > 0 ? "warn" : undefined}
                />
            </div>

            {result.droppedByTokenCap > 0 ? (
                <p className="sem-footnote">
                    {result.droppedByTokenCap} memor
                    {result.droppedByTokenCap === 1 ? "y was" : "ies were"} dropped to stay within the token cap.
                </p>
            ) : null}

            {result.queries.length > 0 ? (
                <div className="sem-queries">
                    <span className="sem-label-sm">Queries used</span>
                    {result.queries.map((query, position) => (
                        <p key={position} className="sem-query sem-mono">
                            {query}
                        </p>
                    ))}
                </div>
            ) : null}
        </div>
    );
}

function MemoryRow({ memory }: { memory: Memory }) {
    const [open, setOpen] = useState(false);
    const [clipped, setClipped] = useState(false);
    const textRef = useRef<HTMLParagraphElement>(null);

    // A short memory fits in two lines already; a "More" button there would be a lie.
    // Re-measured on resize too: the modal is the same DOM at 360 and at 1280.
    useEffect(() => {
        const element = textRef.current;

        if (!element || open) {
            return;
        }

        // Measured now and once more after layout settles: a resize callback can
        // arrive while scrollHeight still describes the previous width.
        let timer: ReturnType<typeof setTimeout> | undefined;
        const read = () => setClipped(element.scrollHeight > element.clientHeight + 1);
        const measure = () => {
            read();

            if (timer !== undefined) {
                clearTimeout(timer);
            }

            timer = setTimeout(read, 80);
        };

        measure();
        window.addEventListener("resize", measure);

        const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
        observer?.observe(element);

        return () => {
            if (timer !== undefined) {
                clearTimeout(timer);
            }

            window.removeEventListener("resize", measure);
            observer?.disconnect();
        };
    }, [memory.text, open]);

    const ago =
        memory.messagesAgo < 0
            ? "message gone from chat"
            : memory.messagesAgo === 0
              ? "newest message"
              : `${memory.messagesAgo} message${memory.messagesAgo === 1 ? "" : "s"} ago`;

    return (
        <li className="sem-memory">
            <div className="sem-memory-head">
                <span className="sem-score sem-mono" title="Cosine similarity">
                    {memory.score.toFixed(2)}
                </span>
                <span className="sem-memory-ago">{ago}</span>
                {clipped || open ? (
                    <button
                        type="button"
                        className="sem-memory-toggle"
                        aria-expanded={open}
                        onClick={() => setOpen((current) => !current)}
                    >
                        {open ? "Less" : "More"}
                    </button>
                ) : null}
            </div>
            <p ref={textRef} className={cx("sem-memory-text", open && "is-open")}>
                {memory.text}
            </p>
        </li>
    );
}

/**
 * Settings panel: what the plugin is doing right now, a three-step setup wizard
 * that a beginner can finish without reading anything else, and every knob behind
 * an Advanced disclosure.
 *
 * The panel owns no state the plugin needs to keep: settings live in the
 * controller, index/server state is polled from it, and the only local state is
 * transient UI (busy flag, last result, which failure to show).
 */

import type { ComponentChildren } from "preact";
import { useRef, useState } from "preact/hooks";

import type { IndexStatus, IndexerProgress, MemoryController } from "./controller";
import { DEFAULT_QWEN3_TASK, LOCAL_MODELS } from "./models";
import type { PluginSettings } from "./settings";
import { effectiveThreshold, serverConfigOf } from "./settings";
import type { Backend } from "./server/releases";
import type {
    DeviceChoice,
    GpuBackendChoice,
    ModelProfileChoice,
    ProcessStatus,
    TaskStatus,
} from "./server/types";
import type { PrefixProfile } from "./models";
import {
    Button,
    ConfirmButton,
    ErrorBox,
    Note,
    NumberField,
    PasswordField,
    ProgressBar,
    Section,
    Segmented,
    SelectField,
    StatusChip,
    Switch,
    TextAreaField,
    TextField,
    cx,
    describeIndexState,
    describeProcessState,
    errorParts,
    formatBytes,
    useController,
    usePoll,
    type ErrorParts,
} from "./ui-kit";

type Work = "install" | "test" | "index";

type Failure = { at: Work | "advanced"; parts: ErrorParts };

/** One press of the setup button walks these stages in order. */
const LOCAL_STAGES = ["config", "binary", "model", "start", "test"] as const;
type Stage = (typeof LOCAL_STAGES)[number];

const STAGE_TEXT: Record<Stage, string> = {
    config: "Saving the connection",
    binary: "Installing llama.cpp for this device",
    model: "Downloading the model",
    start: "Starting the local server",
    test: "Embedding a test sentence",
};

const POLL_MS = 700;
const TASK_TIMEOUT_MS = 30 * 60_000;
const START_TIMEOUT_MS = 150_000;

/** The user pressed Cancel; not a failure, so it must not paint a red box. */
class ChainCancelled extends Error {}

/** A stage failed for a reason this file can explain, with the fix in `hint`. */
class ChainError extends Error {
    readonly hint: string;

    constructor(message: string, hint: string) {
        super(message);
        this.hint = hint;
    }
}

const pause = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function chainParts(error: unknown): ErrorParts {
    return error instanceof ChainError ? { message: error.message, hint: error.hint } : errorParts(error);
}

const PREFIX_OPTIONS: Array<{ value: PrefixProfile; label: string }> = [
    { value: "none", label: "None (plain text)" },
    { value: "qwen3", label: "Qwen3 (Instruct / Query)" },
    { value: "gemma", label: "Gemma (task / query)" },
];

export function MemorySettingsPanel({ controller }: { controller: MemoryController }) {
    useController(controller);

    const settings = controller.settings();
    const local = settings.source === "local";

    const [busy, setBusy] = useState<Work | null>(null);
    const [failure, setFailure] = useState<Failure | null>(null);
    const [outcome, setOutcome] = useState<{ dim: number; latencyMs: number } | null>(null);
    const [stage, setStage] = useState<Stage | null>(null);
    const [notice, setNotice] = useState<string | null>(null);
    // A red box about a state that is still true (the server's last error) has to be
    // dismissible, and a new attempt must not show the error of the previous one.
    const [hiddenError, setHiddenError] = useState<string | null>(null);
    const cancelRef = useRef(false);
    // Only the open state is local: whether setup finished is a saved setting, so
    // a reload shows the same state on every device the profile is used on.
    const [setupOpen, setSetupOpen] = useState(!settings.setupComplete);

    const server = usePoll(
        () => controller.host.status(),
        busy === "install" ? 1000 : 4000,
        local || busy === "install",
    );
    const indexPoll = usePoll(() => controller.indexStatus(), 3000, true);

    const index = indexPoll.value;
    const progress = controller.progress();
    const process = server.value?.process;
    const installer = server.value?.installer;

    const commit = (patch: Partial<PluginSettings>) => controller.updateSettings(patch);
    const failureAt = (at: Failure["at"]) => (failure?.at === at ? failure.parts : null);

    function reloadAll() {
        server.reload();
        indexPoll.reload();
    }

    function saveTemplate(next: string) {
        if (!next.includes("{{memories}}")) {
            setFailure({
                at: "advanced",
                parts: {
                    message: "The injection template must contain {{memories}}.",
                    hint: "Put the marker where the retrieved fragments should appear; nothing was saved.",
                },
            });
            return;
        }

        setFailure(null);
        commit({ template: next });
    }

    /**
     * Step 3 of the wizard. One press runs the whole chain so a beginner never has
     * to guess how many times to click: save the connection, install the build that
     * matches the current device, download the model, start the server, self-check.
     * Stages that are already satisfied are skipped, so pressing the button again
     * after a failure continues where the chain stopped.
     */
    async function installAndTest() {
        if (busy) {
            return;
        }

        cancelRef.current = false;
        setBusy("install");
        setFailure(null);
        setOutcome(null);
        setNotice(null);
        setStage("config");
        dismissLastError();

        try {
            const patch: Partial<PluginSettings> =
                settings.source === "local"
                    ? {
                          source: "local",
                          profile: wizardProfile(settings),
                          device: settings.device,
                      }
                    : { source: "api" };

            commit(patch);
            await controller.host.saveConfig(serverConfigOf({ ...controller.settings(), ...patch }));

            if (controller.settings().source === "local") {
                await ensureBinary();
                await ensureModel();
                await startServer();
            }

            setStage("test");
            const result = await controller.test();

            setOutcome(result);
            commit({ setupComplete: true });
            dismissLastError();
        } catch (error) {
            if (error instanceof ChainCancelled) {
                setNotice("Setup was cancelled. Press the button again to continue from here.");
            } else {
                setFailure({ at: "install", parts: chainParts(error) });
            }
        } finally {
            setStage(null);
            setBusy(null);
            reloadAll();
        }
    }

    /** `installed` is device-aware on the server, so this is the only honest check. */
    async function ensureBinary() {
        const status = await controller.host.status();

        if (status.installer.binary.installed) {
            return;
        }

        setStage("binary");
        await controller.host.installBinary();
        await waitForTask("binary");
    }

    async function ensureModel() {
        const status = await controller.host.status();

        if (status.installer.model.installed) {
            return;
        }

        setStage("model");
        await controller.host.downloadModel();
        await waitForTask("model");
    }

    /**
     * The install call returns as soon as the download starts, so the chain watches
     * the task itself; starting a server before the files exist would only fail.
     */
    async function waitForTask(kind: "binary" | "model") {
        const deadline = Date.now() + TASK_TIMEOUT_MS;

        while (Date.now() < deadline) {
            if (cancelRef.current) {
                throw new ChainCancelled();
            }

            await pause(POLL_MS);

            const status = await controller.host.status();
            const entry = kind === "binary" ? status.installer.binary : status.installer.model;

            if (entry.installed) {
                return;
            }

            if (entry.task.state === "cancelled") {
                throw new ChainCancelled();
            }

            if (entry.task.state === "error") {
                throw new ChainError(
                    kind === "binary" ? "The llama.cpp download failed." : "The model download failed.",
                    entry.task.error ?? "Check the connection and try again.",
                );
            }
        }

        throw new ChainError(
            kind === "binary" ? "The llama.cpp download did not finish." : "The model download did not finish.",
            "Finished parts are kept: press the button again to resume the download.",
        );
    }

    async function startServer() {
        const before = await controller.host.status();

        if (before.process.state === "ready") {
            return;
        }

        setStage("start");
        await controller.host.start();

        const deadline = Date.now() + START_TIMEOUT_MS;

        while (Date.now() < deadline) {
            if (cancelRef.current) {
                throw new ChainCancelled();
            }

            await pause(POLL_MS);

            const status = await controller.host.status();

            if (status.process.state === "ready") {
                return;
            }

            if (status.process.state === "error") {
                throw new ChainError(
                    "llama-server stopped before it was ready.",
                    status.process.lastError ?? "Open Diagnostics and read the first error line in the logs.",
                );
            }
        }

        throw new ChainError(
            "The local server did not become ready.",
            "A GPU build on an unsupported driver is the usual cause; set Device to CPU and try again.",
        );
    }

    async function runTest(at: "install" | "advanced") {
        if (busy) {
            return;
        }

        setBusy("test");
        setFailure(null);
        setOutcome(null);
        setNotice(null);
        dismissLastError();

        try {
            const result = await controller.test();

            setOutcome(result);
            commit({ setupComplete: true });
            dismissLastError();
            reloadAll();
        } catch (error) {
            setFailure({ at, parts: errorParts(error) });
        } finally {
            setBusy(null);
        }
    }

    async function cancelInstall() {
        // The chain checks this between polls; the host calls below only make the
        // download stop in the server as well.
        cancelRef.current = true;

        try {
            const status = await controller.host.status();

            if (status.installer.binary.task.state === "running") {
                await controller.host.cancelBinary();
            }

            if (status.installer.model.task.state === "running") {
                await controller.host.cancelModel();
            }
        } catch {
            // Cancelling is best effort; the chain itself reports the real failure.
        }

        reloadAll();
    }

    async function indexChat() {
        if (busy) {
            return;
        }

        setBusy("index");
        setFailure(null);
        setNotice(null);
        dismissLastError();

        try {
            await controller.indexAll();
        } catch (error) {
            setFailure({ at: "index", parts: errorParts(error) });
        } finally {
            setBusy(null);
            reloadAll();
        }
    }

    async function runAdvanced(action: () => Promise<void>) {
        if (busy) {
            return;
        }

        setBusy("index");
        setFailure(null);
        setNotice(null);
        dismissLastError();

        try {
            await action();
        } catch (error) {
            setFailure({ at: "advanced", parts: errorParts(error) });
        } finally {
            setBusy(null);
            reloadAll();
        }
    }

    const indexReason = disabledReason(settings, index, progress, Boolean(busy), controller.activeChatId());
    const model = describeModel(settings);
    const indexChip = index ? describeIndexState(index.state) : null;
    const processChip = process ? describeProcessState(process.state) : null;
    const lastError = index?.lastError ?? process?.lastError;
    const visibleLastError = lastError && lastError !== hiddenError ? lastError : null;
    const partInstalled = Boolean(
        installer && (!installer.binary.installed || !installer.model.installed),
    );
    const backendGap = deviceGapText(installer?.binary.installedBackends, settings.device);
    const autoThreshold = autoThresholdFor(settings, controller);
    const stages: Stage[] = local ? [...LOCAL_STAGES] : ["config", "test"];
    const stageIndex = stage ? stages.indexOf(stage) : -1;

    /** Every attempt hides the error it is about to retry, so only new ones appear. */
    function dismissLastError() {
        setHiddenError(lastError ?? null);
    }

    return (
        <div className="sem-settings">
            <Section
                title="Status"
                description="What memory is doing for the chat that is open right now."
                actions={
                    indexChip ? <StatusChip tone={indexChip.tone}>{indexChip.label}</StatusChip> : null
                }
            >
                <Switch
                    label="Memory"
                    help="Indexes chat messages and injects the most relevant old fragments into each prompt."
                    checked={settings.enabled}
                    onChange={(enabled) => commit({ enabled })}
                />

                <div className="sem-facts">
                    <Fact
                        label="Model"
                        value={
                            <span className="sem-fact-inline">
                                <span className="sem-strong">{model.name}</span>
                                <small>
                                    {model.source === "api" ? "API" : "Local"}
                                    {model.dims ? ` · ${model.dims} dims` : ""}
                                </small>
                            </span>
                        }
                    />

                    <Fact
                        label="Index"
                        value={
                            <span className="sem-fact-inline">
                                <span>{index ? `${index.chunks} chunks` : "—"}</span>
                                {index && index.messagesPending > 0 ? (
                                    <small>· {index.messagesPending} pending</small>
                                ) : null}
                                {index && index.state === "model-mismatch" ? (
                                    <small className="sem-tone-warn">
                                        {index.detail ?? "Built with another model; re-index to use this one."}
                                    </small>
                                ) : null}
                            </span>
                        }
                    />

                    {local ? (
                        <>
                            <Fact
                                label="Process"
                                value={
                                    <span className="sem-fact-inline">
                                        {processChip ? (
                                            <StatusChip tone={processChip.tone}>{processChip.label}</StatusChip>
                                        ) : (
                                            <span>—</span>
                                        )}
                                        {process?.port ? <small>port {process.port}</small> : null}
                                    </span>
                                }
                            />
                            {process?.state === "ready" ? (
                                <Fact
                                    label="Device"
                                    value={<span className="sem-fact-inline">{deviceText(process)}</span>}
                                />
                            ) : null}
                            <Fact
                                label="Disk"
                                value={
                                    <div className="sem-disk">
                                        <div className="sem-disk-row">
                                            <span>llama.cpp</span>
                                            <small className="sem-mono">
                                                {installer?.binary.installed
                                                    ? formatBytes(installer.binary.bytesOnDisk)
                                                    : "not installed"}
                                            </small>
                                            {installer?.binary.installed ? (
                                                <ConfirmButton
                                                    label="Delete"
                                                    confirmLabel="Delete?"
                                                    disabled={busy !== null}
                                                    onConfirm={() =>
                                                        void runAdvanced(() => controller.host.removeBinary())
                                                    }
                                                />
                                            ) : null}
                                        </div>
                                        <div className="sem-disk-row">
                                            <span>Model</span>
                                            <small className="sem-mono">
                                                {installer?.model.installed
                                                    ? formatBytes(installer.model.bytesOnDisk)
                                                    : "not installed"}
                                            </small>
                                            {installer?.model.installed ? (
                                                <ConfirmButton
                                                    label="Delete"
                                                    confirmLabel="Delete?"
                                                    disabled={busy !== null}
                                                    onConfirm={() =>
                                                        void runAdvanced(() => controller.host.removeModels())
                                                    }
                                                />
                                            ) : null}
                                        </div>
                                    </div>
                                }
                            />
                        </>
                    ) : null}
                </div>

                {local && server.error ? (
                    <Note tone="warn">
                        The plugin&apos;s server tools are not reachable: {server.error.message}
                    </Note>
                ) : null}

                {visibleLastError ? (
                    <ErrorBox
                        message={visibleLastError}
                        onDismiss={() => setHiddenError(visibleLastError)}
                    />
                ) : null}
            </Section>

            <Section
                title="Setup"
                actions={
                    <Button
                        className="sem-section-button"
                        aria-expanded={setupOpen}
                        onClick={() => setSetupOpen((open) => !open)}
                    >
                        {setupOpen ? "Hide" : "Show"}
                    </Button>
                }
            >
                {setupOpen ? (
                    <>
                        <div className="sem-step">
                            <StepHead n={1} title="Where should embeddings run?" hint="Embedding turns text into vectors; the source decides where that happens." />
                            <div className="sem-choice-row">
                                <ChoiceCard
                                    title="On this computer"
                                    description="Installs llama.cpp and a small model once. No API key, no chat text leaves the device."
                                    active={local}
                                    onClick={() => commit({ source: "local" })}
                                />
                                <ChoiceCard
                                    title="OpenAI-compatible API"
                                    description="Uses an embeddings endpoint you already pay for. Message text is sent to it."
                                    active={!local}
                                    onClick={() => commit({ source: "api" })}
                                />
                            </div>
                        </div>

                        {local ? (
                            <div className="sem-step">
                                <StepHead n={2} title="Pick a model and where it runs" hint="Both models are checked for this plugin; the default also runs on CPU." />
                                <div className="sem-choice-row">
                                    <ChoiceCard
                                        title={`Default · ${modelCard("default")}`}
                                        description="Light and fine on CPU. The safe pick for weak machines."
                                        active={settings.profile === "default"}
                                        onClick={() => commit({ profile: "default" })}
                                    />
                                    <ChoiceCard
                                        title={`Strong · ${modelCard("strong")}`}
                                        description="Better retrieval, but needs a capable GPU to be usable."
                                        active={settings.profile === "strong"}
                                        onClick={() => commit({ profile: "strong" })}
                                    />
                                </div>
                                {settings.profile === "custom" ? (
                                    <Note>
                                        A custom model is selected in Advanced. Pick one of the cards above to
                                        go back to a tested model.
                                    </Note>
                                ) : null}

                                <div className="sem-field">
                                    <span className="sem-field-label">Device</span>
                                    <Segmented
                                        ariaLabel="Device"
                                        value={settings.device}
                                        onChange={(device) => commit({ device })}
                                        options={[
                                            { value: "auto", label: "Auto" },
                                            { value: "gpu", label: "GPU" },
                                            { value: "cpu", label: "CPU" },
                                        ]}
                                    />
                                    <small className="sem-field-help">{DEVICE_HELP[settings.device]}</small>
                                    {backendGap ? (
                                        <small className="sem-field-help sem-tone-warn">{backendGap}</small>
                                    ) : null}
                                </div>
                            </div>
                        ) : (
                            <div className="sem-step">
                                <StepHead n={2} title="Connect the endpoint" hint="Any endpoint that implements POST /embeddings works." />
                                <div className="sem-fields">
                                    <TextField
                                        label="Base URL"
                                        value={settings.api.baseUrl}
                                        placeholder="https://api.openai.com/v1"
                                        mono
                                        help="Embeddings are requested at {base URL}/embeddings."
                                        onCommit={(baseUrl) => commit({ api: { ...settings.api, baseUrl } })}
                                    />
                                    <PasswordField
                                        label="API key"
                                        value={settings.api.apiKey}
                                        placeholder="sk-..."
                                        onCommit={(apiKey) => commit({ api: { ...settings.api, apiKey } })}
                                    />
                                    <TextField
                                        label="Model"
                                        value={settings.api.model}
                                        placeholder="text-embedding-3-small"
                                        mono
                                        onCommit={(modelId) => commit({ api: { ...settings.api, model: modelId } })}
                                    />
                                </div>

                                <Note tone="warn">
                                    <span>
                                        Your chat messages will be sent to{" "}
                                        <strong>{hostLabel(settings.api.baseUrl)}</strong> to be embedded.
                                    </span>
                                    <Switch
                                        label="I understand"
                                        checked={settings.apiNoticeAccepted}
                                        onChange={(apiNoticeAccepted) => commit({ apiNoticeAccepted })}
                                    />
                                </Note>
                            </div>
                        )}

                        <div className="sem-step">
                            <StepHead n={3} title="Finish setup" hint={local ? "Downloads once, starts llama-server, then embeds a test sentence." : "Sends one test sentence to your endpoint."} />

                            {local && !settings.downloadNoticeAccepted ? (
                                <ThirdPartyNotice
                                    settings={settings}
                                    disabled={busy !== null || !settings.enabled}
                                    onAccept={() => {
                                        commit({ downloadNoticeAccepted: true });
                                        void installAndTest();
                                    }}
                                />
                            ) : (
                            <div className="sem-button-row">
                                <Button
                                    variant="primary"
                                    disabled={
                                        busy !== null ||
                                        (!local && !settings.apiNoticeAccepted) ||
                                        !settings.enabled
                                    }
                                    title={
                                        !settings.enabled
                                            ? "Turn memory on first."
                                            : !local && !settings.apiNoticeAccepted
                                              ? "Accept the notice above first."
                                              : undefined
                                    }
                                    onClick={() =>
                                        busy ? undefined : local ? void installAndTest() : void runTest("install")
                                    }
                                >
                                    {busy === "install" || busy === "test"
                                        ? "Working…"
                                        : local
                                          ? partInstalled
                                              ? "Continue setup"
                                              : "Install and test"
                                          : "Test"}
                                </Button>
                                {busy === "install" ? (
                                    <Button onClick={() => void cancelInstall()}>Cancel</Button>
                                ) : null}
                            </div>
                            )}

                            {stage ? (
                                <small className="sem-stage" role="status">
                                    <span className="sem-stage-step">
                                        Step {stageIndex + 1} of {stages.length}
                                    </span>
                                    <span>{STAGE_TEXT[stage]}…</span>
                                </small>
                            ) : null}

                            {!settings.enabled ? (
                                <small className="sem-reason">Turn memory on first.</small>
                            ) : null}
                            {!local && !settings.apiNoticeAccepted ? (
                                <small className="sem-reason">Accept the notice above first.</small>
                            ) : null}

                            <TaskRow
                                label="Installing llama.cpp"
                                task={installer?.binary.task}
                                onCancel={() => void cancelInstall()}
                            />
                            <TaskRow
                                label="Downloading the model"
                                task={installer?.model.task}
                                onCancel={() => void cancelInstall()}
                            />

                            {outcome ? (
                                <div className="sem-ok" role="status">
                                    OK: {outcome.dim} dimensions, {Math.round(outcome.latencyMs)} ms
                                </div>
                            ) : null}

                            {notice ? <Note tone="warn">{notice}</Note> : null}

                            {failureAt("install") ? (
                                <ErrorBox
                                    message={failureAt("install")!.message}
                                    hint={failureAt("install")!.hint}
                                    onDismiss={() => setFailure(null)}
                                />
                            ) : null}
                        </div>

                        <div className="sem-step">
                            <StepHead n={4} title="Build the index" hint="Indexes the messages of the open chat so older events can be found later." />
                            <div className="sem-button-row">
                                <Button
                                    disabled={indexReason !== null}
                                    title={indexReason ?? undefined}
                                    onClick={() => void indexChat()}
                                >
                                    {progress.running ? "Indexing…" : "Index this chat"}
                                </Button>
                                {progress.running ? (
                                    <Button onClick={() => controller.cancelIndexing()}>Cancel</Button>
                                ) : null}
                            </div>
                            {indexReason ? <small className="sem-reason">{indexReason}</small> : null}

                            {progress.running ? (
                                <ProgressBar
                                    value={progress.done}
                                    max={progress.total}
                                    label={`${progress.done} of ${progress.total} messages · ${progress.chunks} chunks`}
                                    detail={progress.cancelling ? "Cancelling…" : undefined}
                                />
                            ) : null}

                            {failureAt("index") ? (
                                <ErrorBox
                                    message={failureAt("index")!.message}
                                    hint={failureAt("index")!.hint}
                                    onDismiss={() => setFailure(null)}
                                />
                            ) : null}
                        </div>
                    </>
                ) : (
                    <p className="sem-collapsed-hint">
                        {settings.setupComplete
                            ? "Setup is finished. Use Show to change the source, pick another model, or run the test again."
                            : "Open setup to choose a source and install the model."}
                    </p>
                )}
            </Section>

            <details className="sem-advanced">
                <summary>Advanced</summary>

                <div className="sem-advanced-body">
                    <div className="sem-group">
                        <h6>Embedding source</h6>
                        <div className="sem-fields">
                            <SelectField
                                label="Source"
                                value={settings.source}
                                options={[
                                    { value: "local", label: "Local (this computer)" },
                                    { value: "api", label: "OpenAI-compatible API" },
                                ]}
                                onChange={(source) => commit({ source })}
                            />

                            {local ? (
                                <>
                                    <SelectField
                                        label="Model profile"
                                        value={settings.profile}
                                        options={[
                                            { value: "default", label: "Default · EmbeddingGemma 2 · 176 MB" },
                                            { value: "strong", label: "Strong · Qwen3 Embedding 4B · 2.5 GB" },
                                            { value: "custom", label: "Custom (Hugging Face repo)" },
                                        ]}
                                        onChange={(profile) => commit({ profile })}
                                    />
                                    {settings.profile === "custom" ? (
                                        <>
                                            <TextField
                                                label="Repository"
                                                value={settings.customModel.repo}
                                                placeholder="org/repo-GGUF"
                                                mono
                                                onCommit={(repo) =>
                                                    commit({
                                                        customModel: { ...settings.customModel, repo },
                                                    })
                                                }
                                            />
                                            <TextField
                                                label="File"
                                                value={settings.customModel.file}
                                                placeholder="model-Q4_K_M.gguf"
                                                mono
                                                onCommit={(file) =>
                                                    commit({
                                                        customModel: { ...settings.customModel, file },
                                                    })
                                                }
                                            />
                                            <SelectField
                                                label="Pooling"
                                                value={settings.customModel.pooling}
                                                options={[
                                                    { value: "mean", label: "Mean" },
                                                    { value: "last", label: "Last token" },
                                                ]}
                                                onChange={(pooling) =>
                                                    commit({
                                                        customModel: { ...settings.customModel, pooling },
                                                    })
                                                }
                                            />
                                            <SelectField
                                                label="Text prefixes"
                                                help="How this model expects queries and stored text to be formatted."
                                                value={settings.customModel.prefixProfile}
                                                options={PREFIX_OPTIONS}
                                                onChange={(prefixProfile) =>
                                                    commit({
                                                        customModel: {
                                                            ...settings.customModel,
                                                            prefixProfile,
                                                        },
                                                    })
                                                }
                                            />
                                        </>
                                    ) : null}
                                </>
                            ) : (
                                <>
                                    <TextField
                                        label="Base URL"
                                        value={settings.api.baseUrl}
                                        mono
                                        onCommit={(baseUrl) => commit({ api: { ...settings.api, baseUrl } })}
                                    />
                                    <PasswordField
                                        label="API key"
                                        value={settings.api.apiKey}
                                        onCommit={(apiKey) => commit({ api: { ...settings.api, apiKey } })}
                                    />
                                    <TextField
                                        label="Model"
                                        value={settings.api.model}
                                        mono
                                        onCommit={(modelId) => commit({ api: { ...settings.api, model: modelId } })}
                                    />
                                    <SelectField
                                        label="Text prefixes"
                                        value={settings.api.prefixProfile}
                                        options={PREFIX_OPTIONS}
                                        onChange={(prefixProfile) =>
                                            commit({ api: { ...settings.api, prefixProfile } })
                                        }
                                    />
                                </>
                            )}

                            <TextAreaField
                                label="Qwen3 query instruction"
                                help="Only used when the model expects the Qwen3 prefix."
                                value={settings.qwenTask}
                                rows={2}
                                placeholder={DEFAULT_QWEN3_TASK}
                                onCommit={(qwenTask) => commit({ qwenTask })}
                            />
                        </div>
                    </div>

                    <div className="sem-group">
                        <h6>Local runtime</h6>
                        <div className="sem-fields">
                            <TextField
                                label="Binary path"
                                value={settings.binaryPath}
                                placeholder="Automatic download"
                                mono
                                onCommit={(binaryPath) => commit({ binaryPath })}
                            />
                            <TextField
                                label="Model path"
                                value={settings.modelPath}
                                placeholder="Automatic download"
                                mono
                                onCommit={(modelPath) => commit({ modelPath })}
                            />
                            <p className="sem-help-wide">
                                Manual paths override the downloaded files. Clear a path to go back to
                                downloading.
                            </p>
                            <TextField
                                label="Release tag"
                                value={settings.releaseTag}
                                placeholder="b12686"
                                mono
                                help="llama.cpp release tag, or latest."
                                onCommit={(releaseTag) => commit({ releaseTag })}
                            />
                            <SelectField<GpuBackendChoice>
                                label="GPU backend"
                                help="Which llama.cpp build to download. Auto picks the one that matches your GPU."
                                value={settings.gpuBackend}
                                options={[
                                    { value: "auto", label: "Auto" },
                                    { value: "cuda", label: "CUDA" },
                                    { value: "vulkan", label: "Vulkan" },
                                ]}
                                onChange={(gpuBackend) => commit({ gpuBackend })}
                            />
                            <SelectField<DeviceChoice>
                                label="Device"
                                help={DEVICE_HELP[settings.device]}
                                value={settings.device}
                                options={[
                                    { value: "auto", label: "Auto" },
                                    { value: "gpu", label: "GPU" },
                                    { value: "cpu", label: "CPU" },
                                ]}
                                onChange={(device) => commit({ device })}
                            />
                            <NumberField
                                label="Context size"
                                value={settings.ctx}
                                min={512}
                                max={32768}
                                step={512}
                                suffix="tokens"
                                onCommit={(ctx) => commit({ ctx })}
                            />
                            <NumberField
                                label="Idle shutdown"
                                value={settings.idleMinutes}
                                min={0}
                                max={1440}
                                suffix="min"
                                help="0 keeps the local server running."
                                onCommit={(idleMinutes) => commit({ idleMinutes })}
                            />
                        </div>
                    </div>

                    <div className="sem-group">
                        <h6>Indexing and retrieval</h6>
                        <div className="sem-fields">
                            <NumberField
                                label="Chunk size"
                                value={settings.chunkSize}
                                min={100}
                                max={2000}
                                step={50}
                                onCommit={(chunkSize) => commit({ chunkSize })}
                            />
                            <NumberField
                                label="Chunk overlap"
                                value={settings.chunkOverlap}
                                min={0}
                                max={Math.floor(settings.chunkSize / 2)}
                                step={10}
                                onCommit={(chunkOverlap) => commit({ chunkOverlap })}
                            />
                            <NumberField
                                label="Query messages"
                                value={settings.queryMessages}
                                min={1}
                                max={10}
                                onCommit={(queryMessages) => commit({ queryMessages })}
                            />
                            <NumberField
                                label="Top-K memories"
                                value={settings.topK}
                                min={1}
                                max={20}
                                onCommit={(topK) => commit({ topK })}
                            />
                            <NumberField
                                label="Similarity threshold"
                                value={settings.threshold}
                                min={0}
                                max={1}
                                step={0.05}
                                integer={false}
                                autoPlaceholder={`Auto (${autoThreshold.toFixed(2)} for this model)`}
                                help="Higher keeps only closer matches. Auto uses the value calibrated for the active model."
                                onAuto={() => commit({ threshold: null })}
                                onCommit={(threshold) => commit({ threshold })}
                            />
                            <NumberField
                                label="Keep recent messages"
                                value={settings.retain}
                                min={0}
                                max={50}
                                help="Excluded from retrieval; they are already in the prompt."
                                onCommit={(retain) => commit({ retain })}
                            />
                            <NumberField
                                label="Embedding timeout"
                                value={settings.timeoutMs}
                                min={1000}
                                max={60000}
                                step={500}
                                suffix="ms"
                                onCommit={(timeoutMs) => commit({ timeoutMs })}
                            />
                        </div>
                        <Switch
                            label="Index new messages automatically"
                            help="Indexes the backlog of an open chat in the background."
                            checked={settings.autoIndexBacklog}
                            onChange={(autoIndexBacklog) => commit({ autoIndexBacklog })}
                        />
                    </div>

                    <div className="sem-group">
                        <h6>Injection</h6>
                        <div className="sem-fields">
                            <NumberField
                                label="Injection depth"
                                value={settings.depth}
                                min={0}
                                max={50}
                                help="Messages up from the end where the block is inserted."
                                onCommit={(depth) => commit({ depth })}
                            />
                            <NumberField
                                label="Memory token cap"
                                value={settings.maxMemoryTokens}
                                min={50}
                                max={4000}
                                step={50}
                                onCommit={(maxMemoryTokens) => commit({ maxMemoryTokens })}
                            />
                        </div>
                        <TextAreaField
                            label="Injection template"
                            value={settings.template}
                            rows={5}
                            help="Must contain {{memories}}; the retrieved fragments replace that marker."
                            onCommit={saveTemplate}
                        />
                    </div>

                    {local && partInstalled ? (
                        <div className="sem-group">
                            <h6>Program and model files</h6>
                            {settings.downloadNoticeAccepted ? (
                                <div className="sem-button-row">
                                    <Button
                                        variant="primary"
                                        disabled={busy !== null}
                                        onClick={() => void installAndTest()}
                                    >
                                        Install and test
                                    </Button>
                                    <small className="sem-reason">Only the missing parts are downloaded.</small>
                                </div>
                            ) : (
                                <ThirdPartyNotice
                                    settings={settings}
                                    disabled={busy !== null || !settings.enabled}
                                    onAccept={() => {
                                        commit({ downloadNoticeAccepted: true });
                                        void installAndTest();
                                    }}
                                />
                            )}
                        </div>
                    ) : null}

                    <div className="sem-group">
                        <h6>Maintenance</h6>
                        <div className="sem-button-row">
                            <Button disabled={busy !== null} onClick={() => void runTest("advanced")}>
                                Test
                            </Button>
                            <ConfirmButton
                                label="Re-index"
                                confirmLabel="Rebuild the index?"
                                variant="primary"
                                disabled={busy !== null || indexReason !== null}
                                reason={indexReason ?? undefined}
                                onConfirm={() => void runAdvanced(() => controller.reindex())}
                            />
                            <ConfirmButton
                                label="Purge index"
                                confirmLabel="Delete the index?"
                                disabled={busy !== null || !index || index.chunks === 0}
                                reason={index?.chunks ? undefined : "There is no index to delete."}
                                onConfirm={() => void runAdvanced(() => controller.purge())}
                            />
                        </div>

                        {progress.running ? (
                            <ProgressBar
                                value={progress.done}
                                max={progress.total}
                                label={`${progress.done} of ${progress.total} messages`}
                                detail={progress.cancelling ? "Cancelling…" : undefined}
                            />
                        ) : null}

                        {outcome ? (
                            <div className="sem-ok" role="status">
                                OK: {outcome.dim} dimensions, {Math.round(outcome.latencyMs)} ms
                            </div>
                        ) : null}

                        {failureAt("advanced") ? (
                            <ErrorBox
                                message={failureAt("advanced")!.message}
                                hint={failureAt("advanced")!.hint}
                                onDismiss={() => setFailure(null)}
                            />
                        ) : null}
                    </div>

                    <div className="sem-group">
                        <h6>Privacy</h6>
                        <p className="sem-privacy">
                            With the API source, message text is sent to the endpoint to be embedded, and the
                            API key is stored in plain local JSON in this plugin&apos;s settings. With the local
                            source, the embedding program runs on this computer, listens on 127.0.0.1 only, and
                            chat text never leaves this device. Downloads come only from GitHub and Hugging
                            Face.
                        </p>
                    </div>
                </div>
            </details>
        </div>
    );
}

/* ------------------------------------------------------------------ sub-blocks */

function StepHead({ n, title, hint }: { n: number; title: string; hint?: string }) {
    return (
        <div className="sem-step-copy">
            <div className="sem-step-head">
                <span className="sem-step-num" aria-hidden="true">
                    {n}
                </span>
                <h6>{title}</h6>
            </div>
            {hint ? <p className="sem-step-hint">{hint}</p> : null}
        </div>
    );
}

function ChoiceCard({
    title,
    description,
    active,
    onClick,
}: {
    title: string;
    description: string;
    active: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            aria-pressed={active}
            className={cx("sem-choice", active && "is-active")}
            onClick={onClick}
        >
            <span className="sem-choice-title">
                {title}
                {active ? (
                    <span className="sem-choice-check" aria-hidden="true">
                        ✓
                    </span>
                ) : null}
            </span>
            <small>{description}</small>
        </button>
    );
}

function Fact({ label, value }: { label: string; value: ComponentChildren }) {
    return (
        <div className="sem-fact">
            <span className="sem-fact-label">{label}</span>
            <div className="sem-fact-value">{value}</div>
        </div>
    );
}

function TaskRow({
    label,
    task,
    onCancel,
}: {
    label: string;
    task: TaskStatus | undefined;
    onCancel: () => void;
}) {
    // Hooks cannot sit after the early returns below, so they come first.
    const [dismissed, setDismissed] = useState<string | null>(null);

    if (!task || (task.state !== "running" && task.state !== "error")) {
        return null;
    }

    if (task.state === "error") {
        const message = task.error ?? `${label} failed.`;

        if (message === dismissed) {
            return null;
        }

        return (
            <ErrorBox
                message={message}
                hint="Check the connection, then run Install and test again."
                onDismiss={() => setDismissed(message)}
            />
        );
    }

    return (
        <div className="sem-task">
            <ProgressBar
                value={task.received}
                max={task.total}
                label={`${label}: ${formatBytes(task.received)} of ${formatBytes(task.total)}`}
                detail={task.message}
            />
            <Button onClick={onCancel}>Cancel</Button>
        </div>
    );
}

/* --------------------------------------------------------------------- helpers */

const DEVICE_HELP: Record<DeviceChoice, string> = {
    auto: "GPU when one is usable, otherwise CPU.",
    gpu: "Never falls back to CPU: it fails loudly if no GPU works.",
    cpu: "Slower, but works on every machine.",
};

/** The wizard only offers the two tested profiles; a custom model stays as chosen. */
function wizardProfile(settings: PluginSettings): ModelProfileChoice {
    return settings.profile === "strong" ? "strong" : settings.profile === "custom" ? "custom" : "default";
}

function modelCard(id: "default" | "strong"): string {
    const spec = LOCAL_MODELS[id];

    return id === "default" ? `${formatBytes(spec.bytes)}, light` : `${formatBytes(spec.bytes)}, needs a GPU`;
}

/** What the download pulls from Hugging Face, for the consent notice. */
function downloadTargets(settings: PluginSettings): { repo: string; file: string; bytes?: number } {
    if (settings.profile === "custom") {
        return {
            repo: settings.customModel.repo || "your custom repository",
            file: settings.customModel.file || "the file you name",
        };
    }

    const spec = settings.profile === "strong" ? LOCAL_MODELS.strong : LOCAL_MODELS.default;

    return { repo: spec.repo, file: spec.file, bytes: spec.bytes };
}

/**
 * The cut-off retrieval will actually use, so the Advanced field can show it while
 * the stored value stays `null` (auto).
 */
function autoThresholdFor(settings: PluginSettings, controller: MemoryController): number {
    try {
        return effectiveThreshold(settings, controller.embedder().model.prefixProfile);
    } catch {
        const fallback = settings.source === "local" ? settings.profile === "strong"
            ? LOCAL_MODELS.strong.prefixProfile
            : LOCAL_MODELS.default.prefixProfile
            : settings.api.prefixProfile;

        return effectiveThreshold(settings, fallback);
    }
}

const BACKEND_LABELS: Record<Backend, string> = {
    cpu: "CPU",
    cuda: "CUDA",
    vulkan: "Vulkan",
    metal: "Metal",
};

/**
 * The server reports which builds exist, which is not the same as "usable now":
 * a Vulkan build on a machine asked for CPU still needs a download.
 */
function deviceGapText(backends: Backend[] | undefined, device: DeviceChoice): string | null {
    const usable = backends ?? [];

    if (usable.length === 0) {
        return null;
    }

    const listed = usable.map((backend) => BACKEND_LABELS[backend]).join(" and ");
    const hasGpu = usable.some((backend) => backend !== "cpu");

    if (device === "cpu" && !usable.includes("cpu")) {
        return `${listed} installed - CPU build missing`;
    }

    if (device === "gpu" && !hasGpu) {
        return `${listed} installed - GPU build missing`;
    }

    return null;
}

/**
 * Third-party download consent. Shown before the first install so nobody is
 * surprised that setup reaches outside the machine.
 */
function ThirdPartyNotice({
    settings,
    disabled,
    onAccept,
}: {
    settings: PluginSettings;
    disabled: boolean;
    onAccept: () => void;
}) {
    const model = downloadTargets(settings);

    return (
        <div className="sem-tp" role="note">
            <p className="sem-tp-text">
                Installing downloads files from third parties: the llama.cpp program from{" "}
                <strong>github.com</strong> (ggml-org/llama.cpp releases, about 25 MB (CPU) to 620 MB (NVIDIA
                CUDA)) and the embedding model from <strong>huggingface.co</strong> (
                <span className="sem-mono">
                    {model.repo}/{model.file}
                </span>
                {model.bytes ? `, ${formatBytes(model.bytes)}` : ", size known once the download starts"}).
                Nothing else leaves your device, and your chat text is never sent anywhere with the local
                source.
            </p>
            <div className="sem-button-row">
                <Button variant="primary" disabled={disabled} onClick={onAccept}>
                    Download and install
                </Button>
            </div>
        </div>
    );
}

function describeModel(settings: PluginSettings): { name: string; source: string; dims?: number } {
    if (settings.source === "api") {
        return { name: settings.api.model || "no model set", source: "api" };
    }

    if (settings.modelPath) {
        return { name: baseName(settings.modelPath), source: "local" };
    }

    if (settings.profile === "custom") {
        return {
            name: settings.customModel.file || settings.customModel.repo || "custom model",
            source: "local",
        };
    }

    const spec = LOCAL_MODELS[settings.profile];

    return { name: spec.label.split(" (")[0], source: "local", dims: spec.dim };
}

function deviceText(process: ProcessStatus | undefined): string {
    if (!process || process.state !== "ready") {
        return "—";
    }

    const device = process.device ?? process.backend ?? "unknown";

    if (process.layersTotal !== undefined && process.layersOffloaded !== undefined) {
        return `${device} · ${process.layersOffloaded}/${process.layersTotal} layers on GPU`;
    }

    return device;
}

function disabledReason(
    settings: PluginSettings,
    index: IndexStatus | null,
    progress: IndexerProgress,
    busy: boolean,
    chatId: string | null,
): string | null {
    if (!settings.enabled) {
        return "Turn memory on first.";
    }

    if (!chatId && !index?.chatId) {
        return "Open a chat first.";
    }

    if (settings.source === "api" && !settings.apiNoticeAccepted) {
        return "Accept the API notice in setup first.";
    }

    if (progress.running) {
        return "Indexing is already running.";
    }

    if (busy) {
        return "Working…";
    }

    if (index && index.messagesTotal === 0) {
        return "This chat has no messages to index yet.";
    }

    return null;
}

/** Same rule as settings.ts: a bare host must not be parsed as a protocol. */
function hostLabel(url: string): string {
    try {
        const parsed = new URL(url);

        if (parsed.host) {
            return parsed.host;
        }
    } catch {
        // No scheme: fall through.
    }

    return url.replace(/^[a-z]+:\/\//i, "").replace(/\/.*$/, "") || "that host";
}

function baseName(path: string): string {
    const parts = path.split(/[\\/]/);

    return parts[parts.length - 1] || path;
}

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
import { useState } from "preact/hooks";

import type { IndexStatus, IndexerProgress, MemoryController } from "./controller";
import { DEFAULT_QWEN3_TASK, LOCAL_MODELS } from "./models";
import type { PluginSettings } from "./settings";
import { serverConfigOf } from "./settings";
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
     * Step 3 of the wizard. The settings push is debounced, so the config is saved
     * here first: the installer/launcher must already know which model and backend
     * were just picked, otherwise it downloads or starts the previous choice.
     */
    async function installAndTest() {
        if (busy) {
            return;
        }

        setBusy("install");
        setFailure(null);
        setOutcome(null);

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

            if (settings.source === "local") {
                let status = await controller.host.status();

                if (!status.installer.binary.installed) {
                    await controller.host.installBinary();
                    status = await controller.host.status();

                    if (!status.installer.binary.installed) {
                        throw new Error("The llama.cpp binary did not finish installing.");
                    }
                }

                if (!status.installer.model.installed) {
                    await controller.host.downloadModel();
                    status = await controller.host.status();

                    if (!status.installer.model.installed) {
                        throw new Error("The model file did not finish downloading.");
                    }
                }

                await controller.host.start();
            }

            const result = await controller.test();

            setOutcome(result);
            // The wizard stays collapsed on later visits; the app's settings store
            // survives reloads and updates, unlike a browser-local flag.
            commit({ setupComplete: true });
        } catch (error) {
            setFailure({ at: "install", parts: errorParts(error) });
        } finally {
            setBusy(null);
            reloadAll();
        }
    }

    async function runTest(at: "install" | "advanced") {
        if (busy) {
            return;
        }

        setBusy("test");
        setFailure(null);
        setOutcome(null);

        try {
            const result = await controller.test();

            setOutcome(result);
            commit({ setupComplete: true });
            reloadAll();
        } catch (error) {
            setFailure({ at, parts: errorParts(error) });
        } finally {
            setBusy(null);
        }
    }

    async function cancelInstall() {
        try {
            if (installer?.binary.task.state === "running") {
                await controller.host.cancelBinary();
            }

            if (installer?.model.task.state === "running") {
                await controller.host.cancelModel();
            }
        } finally {
            reloadAll();
        }
    }

    async function indexChat() {
        if (busy) {
            return;
        }

        setBusy("index");
        setFailure(null);

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

                {lastError ? <ErrorBox message={lastError} /> : null}
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
                                          ? "Install and test"
                                          : "Test"}
                                </Button>
                                {busy === "install" ? (
                                    <Button onClick={() => void cancelInstall()}>Cancel</Button>
                                ) : null}
                            </div>

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

                            {failureAt("install") ? (
                                <ErrorBox
                                    message={failureAt("install")!.message}
                                    hint={failureAt("install")!.hint}
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
                            />
                        ) : null}
                    </div>

                    <div className="sem-group">
                        <h6>Privacy</h6>
                        <p className="sem-privacy">
                            With the API source, message text is sent to the endpoint to be embedded, and the
                            API key is stored in plain local JSON in this plugin&apos;s settings. With the local
                            source, chat text never leaves this device. Downloads come only from GitHub and
                            Hugging Face.
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
    if (!task || (task.state !== "running" && task.state !== "error")) {
        return null;
    }

    if (task.state === "error") {
        return (
            <ErrorBox
                message={task.error ?? `${label} failed.`}
                hint="Check the connection, then run Install and test again."
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

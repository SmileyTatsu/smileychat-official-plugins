# Smiley Embeddings Memory — design

Status: **approved** (decisions of the review are folded in below: `smiley-` folder prefix, the Gemma profile is named `default`, CPU/GPU is a user choice).
Plugin id: `smiley-embeddings-memory`. Folder: `plugins/smiley-embeddings-memory/` (the other four plugins drop the prefix in their folder name; this one keeps it by request).

## Dependency on an unmerged SmileyChat branch

This plugin needs the server-module runtime (`serverMain`, `api.host`, `host.spawn`), which lives in SmileyChat's branch `feat/plugin-server-runtime` and is **not in SmileyChat `main` yet** (`main` is 0.0.5 and its typings have no `PluginHostApi`). Consequences:

- `types/smileychat-plugin-api.d.ts` is a byte copy of that branch at commit `eb278f9`. `bun run update-types` defaults to `main` and would replace it with typings without `api.host`, breaking `typecheck`: until the branch is merged, refresh with `bun run update-types --ref feat/plugin-server-runtime` (or a commit), and go back to the default afterwards.
- `minAppVersion` is `0.0.6`, the version the docs assign to server modules; the plugin reports as incompatible on 0.0.5 builds. It must not be tagged or listed in the registry before that SmileyChat release exists. Note that `release.yml` packs every plugin in `plugins/` on any `v*` tag, so a tag for another plugin would also attach this ZIP.
- The end-to-end checks ran against a copy of that branch with only the copy's `package.json` version raised to 0.0.6.

Semantic memory for long chats: embed the chat's messages in the background, and on every turn retrieve the most relevant *old* fragments and inject them into the prompt as "memories of past events".

## 1. What was measured (Windows 11, RTX 5060 Ti 16 GB, llama.cpp `b12686`)

All numbers come from real runs; scratch files live in `%LOCALAPPDATA%\hermes\cache\scratch\{embeddings-spike,index-bench}`.

| Model file | Download | dim / pooling | VRAM (log breakdown) | Start to `/health` | emb/s GPU b32 | emb/s CPU b8 | Retrieval (4 paraphrase queries) |
|---|---|---|---|---|---|---|---|
| `unsloth/embeddinggemma-2-GGUF` `UD-Q4_K_XL` | 175,673,856 B | 768 / mean | 133 MiB | 1.6–3.1 s | 203 | **18.4** | 4/4, margin ≥ 0.19 |
| same repo `Q8_0` | 309,855,520 B | 768 / mean | 197 MiB | 1.5–2.8 s | 217 | 11.0 | 4/4 |
| `Qwen/Qwen3-Embedding-4B-GGUF` `Q4_K_M` | 2,496,703,776 B | 2560 / last | 3612 MiB at `-c 8192`; **2742 MiB at `-c 2048`** | 1.7–5.8 s (18.7 s first load) | 57 | not measured | 4/4, margin ≥ 0.27 |
| same repo `Q8_0` | 4,279,660,224 B | 2560 / last | 5312 MiB | 8.9 s (32–36 s on the first two loads of a fresh file) | 61 | not measured | 4/4, no better than Q4 |
| `Qwen3-Embedding-0.6B` `Q8_0` (spike only) | 639,150,592 B | 1024 / last | — | 1.2 s | 261 | 34 | cos 0.66 similar vs 0.16–0.22 unrelated |

Decisions that follow:

- **`default` profile: EmbeddingGemma-2 `UD-Q4_K_XL`** (176 MB, usable on CPU, about 5.5 s to index 100 chunks); it is what the wizard preselects and suits weak machines and phones. **`strong` profile: Qwen3-Embedding-4B `Q4_K_M`** for powerful machines; Q8_0 costs 1.7 GB more and retrieves no better here. Both repos exist (the prompt's doubt about `embeddinggemma-2` was unfounded); both are Apache-2.0.
- **Pooling is passed explicitly**: `--pooling last` (Qwen3), `--pooling mean` (Gemma). L2 normalisation is already done by `llama-server` (`--embd-normalize 2` default; norm measured 1.000000).
- **Prefixes** (verbatim from the model cards / `config_sentence_transformers.json`):
  - Qwen3 query: `Instruct: {task}\nQuery:{text}`; document: none. Task default: `Given a chat message, retrieve relevant past events from the same conversation`.
  - Gemma-2 query: `task: search result | query: {text}`; document: `title: none | text: {text}`.
- **Qwen3 EOS: the plugin must NOT append `<|endoftext|>`.** One recon report said it was required; I tested it: embedding the text equals embedding the pre-tokenised ids plus EOS (cosine 1.0), so the server already appends it, and a literal suffix adds a second one (cosine 0.990 vs the correct vector).
- **`-c` must be set small.** Qwen 4B reserves 1152 MiB of context at `-c 8192` for nothing. Plan: `-c 4096 -b 4096 -ub 4096 -np 1`, to be verified in the server-manager task (the spike only showed that the *sum* of tokens in one request must fit the context, so batch size and `-c` are coupled).
- **GPU info needs verbosity.** At the default log level `llama-server` prints no device or offload lines. With `-lv 4` it prints `  - CUDA0 : NVIDIA GeForce RTX 5060 Ti (...)` and `load_tensors: offloaded 29/29 layers to GPU`. `host.spawn` drains stdout/stderr into `logger.debug` and exposes no stream, so the plugin passes `--log-file <dataDir>/llama-server.log` and parses that (to verify that `--log-file` honours `-lv 4`).
- **Index weight** (random unit vectors, 768/1024/2560 dims, medians of 5):

  | 20,000 chunks | file | stringify | load (parse+decode) | heap | search one query |
  |---|---|---|---|---|---|
  | 768d JSON number arrays | 451 MB | 735 ms | 814 ms | 1.8 GB | 18 ms |
  | 768d base64 float32 | 88 MB | 46 ms | 139 ms | 343 MB | 20 ms |
  | 2560d JSON number arrays | 1485 MB | 6.7 s | 2.3 s | 6.1 GB | 53 ms |
  | 2560d base64 float32 | 263 MB | 133 ms | 301 ms | 1.0 GB | 63 ms |
  | 2560d base64 int8 | 68 MB | 26 ms | 107 ms | 320 MB | 61 ms (top-10 agreement 0.97–0.98) |

  Server facts: no body-size check in `readJsonBody`, but Bun's default `maxRequestBodySize` (128 MiB) silently drops larger PUTs (no 413); every save is a full, pretty-printed rewrite (+42% on number arrays); reads stream the file untouched.

## 2. Architecture

```
browser (Preact, dist/index.js)                      server (Bun, dist/server.js, one self-contained ESM bundle)
  settings panel + wizard                              activate(host) -> { handle, dispose }
  header action -> modal (viewer / status / dry run)   /status /config /binary/* /models/* /embed /stop
  indexer (background)  --api.host.fetch-->            llama.cpp installer, model downloader,
  injector (registerPromptInjector)                    llama-server lifecycle (host.spawn, idle timer)
  embedding client: OpenAI API  | host /embed
  index store (api.storage)
```

Browser side (`src/`): `chunking.ts`, `similarity.ts`, `models.ts` (model catalog + prefixes), `embedding-client.ts`, `index-store.ts`, `indexer.ts`, `retrieval.ts`, `injection.ts`, `settings.ts`, `settings-panel.tsx`, `modal.tsx`, `log-buffer.ts`, `index.tsx`. Server side (`src/server/`): `releases.ts` (asset selection), `download.ts`, `unzip.ts`, `llama-process.ts`, `config.ts`, `main.ts`. One writer owns each file.

### 2.1 Embedding sources

- **A. OpenAI-compatible API**: `POST {baseUrl}/embeddings` `{ model, input: string[], encoding_format: "float" }` from the browser, batches of 16–32, retries with backoff on 429/5xx, hard check that the returned dimension matches the index. Subject to CORS like the app's other direct provider calls. The first use shows an explicit "your messages will be sent to {host}" notice.
- **B. Local `llama-server`**, owned by the server module. The browser calls `api.host.fetch("/embed", { texts })`; the module ensures the process is up (cold start on demand), forwards to `http://127.0.0.1:<port>/v1/embeddings` and returns the vectors. The browser never talks to `127.0.0.1` directly.

Prefixes are applied in the browser (`models.ts`), so a custom or API model can use any of the prefix profiles (`qwen3`, `gemma`, `none`).

### 2.2 Server module

Routes (all under `/api/plugins/smiley-embeddings-memory/host`, already protected by auth, CSRF, same-origin):

| Route | Purpose |
|---|---|
| `GET /status` | process state (`stopped/downloading/starting/ready/error`), pid, port, backend, device requested and used, layers offloaded/total, model, last used, last error; binary (tag, asset, bytes, path); models with disk bytes; running downloads with progress |
| `POST /config` | persists `{ binaryPath?, modelPath?, ctx, idleMinutes, device: "auto"|"gpu"|"cpu", gpuBackend: "auto"|"cuda"|"vulkan" }` to `dataDir/config.json` |
| `POST /binary/install`, `POST /binary/cancel`, `DELETE /binary` | llama.cpp download, unzip, delete |
| `POST /models/download`, `POST /models/cancel`, `DELETE /models/:id` | model files |
| `POST /embed` | `{ texts: string[] }` → `{ vectors: number[][] }` (starts the process if needed) |
| `POST /stop` | stop now |

Security rules: the module only ever executes (a) the `llama-server` it extracted under `dataDir/bin/<tag>/`, or (b) the manual path the user saved in config, which must be an existing file whose basename starts with `llama-server`. **Arguments are fixed in code** (`--embedding --pooling <profile> --host 127.0.0.1 --port <free> -ngl <n> -c <n> --no-webui --log-file … -lv 4`); no request field reaches a command line. Bound to `127.0.0.1` only. Downloads are restricted to hosts `api.github.com`, `github.com`, `*.githubusercontent.com`, `huggingface.co`, `*.hf.co`, redirects re-checked hop by hop; HF repo/file for custom models are regex-validated.

Binary selection (`releases.ts`, pure, unit-tested with fixture asset lists):

- Release: default pinned to the tested tag `b12686`; advanced setting accepts `latest` (newest `b####` from `GET /repos/ggml-org/llama.cpp/releases?per_page=30`; **`/releases/latest` is useless**, it returns a `v1.0.0` tag with no binaries).
- Compute device is a user setting, `auto` (default) / `gpu` / `cpu`, independent of the profile. `cpu` selects the `cpu` build (24 MB) and launches with `-ngl 0`; `gpu` selects the best GPU build for the platform and `-ngl 99` and fails with an actionable error if none can start (no silent CPU fallback); `auto` picks GPU when a GPU build starts and otherwise CPU. Binaries are kept per backend (`dataDir/bin/<tag>/<asset>/`), so switching device downloads the other build once and never deletes the first. A manual binary path is used as given and only `-ngl` changes. Status always shows the device actually used.
- Windows x64: NVIDIA (`nvidia-smi` reports CUDA ≥ 13) → `cuda-13.5` + `cudart-llama-bin-win-cuda-13.5` (216 + 407 MB); CUDA 12 → `cuda-12.4`; otherwise or by preference → `vulkan` (37 MB); no GPU → `cpu`. Linux: `ubuntu-cuda-12.8/13.5`, `ubuntu-vulkan-x64`, `ubuntu-x64`, `ubuntu-arm64`. macOS: `macos-arm64` (Metal) / `macos-x64`. Android: `android-arm64` exists; manual path remains the supported route on Termux.
- After install, a start probe. Under `auto`, if the GPU backend fails to start the module falls back CUDA → Vulkan → CPU and says so in status; under `gpu`, it reports the failure and stops.
- Download size is checked against the GitHub asset `size`; sha256 against the asset `digest` when present. Models: `.part` file with `Range` resume (verified: HF answers 206 and a resumed file hashes identically), size check, and sha256 against the `lfs.oid` from the HF tree API.
- ZIP extraction uses a bundled pure-JS unzip (`fflate`) with zip-slip checks; no dependency on `unzip`/`tar` being installed.
- Idle stop: a timer reset by every `/embed`; default 10 min; `0` = keep running. Cold start budget: the first load of a fresh multi-GB file took 32–36 s here (antivirus scan), so the start timeout is 120 s with a visible "starting" state. `activate()` does no I/O beyond reading config (15 s limit); `dispose()` stops the process and cancels downloads (5 s limit). Spawned via `host.spawn`, so the app kills it on disable, update, uninstall and exit. Short probes (`nvidia-smi`) use `Bun.spawn` with a 5 s timeout, since `host.spawn` exposes no output.

### 2.3 Index (per chat, never global)

Storage keys (one namespace per chat): `index-<chatId>` (manifest) and `index-<chatId>-<n>` (shards of ≤ 2000 chunks). **This deviates from "one key per chat" on purpose**: with 2560-d vectors a single key passes the 128 MiB PUT cap at about 2.4k chunks if stored as number arrays, and still means a 263 MB rewrite per saved message at 20k chunks when stored as base64. Sharding keeps appends to ~27 MB (2560d) / 9 MB (768d) and ~17 ms.

```ts
Manifest { version: 1, chatId, model: { id, source, dim, prefixProfile }, shards: number[], counts: number[],
           messages: { [messageId]: { hash: string, chunks: number } }, updatedAt }
Shard    { chunks: { id, messageId, ord, text, vec /* base64 float32 */ }[] }
registry key `index-registry`: { chatIds: string[] }   // lets us sweep orphans without reading every key
```

- Vectors are base64 float32 (5.1–5.7× smaller than JSON arrays, 16× faster to stringify). int8 is a documented `shortcut:` upgrade for ≥ 2560-d beyond ~20k chunks.
- `model.id` + `dim` are written at creation. If the active model differs, retrieval is skipped with `no match: index built with <A>, active <B>`, status says "model mismatch", and the UI offers **Re-index** (wipes then rebuilds). Vectors from different models are never mixed or compared.
- Whole index of the active chat is held decoded in memory (20k × 768d ≈ 343 MB peak while loading, ≈ 61 MB retained); other chats are not cached. `shortcut: whole-chat index in memory, >50k chunks needs ANN or on-disk search`.

### 2.4 Keeping the index coherent without hooks

The app has no chat-deleted, chat-forked or chat-switched events (only `registerMessageUpdateMiddleware` for edit/swipe/update, which does not fire for new or deleted messages). So correctness does not depend on events. A single `reconcile(chat)` compares `chat.messages` (active swipe text, `<msg>` tags stripped, image markers removed) with the manifest by `messageId` + content hash:

- in chat, not in index → embed; hash differs (edit, swipe) → drop its chunks, re-embed; in index, not in chat (deleted, regenerated) → drop.
- It runs on chat switch (`api.state.subscribe` on `activeChat.id`), debounced after message changes, after the message-update middleware fires, and at the start of every retrieval (cheap: hash compare only). Messages still streaming are skipped until complete.
- **Fork** = new chat id = empty index, built by reconcile; nothing is shared or copied. **Delete chat**: the index can't be deleted at that moment, so on activation (and when opening the viewer) the registry is swept against `api.actions.listChats()` and orphaned indexes are removed. Between deletion and the next sweep the orphan is inert (keyed by a chat id that never reappears). Documented, not prevented.
- Backlog of an existing chat is indexed automatically only for the local source; for source A the user clicks **Index all** after seeing the message/chunk count (it spends their API credit).

### 2.5 Retrieval and injection

On each prompt build (`registerPromptInjector`, awaited, wrapped so it **never throws** — an uncaught injector aborts the whole prompt build — and bounded by a timeout, default 8 s, returning `[]` with `no match: timeout`):

1. Take the last `N` messages (default 1) from `context.messages` and embed each as a query (one batched request); a chunk's score is its max over the queries. The default is 1 because, on three test chats, a second query message (the previous character reply) inherited high scores against repetitive history and made unrelated turns inject memories (4 of 6 with N=2 versus 0 of 6 with N=1); raise it for chats whose context lives in the previous reply.
2. Exclude chunks whose message is among the last `R` messages (default 5), count the exclusions.
3. Keep chunks with cosine ≥ threshold (default 0.3), top-K (default 3), merge adjacent chunks of the same message, order chronologically.
4. Render the editable template (default: "Relevant memories of past events in this conversation (they already happened; do not repeat them as new):\n{{memories}}"), cap at the plugin's own token limit (default 600, `api.model.estimateTokens`; lowest-score memories dropped first), also bounded by `getContextBudget`.
5. Return `{ source: "plugin", role: "system", anchor: "at-depth", depth: 2, priority: 10, tokenBudgetBehavior: "counted" }` (priority below the summarizer's 20 so these are trimmed first). Because at-depth `n` lands before the n-th message from the end, only the tail changes and the cached prefix is untouched.

Logs (`api.logger` + an in-memory ring buffer of 200 events shown in the viewer): `match: 3 memories (scores 0.71, 0.58, 0.44) -> 212 tokens`, `no match: <reason>` with reasons `empty index`, `model mismatch`, `all below threshold (best 0.21)`, `all inside recent window`, `timeout`, `embedding error: <actionable text>`; indexing lines with chunks and ms; server module logs download, start (backend, layers), idle stop and exit.

Query embeddings are cached by text hash (last 32) so swipes and regenerations do not re-embed.

### 2.6 UI

- **Settings panel**: wizard first (1: source API/local; 2: profile `default`/`strong` and device auto/GPU/CPU, `default` + `auto` preselected; 3: one button "Install and test" that downloads binary + model, starts, embeds a test sentence, then "Index this chat"). Below it an *Advanced* section with every knob from the brief (model, URL, key, binary/model paths, chunk size and overlap, query messages, top-K, threshold, retain, depth, template, token cap, idle minutes, custom model: HF repo + file or local path + prefix profile). Status block: active model, chunks indexed, index state (up to date / stale / other model), process state, GPU backend and layers offloaded, last error, disk used with delete buttons. **Test** reports vector dimension and latency, with actionable errors (401 → "check the API key", ECONNREFUSED → "endpoint not reachable", HTTP 404 on `/embeddings` → "this server has no embeddings endpoint", dimension mismatch, process failed to start with the last log line). Index all / Re-index with progress bar and Cancel; Purge.
- **Header action → modal** with: last injection (memories, score, source message and position), status, **dry run** (the current composer draft if the snapshot exposes it, otherwise a prefilled text box — to verify), and the log tail. Layout works at 360 px (single column, 44 px touch targets); verified with measurements and screenshots, not assumed.
- The API key lives in `api.settings` (plain local JSON); stated in the panel and README. `connections:secrets` is not used.

### 2.7 Manifest

```json
{ "id": "smiley-embeddings-memory", "version": "0.1.0", "main": "dist/index.js", "serverMain": "dist/server.js",
  "styles": ["dist/styles.css"], "minAppVersion": "0.0.6", "apiVersion": 1, "category": "memory-lore", "enabled": false,
  "permissions": ["state:read","actions","chat:prompt-inject","chat:message-update","ui:settings","ui:header","ui:modals","ui:styles","storage","server:process"] }
```

`actions` is needed for `listChats()`. The description states, in plain words, that the plugin runs a program on the computer and downloads files from GitHub and Hugging Face, and that with an API source chat text is sent to that endpoint.

Packaging: `build` becomes `bun build ./src/index.tsx --outdir ./dist --external preact && bun build ./src/server/main.ts --target bun --format esm --outfile ./dist/server.js`; `pack.ts` already zips all of `dist/` but is extended to fail when `serverMain` is missing or outside `dist/`. Typings are refreshed with `bun run update-types` (the existing copies predate `api.host`). Expected ZIP well under the installer's 25 MB limit.

## 3. Tests (`bun test`)

Chunking (paragraph → line → space, size, overlap, unicode); cosine and top-K with threshold; recent-window exclusion; reconcile (edit, swipe, delete, fork, new message, streaming skipped); base64 float32 round trip and shard split/append; model mismatch; model prefixes and `models.ts` catalog; release asset selection per OS/arch/GPU from fixture lists; download resume and size/sha mismatch with a local HTTP server; zip-slip rejection; manual-path validation; injection template and token cap; injector never throws and respects timeout.

## 4. Open risks

1. `minAppVersion` `0.0.6` is what the docs prescribe, but the branch is `0.0.5`: the plugin reports as incompatible until that release exists. Local testing omits it and I will not commit the omission.
2. `-c`/`-b`/`-np` batching and `--log-file` + `-lv 4` are verified only indirectly; the server-manager task measures them first.
3. Only Windows + NVIDIA was run. Linux, macOS, Android/Termux selection is unit-tested against asset names, never executed. Termux: manual path is the supported route.
4. The CUDA path is a 620 MB download; Vulkan (37 MB) is selectable and measured at 192 vs 261 emb/s on the small model.
5. The injector is awaited on every generation: a slow remote embedding endpoint delays sends up to the timeout.
6. The "recent window" is a message count; if the app trims history below `R` messages, the gap between them and the first retrievable memory is not covered.
7. Orphan indexes after chat deletion persist until the next sweep.
8. A 2560-d index of 20k chunks is 263 MB on disk; fine under the 128 MiB-per-key cap only because of sharding.
9. The server module is not sandboxed; `server:process` plus owner approval is a consent gate, not a boundary.

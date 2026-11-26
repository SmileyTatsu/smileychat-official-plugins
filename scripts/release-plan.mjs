/**
 * Plans a release from the plugin manifests.
 *
 * Releases are tag-driven: `.github/workflows/release.yml` builds every plugin at
 * the tag and attaches one ZIP per plugin. This script exists so nobody has to
 * mint that tag by hand. It compares `plugins/<dir>/plugin.json` against the
 * newest release tag and, when a plugin is new or its version moved forward,
 * returns the next tag plus the release body. Detection is version-based on
 * purpose: a plugin added for the first time releases exactly like a bump, and a
 * commit that changes no version releases nothing.
 *
 * Run by the workflow with `--github-output "$GITHUB_OUTPUT"`; `--json` prints
 * the plan for local inspection. Plain ESM on purpose: the repo has no root
 * package.json, so this must not need a build or installed type definitions.
 */
import { execFileSync } from "node:child_process";
import { appendFileSync, existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));
const pluginRoot = "plugins";

function fail(message) {
    console.error(`release-plan: ${message}`);
    process.exit(1);
}

function git(args) {
    try {
        return execFileSync("git", args, {
            cwd: repoRoot,
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
        }).trim();
    } catch (error) {
        const stderr = error.stderr?.toString().trim();
        fail(`git ${args.join(" ")} failed${stderr ? `: ${stderr}` : ""}`);
    }
}

function parseSemver(value) {
    const match = /^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$/.exec(String(value).trim());
    if (!match) return null;
    return {
        major: Number(match[1]),
        minor: Number(match[2]),
        patch: Number(match[3]),
        prerelease: match[4] ? match[4].split(".") : [],
    };
}

function compareSemver(a, b) {
    for (const key of ["major", "minor", "patch"]) {
        if (a[key] !== b[key]) return a[key] < b[key] ? -1 : 1;
    }
    // A release outranks its own prereleases.
    if (a.prerelease.length === 0 || b.prerelease.length === 0) {
        return a.prerelease.length === b.prerelease.length ? 0 : a.prerelease.length === 0 ? 1 : -1;
    }
    for (let index = 0; index < Math.max(a.prerelease.length, b.prerelease.length); index += 1) {
        const left = a.prerelease[index];
        const right = b.prerelease[index];
        if (left === undefined) return -1;
        if (right === undefined) return 1;
        const leftIsNumber = /^\d+$/.test(left);
        const rightIsNumber = /^\d+$/.test(right);
        if (leftIsNumber && rightIsNumber) {
            const difference = Number(left) - Number(right);
            if (difference !== 0) return difference < 0 ? -1 : 1;
            continue;
        }
        if (leftIsNumber !== rightIsNumber) return leftIsNumber ? -1 : 1;
        if (left !== right) return left < right ? -1 : 1;
    }
    return 0;
}

function parseManifest(path, text) {
    let raw;
    try {
        raw = JSON.parse(text);
    } catch {
        fail(`${path} is not valid JSON`);
    }

    if (typeof raw.id !== "string" || raw.id.trim() === "") fail(`${path} needs a string id`);
    if (typeof raw.version !== "string" || !parseSemver(raw.version)) {
        fail(`${path} needs a semver version, found ${JSON.stringify(raw.version)}`);
    }

    return {
        dir: path.slice(0, path.lastIndexOf("/")),
        id: raw.id,
        name: typeof raw.name === "string" && raw.name.trim() !== "" ? raw.name : raw.id,
        version: raw.version,
    };
}

function assertUniqueIds(manifests, where) {
    const seen = new Map();
    for (const manifest of manifests) {
        const other = seen.get(manifest.id);
        if (other) fail(`plugin id "${manifest.id}" is used by both ${other} and ${manifest.dir} (${where})`);
        seen.set(manifest.id, manifest.dir);
    }
}

function worktreeManifests() {
    const root = join(repoRoot, pluginRoot);
    return readdirSync(root, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => `${pluginRoot}/${entry.name}/plugin.json`)
        .filter((path) => existsSync(join(repoRoot, path)))
        .map((path) => parseManifest(path, readFileSync(join(repoRoot, path), "utf8")));
}

function refManifests(ref) {
    return git(["ls-tree", "-r", "--name-only", ref, "--", `${pluginRoot}/`])
        .split("\n")
        .filter((path) => path.endsWith("/plugin.json"))
        .map((path) => parseManifest(path, git(["show", `${ref}:${path}`])));
}

function releaseTags() {
    return git(["tag", "--list", "v*"])
        .split("\n")
        .filter(Boolean)
        .flatMap((tag) => {
            const semver = parseSemver(tag.slice(1));
            return semver ? [{ tag, semver }] : [];
        })
        .sort((a, b) => compareSemver(b.semver, a.semver));
}

function diffManifests(current, baseline) {
    const previousById = new Map(baseline.map((manifest) => [manifest.id, manifest]));
    const changes = [];

    for (const manifest of current) {
        const previous = previousById.get(manifest.id);
        if (!previous) {
            changes.push({
                id: manifest.id,
                name: manifest.name,
                version: manifest.version,
                previousVersion: null,
                kind: "new",
            });
            continue;
        }

        const version = parseSemver(manifest.version);
        const previousVersion = parseSemver(previous.version);
        if (!version || !previousVersion) continue;

        if (compareSemver(version, previousVersion) > 0) {
            changes.push({
                id: manifest.id,
                name: manifest.name,
                version: manifest.version,
                previousVersion: previous.version,
                kind: "updated",
            });
        } else if (compareSemver(version, previousVersion) < 0) {
            console.error(
                `release-plan: ${manifest.id} is at ${manifest.version}, behind ${previous.version} in ${previous.dir}; not releasing it`,
            );
        }
        // Equal versions are already published: nothing to do.
    }

    return changes.sort((a, b) => a.id.localeCompare(b.id));
}

function nextTag(changes, latest) {
    if (!latest) return "v1.0.0";
    // A plugin appearing for the first time is a feature of the repo; a version
    // bump only replaces artifacts, so it takes a patch.
    const addsPlugin = changes.some((change) => change.kind === "new");
    return addsPlugin
        ? `v${latest.major}.${latest.minor + 1}.0`
        : `v${latest.major}.${latest.minor}.${latest.patch + 1}`;
}

function repositorySlug() {
    const fromEnvironment = process.env.GITHUB_REPOSITORY?.trim();
    if (fromEnvironment) return fromEnvironment;
    try {
        const remote = git(["remote", "get-url", "origin"]);
        const match = /github\.com[:/]([^/]+\/[^/]+?)(?:\.git)?$/.exec(remote);
        return match ? match[1] : null;
    } catch {
        return null;
    }
}

function releaseBody(tag, baseTag, changes, inventory) {
    const lines = [];

    if (changes.length > 0) {
        lines.push(`### Plugin changes${baseTag ? ` since ${baseTag}` : ""}`, "");
        for (const change of changes) {
            lines.push(
                change.kind === "new"
                    ? `- \`${change.id}\` ${change.version} — new plugin`
                    : `- \`${change.id}\` ${change.version} — was ${change.previousVersion}`,
            );
        }
        lines.push("");
    } else {
        lines.push(`No plugin version changes${baseTag ? ` since ${baseTag}` : ""}; the artifacts were rebuilt.`, "");
    }

    lines.push(`### Artifacts in ${tag}`, "", "| Plugin | Version |", "| --- | --- |");
    for (const manifest of [...inventory].sort((a, b) => a.id.localeCompare(b.id))) {
        lines.push(`| ${manifest.id} | ${manifest.version} |`);
    }
    lines.push("");

    const slug = repositorySlug();
    if (slug && baseTag) {
        lines.push(`**Compare:** https://github.com/${slug}/compare/${baseTag}...${tag}`);
    }

    return lines.join("\n").replace(/\n+$/, "");
}

function writeGithubOutput(file, plan) {
    const values = {
        tag: plan.tag,
        needs_release: plan.needsRelease ? "true" : "false",
        ref: plan.ref,
        commit: plan.commit,
        body: plan.body,
    };

    let payload = "";
    for (const [key, value] of Object.entries(values)) {
        payload += value.includes("\n")
            ? `${key}<<RELEASE_PLAN_EOF\n${value}\nRELEASE_PLAN_EOF\n`
            : `${key}=${value}\n`;
    }
    appendFileSync(file, payload);
}

function parseArgs(argv) {
    const args = { tag: null, base: null, force: false, output: null, json: false };
    for (let index = 0; index < argv.length; index += 1) {
        const argument = argv[index];
        switch (argument) {
            case "--tag":
                args.tag = argv[++index] ?? fail("--tag needs a value");
                break;
            case "--base":
                args.base = argv[++index] ?? fail("--base needs a value");
                break;
            case "--github-output":
                args.output = argv[++index] ?? fail("--github-output needs a file");
                break;
            case "--force":
                args.force = true;
                break;
            case "--json":
                args.json = true;
                break;
            default:
                fail(`unknown argument ${argument}`);
        }
    }
    return args;
}

function buildPlan(args) {
    const tags = releaseTags();

    if (args.tag) {
        const tag = args.tag;
        if (!parseSemver(tag.replace(/^v/, ""))) fail(`"${tag}" is not a v<semver> tag`);
        if (!git(["rev-parse", "--verify", `refs/tags/${tag}`])) fail(`tag ${tag} does not exist`);

        // A tag push releases exactly that tag; compare against the newest other
        // tag only to describe the changes in the release body.
        const base = args.base ?? tags.find((candidate) => candidate.tag !== tag)?.tag ?? null;
        const inventory = refManifests(tag);
        assertUniqueIds(inventory, tag);
        const changes = diffManifests(inventory, base ? refManifests(base) : []);

        return {
            tag,
            needsRelease: true,
            ref: tag,
            commit: git(["rev-parse", `${tag}^{commit}`]),
            baseTag: base,
            changes,
            inventory: inventory
                .map(({ id, version }) => ({ id, version }))
                .sort((a, b) => a.id.localeCompare(b.id)),
            body: releaseBody(tag, base, changes, inventory),
        };
    }

    const latest = tags[0] ?? null;
    const base = args.base ?? latest?.tag ?? null;
    const inventory = worktreeManifests();
    assertUniqueIds(inventory, "working tree");

    const changes = diffManifests(inventory, base ? refManifests(base) : []);
    const needsRelease = changes.length > 0 || args.force;
    const tag = needsRelease ? nextTag(changes, latest?.semver ?? null) : "";

    return {
        tag,
        needsRelease,
        ref: process.env.GITHUB_SHA?.trim() || "HEAD",
        commit: git(["rev-parse", "HEAD"]),
        baseTag: base,
        changes,
        inventory: inventory
            .map(({ id, version }) => ({ id, version }))
            .sort((a, b) => a.id.localeCompare(b.id)),
        body: needsRelease ? releaseBody(tag, base, changes, inventory) : "",
    };
}

const args = parseArgs(process.argv.slice(2));
const plan = buildPlan(args);

if (args.output) writeGithubOutput(args.output, plan);

if (args.json) {
    console.log(JSON.stringify(plan, null, 4));
    process.exit(0);
}

if (!plan.needsRelease) {
    console.log(`No plugin version changes since ${plan.baseTag ?? "the start of the repo"}; no release needed.`);
} else {
    console.log(`Planned ${plan.tag} (${plan.commit.slice(0, 7)}) with ${plan.changes.length} change(s):`);
    for (const change of plan.changes) {
        console.log(
            change.kind === "new"
                ? `  + ${change.id} ${change.version} (new)`
                : `  ~ ${change.id} ${change.version} (was ${change.previousVersion})`,
        );
    }
    if (plan.changes.length === 0) console.log("  (forced: rebuilding the current artifacts)");
}

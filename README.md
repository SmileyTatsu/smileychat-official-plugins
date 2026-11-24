# SmileyChat Official Plugins

Verified SmileyChat plugins, built as prebuilt ZIP artifacts and listed in the
[`smileychat-plugins`](https://github.com/SmileyTatsu/smileychat-plugins) registry
as `status: official`.

SmileyChat installs from a ZIP: users never need Git, dependency installation or
a build step. This repo exists so the plugins are developed, typechecked and
released as artifacts instead of shipping inside the app bundle.

## Layout

```text
plugins/<plugin-id>/
    plugin.json       SmileyChat manifest (id, version, permissions, apiVersion)
    src/              plugin sources, built with bun
    types/            generated SmileyChat plugin API typings
    scripts/          pack + update-types helpers
    release/          built ZIP (git-ignored)
```

## Working on a plugin

```sh
cd plugins/workspace-tools
bun install
bun run typecheck        # against the generated typings
bun run update-types     # refresh types/ from SmileyChat main
bun run build            # dist/index.js, keeps preact external
bun run pack             # release/<id>-<version>.zip
```

For local testing, copy `plugin.json` and `dist/` into
`SmileyChat/userData/plugins/<plugin-id>/` and reload the app. `bun run dev`
serves plugin files from the Bun server, so no rebuild of the app is needed.

## Releasing

Pushing a tag (`v1.0.0`) runs `.github/workflows/release.yml`, which builds every
plugin, packs one ZIP per plugin and attaches them to the release. The registry
entry then points at
`https://github.com/SmileyTatsu/smileychat-official-plugins/releases/download/<tag>/<id>-<version>.zip`.

Bump the version in both `plugin.json` and `src/manifest.ts`: the manifest is
what the app reads at runtime, `plugin.json` is what the installer validates.

## Rules

- Import the plugin API from `types/smileychat-plugin-api.d.ts` only. App
  internals (`#frontend/...`) do not exist in a downloaded artifact.
- Keep `preact` external. The app serves an import map for `preact`,
  `preact/hooks` and `preact/jsx-runtime`; bundling a second Preact copy breaks
  hooks inside host-rendered panels.
- Declare only the permissions the plugin uses; `connections:secrets` also needs
  a per-plugin approval in Options.

# rulesForMojo

Cursor agent rules adapted for a Mojo project (from Room21 `.cursor/rules/`).

## Install into a Mojo repo

Copy all `*.mdc` files into the target project's `.cursor/rules/`:

```bash
mkdir -p /path/to/mojo-project/.cursor/rules
cp rulesForMojo/*.mdc /path/to/mojo-project/.cursor/rules/
```

Or symlink if you maintain rules centrally:

```bash
ln -s /path/to/rulesForMojo /path/to/mojo-project/.cursor/rules-mojo
# then copy or symlink individual .mdc into .cursor/rules/
```

## Customize after copy

1. **no-commit-without-green-test.mdc** — set your test gate command if not `make test`.
2. **config-and-env-contracts.mdc** — list your real config files (`pixi.toml`, etc.).
3. **mojo-module-boundaries.mdc** — rename layers to match your directory layout.
4. **generated-artifacts-reading.mdc** — add project-specific generated paths.

## Files (18 rules)

| File | Purpose |
|------|---------|
| pre-production-discipline | One contract, no fallback, 0 warnings |
| decompose-complex-variable-computation | Extract complex assignments |
| silent-exit-must-log | Fail silently → log |
| no-hardcoded-magic | No magic constants |
| typed-arguments-domain-types | Domain types in signatures |
| domain-optional-and-wire-strings | Optional vs sum types; wire strings |
| public-identifiers-boundary | Opaque ids outward |
| no-linter-changes-without-permission | Ask before linter edits |
| no-build-workarounds-without-ask | No arch workarounds for green build |
| changes-and-proposals | Protected contract edits need yes |
| config-and-env-contracts | Env/config as API |
| generated-artifacts-reading | Don't read huge generated blobs |
| cursor-skills-in-repo | Skills under `.cursor/skills/` |
| environment-git-and-tests | Git + test discipline |
| no-commit-without-green-test | Hard gate before commit |
| mojo-build-and-check | `mojo build` / `mojo test` flow |
| mojo-module-boundaries | Layer import rules |
| mojo-ffi-safety | Python/C FFI boundary |
| mojo-code-documentation | Doc comments on public API |

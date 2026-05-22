# burn-the-stock

Taiwan stock trading bot with a burn NN backend.

Workspace: `stock-client` (lib), `trainer` (bin), `trader` (bin).

## Project Conventions

Define all URLs as `pub const` in `crates/stock-client/src/urls.rs`.

## Cargo

Use `cargo add` to add dependencies.

Use `[workspace.dependencies]` for shared dependencies with minimal features and use `.workspace = true` and features in members.

All subcommands take `--workspace --all-targets` if possible.

```bash
cargo check --workspace --all-targets
cargo build  --workspace --all-targets [--release]
cargo test   --workspace --all-targets
```

Before finishing any task, run clippy and fmt.

```bash
cargo clippy --workspace --all-targets
cargo fmt    --all
```

## Exports

Never re-export a module and its members from the same level. Pick one or the other. If flat access is needed, use a `prelude` module.

## Variables and Naming

Spell out variable names fully. Avoid common abbreviations and use complete, self-describing names.

Single-letter loop counters are fine (`i`, `j`, `k`), but prefer meaningful names when the loop is long or complex.

## Writing

Favor explicit conjunctions and transition words to connect your thoughts. Use natural phrasing instead of colons or semicolons, and limit punctuation to standard ASCII characters.

Comments explain why and should be concise and minimal.

Commits use conventional commit format. Subject under 50 chars (hard limit 72).

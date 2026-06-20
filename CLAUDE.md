# burn-the-stock

Taiwan stock trading bot with a burn NN backend. See `pyproject.toml` for Python project config, dependencies, and layout settings.

Workspace: `stock-client` (lib), `trainer` (bin), `trader` (bin).

The course evaluation platform is sim stock (https://ciot.imis.ncku.edu.tw/sim_stock).

## Project Conventions

Define all URLs as `pub const` in `crates/stock-client/src/urls.rs`.

## Cargo

Use `cargo add` to add dependencies.

Use `[workspace.dependencies]` for shared dependencies with minimal features and use `.workspace = true` and features in members.

Always scope by `--workspace`, never `--package`/`-p`. A package filter resolves a narrower feature set and forces recompilation, while `--workspace` keeps the unified feature union warm across runs. Use `--all-targets` where it applies.

```bash
cargo check  --workspace --all-targets
cargo build  --workspace --all-targets [--release]
cargo test   --workspace --all-targets
```

To run a binary or example, select it by `--bin`/`--example` (still no `-p`) so it builds under the same workspace feature union.

```bash
cargo run --bin tickers -- <args>
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

Comments target the surprising. The reader follows ordinary code logic, so a comment earns its place on a genuinely counterintuitive choice and states its why simply and concisely. The default is no comment; when in doubt, leave it out. Keep doc comments and any clippy-required `# Errors` or `# Panics` section in the same simple, concise style.

Commits use conventional commit format. Subject under 50 chars (hard limit 72).

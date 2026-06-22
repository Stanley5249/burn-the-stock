# burn-the-stock

Taiwan stock trading bot with a Burn neural network backend. The trading platform is sim_stock (https://ciot.imis.ncku.edu.tw/sim_stock).

## Cargo

Always use `cargo add` to add dependencies.

Use `[workspace.dependencies]` for shared dependencies with minimal features and use `.workspace = true` and features in members.

Always scope commands by `--workspace`, never `--package`/`-p`. A package filter resolves a narrower feature set and forces recompilation, while `--workspace` keeps the unified feature set active across runs. Use `--all-targets` where it applies.

```bash
cargo check --workspace --all-targets
cargo build --workspace --all-targets [--release]
cargo test --workspace --all-targets
```

To run a binary or example, use `--bin`/`--example` (not `-p`) to build under the same workspace feature set.

```bash
cargo run --bin tickers [--release] -- <args>
```

Before finishing any task, run clippy and fmt.

```bash
cargo clippy --workspace --all-targets
cargo fmt --all
```

## Exports

Never re-export a module and its members from the same level. Pick one or the other. If flat access is needed, use a `prelude` module.

## Variables and Naming

Use complete, self-describing names and avoid abbreviations.

Single-letter names are fine (`i`, `j`, `k`) in local scope, but prefer meaningful names for long or complex context.

## Writing

Concise style, avoid explanations. Favor explicit conjunctions and transition words to connect your thoughts. Use natural phrasing instead of colons or semicolons, and limit punctuation to standard ASCII characters.

## Comments

Add comments only for non-obvious reasons: a workaround, a performance trade-off, a hidden constraint, or why something looks wrong but isn't. Keep them to one line.

## Commits

Conventional commit format. Subject under 50 chars (hard limit 72).

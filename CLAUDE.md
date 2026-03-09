# PocketPostdoc — Claude Code Instructions

## Environment

You are running inside a sandboxed Ubuntu 24.04 dev container.
Available tools: python3, python-pptx, Pillow, pdflatex, latexmk, ghostscript, make, git.

## Examples

This repo includes example workflows in `examples/`. Each has its own `CLAUDE.md`:

### 1. PowerPoint Editor (`examples/pptx-editor/`)
Edit `.pptx` presentations with LaTeX equation rendering.
- User files go in `work/presentations/`
- See `examples/pptx-editor/CLAUDE.md` for the full workflow and style preferences

### 2. LaTeX Double-Checker (`examples/latex-assistant/`)
Review physics/math papers for errors — only flags mistakes it is certain about.
- User files go in `work/papers/`
- Say **"run the double checker"** to analyze all `.tex` files in `work/papers/`
- See `examples/latex-assistant/CLAUDE.md` for the protocol

## LaTeX Build

- Build PDF: `make -C setup pdf`
- Watch mode: `make -C setup watch`
- Clean: `make -C setup clean`

## Git Workflow

- `work/` is gitignored by default except for `.gitkeep` scaffolds
- To version-control a file: `git add -f work/path/to/FILE && git commit -m "message"`
- Use descriptive commit messages
- Commit after each logical set of changes, not after every micro-edit

## Agent Orchestration

**You (Claude main) are a thin orchestrator.** Delegate all heavy work to subagents.
Full setup guide: `scripts/ORCHESTRATION_SETUP.md`

### Golden Rule: Protect Your Context
- **NEVER** read large files or many files directly — delegate to Codex or a Claude subagent
- **NEVER** do bulk code analysis inline — spawn a subagent that writes results to `/tmp/`
- **ALWAYS** receive results as short summaries, then synthesize for the user
- At session end, update memory files so the next session can pick up where you left off

### Codex CLI
- **Wrapper:** `scripts/codex_worker.sh` (handles nvm sourcing + non-interactive flags)
- **Invocation:** `bash scripts/codex_worker.sh [--out /tmp/file.md] "prompt"`
- **Background:** append `&` and `wait` for parallel tasks
- **Config:** `~/.codex/config.toml` (model=gpt-5.3-codex, effort=xhigh)

### Delegation Table
| Task | Delegate to |
|---|---|
| Read/summarize many files | Codex |
| Mechanical edits, refactors | Codex |
| Deep reasoning, planning | Claude subagent |
| Code review (high confidence) | Both → cross-review pattern |
| Quick search (Grep/Glob) | Inline (no subagent overhead) |

### Cross-Review Pattern
1. Launch Claude subagent + Codex in parallel (same response)
2. Each writes to `/tmp/claude_r1.md` and `/tmp/codex_r1.md`
3. Each reviews the other → APPROVE or REQUEST_CHANGES
4. Iterate until both APPROVE (cap at 3 rounds; main Claude breaks ties)
5. Main Claude synthesizes final answer

### Working Memory (MANDATORY)

Your persistent memory lives in **`scripts/working_memory.md`** (in the repo, user-editable).

**Before every user response:** Read `scripts/working_memory.md`. Every time. No exceptions.
This is your only defense against context compression — if the system silently drops earlier
messages, this file is how you recover. Treat it like loading save state.

**After every completed task:** Append a 1-line summary under "## Activity Log" in
`scripts/working_memory.md`. Format: `- YYYY-MM-DD: <what was done>`. This is non-optional.

**When you learn something durable** (user preference, project convention, gotcha):
Add it to the appropriate section in `scripts/working_memory.md`.

**Pruning:** The user may delete lines from this file at any time. Treat the file as
the sole source of truth — if something was removed, forget it.

You have no visibility into your own context size. You cannot trigger compression or
delete context. Checkpointing after every task is the only defense against context loss.

## Constraints

- You are in a container. Do not attempt to install additional system packages.
- Only modify files under the repository root.
- Do not modify files under `.devcontainer/` or `setup/` unless explicitly asked.
- Temporary files should go in `/tmp/`.

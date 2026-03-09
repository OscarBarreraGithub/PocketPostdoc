# IAIFI Paperscape — Orchestration State

**Purpose:** Resume orchestration after context compaction. Read this file if you lose context.

## Master Plan
- Full plan is in `PLAN.md` (1450 lines, v2.1 Final)
- Implements an interactive 2D scatterplot of ~9K physics/ML papers (547 IAIFI + ~8K background)
- Pipeline: arXiv harvest → SPECTER2 embed → kNN select → PCA+UMAP → HDBSCAN cluster → export JSON → static frontend

## Orchestration Steps

### Step 0: Save state ✅
This file exists.

### Step 1: Create atomic task list from PLAN.md
- Launch Claude subagent to draft detailed task list
- Have Codex review/confirm the task list
- Save final task list to `/tmp/iaifi_tasklist.md`
- STATUS: IN PROGRESS

### Step 2: Implement all tasks
- Delegate tasks to Claude subagents and Codex workers
- Codex does more writing (larger token budget, 5-hour window)
- Iterate through tasks until all complete
- STATUS: NOT STARTED

### Step 3: Cross-review (concurrent with Step 2)
- Claude-written code reviewed by Codex
- Codex-written code reviewed by Claude
- STATUS: NOT STARTED

### Step 4: Final sign-off
- Both Claude and Codex confirm codebase matches PLAN.md
- STATUS: NOT STARTED

### Step 5: Write README
- High-level logic + how to view final product
- STATUS: NOT STARTED

## Key Paths
- Plan: `/workspaces/PocketPostdoc/Code/IAIFI/PLAN.md`
- Source: `/workspaces/PocketPostdoc/Code/IAIFI/src/iaifi_paperscape/`
- Web: `/workspaces/PocketPostdoc/Code/IAIFI/web/`
- Configs: `/workspaces/PocketPostdoc/Code/IAIFI/configs/`
- Data: `/workspaces/PocketPostdoc/Code/IAIFI/data/`
- Task list: `/tmp/iaifi_tasklist.md`

## Pipeline Order (dependency-aware)
A (IAIFI metadata) → B (background candidates) → D (embed all 30K) → C (kNN select ~8-10K) → E (PCA+UMAP) → F (HDBSCAN+labels+enrichment) → G (export JSON+neighbors) → H (frontend)

## Delegation Strategy
- Codex: More writing tasks (larger budget), mechanical code, file creation
- Claude subagents: Reasoning-heavy tasks, architecture decisions, reviews
- Main Claude (me): Orchestrator only — never read large files, synthesize short summaries

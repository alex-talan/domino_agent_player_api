# GenAI Domino Player — Development Instructions

## Purpose and source of truth

Implement one GenAI-powered domino player as a Python FastAPI application. Read @docs/specs/domino-genai-player.md before implementation and before changing gameplay or API behavior. Runtime instructions are in @prompts/domino-system.md and @prompts/domino-player.md.

The specification defines what to build; this file defines how to work. Follow applicable repository `AGENTS.md` and other development instructions as well. Surface genuine contradictions; do not silently change requirements.

## Required layout

- `ai-core/`: supplied source/starter kit to reuse for model integration.
- `src/`: this application's FastAPI, use cases, domain rules, and AI-Core adapter.
- `prompts/`: version-controlled runtime prompt files.
- `docs/specs/`: functional specification.
- `tests/`: meaningful automated tests.
- `CLAUDE.MD`: these coding instructions.

This file's requested spelling is `CLAUDE.MD`. When configuring Claude Code, ensure its recognized project instruction entry point loads this file. On case-sensitive filesystems do not assume uppercase `.MD` and `CLAUDE.md` are interchangeable. If needed, create a small `CLAUDE.md` entry point referencing this file using the tool's supported import mechanism; keep the instructions in one place.

## Inspect before coding

1. Inspect `ai-core/` documentation, manifests, supported Python version, configuration conventions, adapters, asynchronous call patterns, response parsing, timeouts, retries and examples.
2. Identify the actual supported integration path. Do not invent AI-Core imports or APIs. Reuse the starter kit rather than constructing a parallel provider abstraction unnecessarily.
3. If `ai-core/` is absent or required capabilities are missing, identify the precise blocker. Continue independent domain/API work with a mock adapter where useful, but do not claim live integration is complete. Do not download an unrelated kit.
4. Make a concise implementation plan and implement incrementally. Use existing dependency and testing conventions where available.

## Architecture

Keep tile definitions, legal-move enumeration, response legality and fallback selection independent of FastAPI and providers. Use typed request/response models and a narrow LLM decision interface backed by AI-Core. Keep routes thin. Prefer asynchronous model calls when supported; do not block the event loop with synchronous network I/O.

Do not modify generic `ai-core/` source unless a real integration issue requires it (ask for confirmation in that case); isolate and explain such changes. Put domino behavior in `src/`.

## Non-negotiable behavior

- Expose `POST /play` and `POST /end` with the exact Master payload fields.
- Use the current `your_tiles` as the complete hand. Selection is stateless across turns; there is no `/tiles` initialization.
- Call the LLM for each valid turn, including a turn whose only permitted response is pass, unless a runtime failure prevents the call.
- The LLM chooses strategy; do not substitute the highest-points rule on the normal path.
- Compute all legal moves independently, then validate the model output against them. Schema-valid JSON alone is not sufficient.
- Never forward an illegal move or pass when a legal tile exists. Use the documented fallback when the model fails.
- Preserve the exact tile-index catalogue. Tile 0 is not false/no-move; -1 is the pass sentinel.
- Do not infer tile orientation from identifier order. Use supplied ends.
- Keep public-history deductions uncertain; do not invent hidden hands or missing move chronology.
- `/end` records the supplied result without a model call and does not block later valid turns.

## Prompt and model integration

Load the two prompt files from a stable configured root. Serialize validated game data as JSON and replace only the three documented literal tokens in the player template. Verify no tokens remain. Never use repository instructions as the model's gameplay prompt.

Use system and user message roles supported by the selected AI-Core adapter. Prefer its supported structured-output features. Strictly parse and validate the two-field response, rejecting extra keys and prose. Never eval model output. Do not request, expose or persist chain-of-thought; only the final move is needed.

Use the actual AI-Core configuration mechanisms for provider/model/credentials. Do not hard-code a provider or model name or introduce direct provider SDK calls when the kit already supports the integration. Keep credentials in environment/secrets and provide a sanitized configuration example without actual keys.

Enforce one total request deadline with margin below the Master timeout. Count adapter retries and optional repair within it. Permit at most one output repair. Ensure cancellation and fallback do not wait for unbounded provider work. Log decision source so fallback never masquerades as an LLM move.

## Verification

Use the tooling found in the repository; document exact installation, run and test commands after confirming them. Do not invent commands in a completion report.

Test domain legality, context rendering, JSON validation, API behavior, null histories, tile 0, doubles, both ends, no-move turns, timeout budgets, repair limits, and fallback. Mock AI-Core in normal tests. Prove that a legal lower-points model choice is honored. Live calls must be opt-in and must not be required by CI.

Create a short README explaining setup, the actual AI-Core integration, required environment variables, launch command/address, prompt customization, an example Master request, fallback behavior and checks run. Do not claim strategic superiority from a smoke test or guarantee reproducible LLM choices.

## Working style

Proceed with routine reversible implementation choices. Ask only for material missing information that cannot be established from the repository. Report what changed, checks run, and any concrete blocker or remaining limitation. Never describe a mocked or untested provider integration as production-ready.

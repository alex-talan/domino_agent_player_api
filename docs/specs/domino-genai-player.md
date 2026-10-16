# GenAI Domino Player — Functional Specification

## Objective

Build a single Python FastAPI Player API. On each valid turn, construct a context from the Master's request, send the system and player prompts to an LLM through the supplied AI-Core starter kit, and return the LLM's selected legal move.

The objective is to win, individually, by playing the last tile first or minimizing remaining points in a blocked game. The LLM may use the public history to choose among legal moves.

## Repository layout

```text
ai-core/                         Existing starter kit/source; inspect before implementation
src/                             FastAPI application and player logic; use the right skill
prompts/
    domino-system.md             System instructions
    domino-player.md             Per-turn template
docs/specs/
    domino-genai-player.md       This specification
tests/                           Tests for rules, API and LLM integration boundaries
CLAUDE.MD                        Coding instructions
```

AI-Core is an implementation dependency, not a runtime prompt. Reuse its existing model configuration, provider adapters and structured-output facilities where available. Its source is not supplied with this document; do not assume package names, imports, interfaces, Python versions, or provider support. Determine these by inspecting `ai-core/`. Keep game-specific logic in `src/`. 

## Master integration

The Master has four configured Player API addresses and owns all game state. An external caller triggers the Master's `POST /start`. The Master distributes seven of the 28 tiles to each player in memory, then calls players sequentially, starting with `p0`.

Implement only the required player callbacks: `POST /play` and `POST /end`. API fields must retain the exact names below.

### POST /play

```json
{
  "table": [22, 4, 2],
  "head": 4,
  "tail": 2,
  "p0": [22],
  "p1": [4],
  "p2": [-1],
  "p3": [2],
  "to_play": "p0",
  "your_tiles": [3, 14, 0, 7, 19, 11]
}
```

| Field | Meaning |
|---|---|
| `table` | Tile identifiers in spatial table order, not necessarily chronological order. |
| `head`, `tail` | Available end values; both null when the table is empty. |
| `p0`–`p3` | Per-player chronological move histories. -1 means pass; null means disqualified. Earlier history may be lost when a player becomes null. |
| `to_play` | Identity of the player called, one of `p0`–`p3`. |
| `your_tiles` | Authoritative remaining hand for this turn. Initially seven; reduced by the Master after accepted plays. |

Return HTTP 200 with exactly these two fields:

```json
{"tile": 19, "position": "head"}
```

When no move exists, return:

```json
{"tile": -1, "position": ""}
```

A pass is forbidden if any legal move exists. The Master disqualifies cheating players and players whose API times out. The Master owns the final legality decision and table orientation.

### POST /end

Accept `{"win": true}` or `{"win": false}`; record/log the supplied outcome and return HTTP 200 with no body. A true value may represent a shared win in a blocked game. Repeated notifications are harmless. No model call is necessary for this endpoint.

The Master declares the first player who empties their hand the sole winner. In a blocked game, the eligible player(s) with the smallest sum of remaining pips win. Disqualified players' tiles leave play. The Master specification's four-consecutive-pass rule needs clarification when fewer than four players remain active; the Player must not implement end detection itself.

## Tile catalogue

All parties use zero-based identifiers in exactly this order:

```python
TILES = [
    (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
    (1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
    (2,2), (2,3), (2,4), (2,5), (2,6),
    (3,3), (3,4), (3,5), (3,6),
    (4,4), (4,5), (4,6), 
    (5,5), (5,6), 
    (6,6)
]
```

Tile points equal its two numbers' sum. Identifier 0 is a genuine tile; -1 is only a pass. Doubles have no special rule or bonus. Tiles can be flipped. Do not draw, invent partnership rules, or assume that the first move must be `(6,6)`.

## Request validation and legal moves

Validate before building any prompt or invoking the LLM:

- All documented fields are required; reject invalid input with HTTP 400 using a consistent error body. Configure FastAPI validation handlers accordingly.
- `table` and `your_tiles` contain distinct integer identifiers from 0 to 27, with no overlap. Booleans and numeric strings are not integer identifiers.
- `your_tiles` contains one to seven tiles. An empty hand is an invalid turn request: the Master should already have ended the game.
- `head` and `tail` are both null for an empty table, or both strict integers from 0 to 6 for a nonempty table.
- Each history is null or an array of strict integer entries from -1 to 27. `to_play` is one of the four identities and its history is not null.
- Reject unexpected request properties; do not forward arbitrary client text into prompts. Histories are context, not an alternative source for reconstructing the hand.

Compute legal moves in Python independently of the LLM. On an empty table every hand tile is legal; represent each once with position `head`. Otherwise include a `(tile, head)` move if the tile contains `head`, and a `(tile, tail)` move if it contains `tail`. Include both when both apply. The only permitted move when this list is empty is `(-1, "")`.

Use the supplied ends; a list of un-oriented identifiers alone does not specify which end of each tile faces outward.

## Per-turn agent workflow

1. Validate the Master request and compute legal moves.
2. Build a fresh context: rules and full catalogue in the system prompt; raw validated state, decoded hand and points, decoded table, legal moves and their resulting ends in the player prompt.
3. Render `prompts/domino-player.md` by replacing its three literal placeholder tokens with serialized JSON. Do not use unrestricted formatting over the JSON examples in either prompt. Do not insert credentials or repository contents.
4. Send `domino-system.md` as a system message and the rendered player prompt as a user message through the AI-Core integration. No conversation history carries over between turns. A valid no-move turn still goes to the LLM, with only the pass available, to meet the per-turn model-decision requirement.
5. Request structured JSON matching the response schema. Where the selected provider lacks schema-constrained output, parse its JSON and perform the same strict local validation. Never execute generated code or use eval.
6. Validate the model response syntactically and against the locally computed move list. A response with extra keys, an unowned tile, wrong end, unjustified pass, or non-JSON prose is invalid. Do not silently extract a different move from explanatory text.
7. Return the accepted move. Do not return model reasoning or diagnostics to the Master.

This is a single bounded decision agent. It needs no web access, tools, multi-agent orchestration, or autonomous loop. Strategic analysis can be internal; only the final JSON is requested and retained as model output.

### Context structure

Replace `{{GAME_STATE_JSON}}` with exactly the validated request object. Replace `{{DECODED_CONTEXT_JSON}}` with an object containing `hand` entries (`tile`, `values`, `points`) in `your_tiles` order and `table` entries (`tile`, `values`) in table order. The values are catalogue values, not oriented tile ends.

Replace `{{LEGAL_MOVES_JSON}}` with an array of objects containing `tile`, `position`, `points`, `resulting_head`, and `resulting_tail`. For a first move use its catalogue pair as the resulting ends and return `head`. For a matching end, replace that end with the tile's other number. A pass has zero points and unchanged ends. These annotations must be produced by Python, not inferred from model output.

The model may reason about possible opponents' holdings from visible plays and passes, but those are uncertain inferences. Per-player histories fully order moves across players, so they always reveal the exact ends at an earlier pass. Never fabricate a historical state or hidden hand.

## Output schema

```json
{
  "type": "object",
  "properties": {
    "tile": {"type": "integer", "minimum": -1, "maximum": 27},
    "position": {"type": "string", "enum": ["head", "tail", ""]}
  },
  "required": ["tile", "position"],
  "additionalProperties": false
}
```

Schema compliance alone is insufficient: always enforce legal-move membership. A tile -1 requires an empty position and no available tile; a nonnegative tile requires a legal end.

## Deadlines and failure behavior

Use a configured total turn budget shorter than the Master's HTTP timeout, with response/network margin. Require deployment-specific values; the Master specification supplies no numerical timeout. Model invocation, adapter retries, optional repair, and parsing all share that budget. Disable or bound hidden provider retries. Cancel remaining work when the client disconnects or the deadline expires, as supported by the stack.

Allow at most one repair call for malformed or illegal model output, only if budget remains. Reuse the same prompts with a short application-generated correction stating the violated rule and asking for a legal JSON response. Do not echo arbitrary invalid model content into the correction.

If the provider fails, times out, returns invalid output after repair, or there is insufficient remaining budget, return a deterministic legal fallback before the Master deadline. Choose the highest-points legal tile, breaking ties by lowest identifier, then `head` before `tail`. If none is legal, pass. This fallback is an operational exception, not the agent's default strategy. Log its source explicitly. A valid incoming request must not be answered with an illegal pass because the LLM failed.

Invalid startup configuration fails startup. Do not silently run with missing prompts, unavailable AI-Core dependencies, or missing credentials. Runtime provider failures use the fallback policy above. Leave the model/provider configurable through the actual AI-Core facilities discovered during implementation.

## State, observability, and configuration

Move selection is stateless across turns and games. The same instance may serve different identities because `to_play` supplies the identity. Do not reject turns because a previous `/end` occurred; there is no game/session identifier for such correlation. No database is required. Identical requests can yield different LLM moves, but must always yield legal moves.

Use environment configuration for provider/model, credentials, total deadline, model-generation settings where supported, and bind address/port. Document actual names after inspecting AI-Core. Keep credentials out of Git, prompts and logs. Load prompts relative to a configured project/package root, not the current working directory.

Log a locally generated request correlation ID, current player, selected move, decision source (`llm` or `fallback`), elapsed time, retry count and sanitized failure category; token usage when available. Do not claim a shared game ID, log secrets, or request/store private chain-of-thought. Default model settings should favor stable structured responses where supported, without promising determinism.

## Acceptance criteria

- API payloads and result callbacks match the Master contract exactly, including null histories and passes.
- First-turn and later-turn context rendering includes the current hand, identity and correct legal moves; no unresolved placeholders remain.
- Model calls use the supplied prompts and actual AI-Core adapter; API tests mock the adapter and do not spend real tokens by default.
- A legal LLM choice is honored even if it is not the highest-points choice.
- Both-end moves, doubles, tile 0, sole legal move, and no-move turns validate correctly.
- Repeated requests do not mutate a cached hand; subsequent requests use their own `your_tiles`.
- Malformed requests never invoke the LLM.
- Prose, extra keys, fabricated identifiers, illegal ends and unjustified passes trigger bounded repair/fallback.
- Timeout and provider-failure tests demonstrate fallback within the configured total budget; no unbounded adapter retries occur.
- End notifications are idempotent and require no model call. A later valid turn is accepted.
- Prompt file loading works from another working directory. Startup detects missing configuration.
- Optional live smoke tests are explicitly enabled and separate from ordinary automated tests. Evaluate live outputs for legality, latency and fallback rate; winning is a measured outcome, not a guaranteed acceptance condition.

## Notes on Master examples

The fields are `p0`–`p3`, not `p4`. The subsequent-round example uses histories `[22]` and `[4]`, consistent with earlier moves. These fix example typos without changing game rules.

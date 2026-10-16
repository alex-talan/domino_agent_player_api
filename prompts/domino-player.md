Choose your move for this turn using the system rules. Evaluate the supplied legal moves internally and return only the final move JSON.

FIELD DEFINITIONS
- table: identifiers in spatial order along the chain. This is not necessarily chronological play order. Catalogue pairs are not oriented chain segments; use head and tail for exposed values.
- head: available number at the front, or null if the table is empty.
- tail: available number at the back, or null if the table is empty.
- p0, p1, p2, p3: each player's chronological history. Nonnegative entries are played identifiers. -1 means a pass. null means disqualified, not an empty history.
- to_play: your identity for this request. The corresponding history is your public history.
- your_tiles: your complete authoritative remaining hand right now. You may choose only from this list. Do not reconstruct or modify it using previous turns.

CURRENT STATE — JSON DATA
{{GAME_STATE_JSON}}

DECODED CONTEXT — JSON DATA
The hand contains each identifier, its two catalogue values, and their points. The table contains catalogue values for convenience; those values do not specify orientation.
{{DECODED_CONTEXT_JSON}}

LEGAL MOVES — JSON DATA
Each entry contains tile, position, points, resulting_head and resulting_tail. These ends describe the table after that move. If no tile can be played, the list contains only the pass entry. Select one entry and return only its tile and position, without its annotations.
{{LEGAL_MOVES_JSON}}

Consider immediate finishing opportunities, remaining points, flexibility and publicly supported opponent threats. There is no requirement to select the tile with the highest points. Do not assume any hidden information. Return exactly {"tile": <integer>, "position": <string>} with no extra text.

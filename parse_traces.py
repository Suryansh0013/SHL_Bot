"""
Parses the SHL-provided sample conversation traces (markdown format) into:
  - the sequence of user utterances (in order), used to script a replay
  - the final expected shortlist (name -> url), taken from the last
    markdown table in the file (the one accompanying end_of_conversation=true)

These traces are pre-recorded *reference* conversations (a specific user
script + a specific reference agent's replies), not just "persona + facts"
like the real evaluator's harness. Replaying the exact same user utterances
against our agent is therefore an approximation of the real evaluator (which
uses an LLM to play the user dynamically, reacting to whatever our agent
asks) -- but it's a strong, fast, deterministic local signal to iterate
against before relying on the real thing.
"""

import re
from pathlib import Path
from typing import List, Dict, Any


def parse_trace(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    turns = re.split(r"###\s*Turn\s*\d+", text)[1:]  # drop preamble before first turn

    user_utterances: List[str] = []
    tables: List[List[Dict[str, str]]] = []

    for turn in turns:
        user_match = re.search(r"\*\*User\*\*\s*\n>\s*(.+?)(?=\n\*\*Agent\*\*)", turn, re.DOTALL)
        if user_match:
            raw = user_match.group(1)
            # collapse multi-line blockquote ("> line1\n> line2") into one string
            lines = [l.lstrip(">").strip() for l in raw.splitlines()]
            user_utterances.append(" ".join(l for l in lines if l).strip())

        table = _extract_last_table(turn)
        if table:
            tables.append(table)

    final_shortlist = tables[-1] if tables else []

    return {
        "trace_id": path.stem,
        "user_utterances": user_utterances,
        "final_shortlist": final_shortlist,  # list of {name, url, test_type}
    }


def _extract_last_table(turn_text: str):
    table_blocks = re.findall(
        r"(\|.+\|\n\|[-\s|]+\|\n(?:\|.*\|\n?)+)", turn_text
    )
    if not table_blocks:
        return None
    return _parse_markdown_table(table_blocks[-1])


def _parse_markdown_table(block: str) -> List[Dict[str, str]]:
    lines = [l for l in block.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip header + separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        rows.append(row)
    normalized = []
    for row in rows:
        name = row.get("Name", "")
        url_field = row.get("URL", "")
        url_match = re.search(r"<(https?://[^>]+)>", url_field) or re.search(r"\((https?://[^)]+)\)", url_field)
        url = url_match.group(1) if url_match else url_field.strip()
        normalized.append({
            "name": name,
            "url": url,
            "test_type": row.get("Test Type", ""),
        })
    return normalized


def load_all_traces(traces_dir: str) -> List[Dict[str, Any]]:
    dir_path = Path(traces_dir)
    return [parse_trace(p) for p in sorted(dir_path.glob("*.md"))]


if __name__ == "__main__":
    import json
    traces = load_all_traces("data/traces")
    for t in traces:
        print(f"{t['trace_id']}: {len(t['user_utterances'])} user turns, "
              f"{len(t['final_shortlist'])} expected recommendations")
    print()
    print(json.dumps(traces[0], indent=2))

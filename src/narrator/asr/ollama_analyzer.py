"""AI analysis of accumulated autoconfigure reports via local Ollama.

Usage:
    narrator --autoconfigure --analyze-reports

Loads every run summary from scripts/autoconfigure_reports/ and asks a local
Ollama model (qwen3.5:9b by default) to analyze patterns and suggest tuning
changes. Summaries are self-contained (every evaluated configuration with its
pooled WER, error breakdown, caption delay, worker load and top inserted
phrases, plus the offline ceiling and the decision), so the model sees
everything needed for a diagnosis.

The prompt is deliberately concise to stay within qwen3.5:9b's 262K context
even with hundreds of clips.  If you add more models to Ollama, edit
DEFAULT_MODEL.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from narrator.asr.autoconfigure import REPORTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3.5:9b"

_ANALYSIS_PROMPT = """\
You are analyzing results of `narrator --autoconfigure`, which searches the
whisper.cpp model and streaming-decode settings under which a LIVE Polish
speech captioner agrees best with reference subtitles (mostly YouTube
auto-captions) of a library of recordings.

Each run summary contains:
- the operator's current decode settings (baseline) and the decision taken
  (whether new settings were written, and why)
- every evaluated configuration: model, decode settings (step_ms,
  agreement_passes, end_silence_ms, vad_threshold, max_context_seconds,
  vad_preroll_ms, prompt_carryover_chars, language_whitelist), the excerpt
  set it was scored on (screen = search, dev = confirmation, holdout =
  never used for tuning, trusted = human subtitles), pooled WER with
  substitution/deletion/insertion counts, the median caption delay, the
  worker load (busy_fraction; near 1.0 the machine barely keeps up) and the
  most frequent inserted phrases (hallucinations)
- the offline ceiling: the WER of a non-streaming large-v3 transcription,
  i.e. the best agreement whisper reaches with these references

Analyze patterns across all runs: which settings consistently help or hurt,
whether errors are dominated by deletions, insertions or substitutions, how
far streaming is from the offline ceiling, recurring hallucinated phrases,
and what to try next. Answer with the analysis fields (summary, best_model,
recommended_settings, issues, suggestions). Do NOT echo the input back.
""".strip()

# JSON schema passed to Ollama's `format` parameter. With a bare
# format="json" the model is free to emit any object -- qwen3.5 reliably
# echoed the input report shape back instead of the requested analysis
# (summary/best_model came out None). A schema constrains it to the answer.
_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "best_model": {"type": "string"},
        "recommended_settings": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "best_model", "recommended_settings", "issues", "suggestions"],
}

_ENTRY_FIELDS = (
    "label", "set", "model", "settings", "wer", "substitutions", "deletions",
    "insertions", "reference_words", "latency_p50", "busy_fraction",
)


def _load_reports(reports_dir: Path = REPORTS_DIR) -> list[dict]:
    """Every run summary (*_summary.json) as a compact dict: the decision,
    the ceiling, and each evaluated configuration's headline numbers (the
    per-excerpt detail stays in the files)."""
    if not reports_dir.is_dir():
        return []

    summaries: list[dict] = []
    for path in sorted(reports_dir.glob("*_summary.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("skipping unreadable report %s: %s", path, e)
            continue
        stages = {
            stage: [
                {
                    **{field: entry.get(field) for field in _ENTRY_FIELDS},
                    "top_insertions": entry.get("top_insertions", [])[:5],
                }
                for entry in entries
            ]
            for stage, entries in report.get("stages", {}).items()
        }
        ceiling = report.get("ceiling", {})
        summaries.append(
            {
                "report": path.name,
                "started": report.get("started"),
                "baseline_model": report.get("baseline", {}).get("model"),
                "decision": report.get("decision"),
                "ceiling": {
                    "model": ceiling.get("model"),
                    "wer": ceiling.get("wer"),
                    "unmatchable_excerpts": len(ceiling.get("unmatchable", [])),
                },
                "library": report.get("library"),
                "stages": stages,
            }
        )
    return summaries


# Ollama generation options. `format: "json"` constrains the model to emit a
# single JSON object, and `think: false` disables the thinking mode qwen3.5
# defaults to -- with thinking on, the model burns its output on a reasoning
# transcript and either answers with prose or runs out before emitting any
# JSON (measured: "Could not parse JSON from model response" every run).
_OLLAMA_TIMEOUT_S = 900.0

def _query_ollama(model: str, prompt: str, context: str) -> str:
    """Ask Ollama a question. Returns the model's text response."""
    import urllib.request
    import urllib.error

    url = "http://localhost:11434/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": f"{prompt}\n\n=== REPORT DATA ===\n{context}",
            "stream": False,
            "format": _ANALYSIS_SCHEMA,
            "think": False,
            "options": {"temperature": 0.3, "top_p": 0.9},
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT_S) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama not reachable at localhost:11434 — is `ollama serve` running? ({e})"
        ) from e


def analyze_reports(model: str = DEFAULT_MODEL) -> dict:
    """Load all reports, ask the Ollama model for analysis, return parsed JSON."""
    reports = _load_reports()
    if not reports:
        return {"error": f"No run summaries found in {REPORTS_DIR}. Run `narrator --autoconfigure` first."}

    # Truncate if there's too much data — qwen3.5:9b has 262K context but
    # we still bound the prompt to keep latency reasonable.
    context = json.dumps(reports, indent=2)
    max_context_chars = 200_000  # leaves room for the prompt + response
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n...(truncated)"
        logger.warning(
            "report context truncated from %d to %d chars", len(context), max_context_chars
        )

    logger.info("sending %d reports (%d chars) to Ollama model %s", len(reports), len(context), model)

    response = _query_ollama(model, _ANALYSIS_PROMPT, context)

    # Strip any thinking blocks if the model used them
    if "<think>" in response.lower():
        start = response.lower().find("</think>")
        if start != -1:
            response = response[start + 8 :]

    if not response.strip():
        return {
            "raw_response": response,
            "error": (
                "Ollama returned an empty response. The model likely exhausted "
                "its context/output on the report data (try fewer reports, or "
                "a larger-context model) or is not the model named in "
                "DEFAULT_MODEL. Run `ollama list` to confirm it is pulled."
            ),
        }

    # Try to parse JSON from the response
    try:
        result = json.loads(response.strip())
    except json.JSONDecodeError:
        # The model might have wrapped it in ```json ... ``` or added prose
        # around the object; recover the outermost {...} if present.
        stripped = response.strip()
        if stripped.startswith("```json"):
            stripped = stripped[7:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        result = None
        try:
            result = json.loads(stripped.strip())
        except json.JSONDecodeError:
            start, end = stripped.find("{"), stripped.rfind("}")
            if start != -1 and end > start:
                try:
                    result = json.loads(stripped[start : end + 1])
                except json.JSONDecodeError:
                    result = None
        if result is None:
            return {
                "raw_response": response,
                "error": "Could not parse JSON from model response",
            }

    return result


def save_analysis_result(result: dict, reports_dir: Path = REPORTS_DIR) -> Path:
    """Persist an analysis result as a timestamped JSON file."""
    from datetime import datetime, timezone

    reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = f"analysis_{now.strftime('%Y%m%dT%H%M%S')}.json"
    path = reports_dir / filename
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("analysis saved to %s", path)
    return path

"""`recant feedback`: numbers only, plus what got in the developer's way.

The bundle (`recant-feedback-<date>.zip`, written here, sent by nobody): each bench and eval run's report, the `recant
cost` summary of a store, Recant's version and commit, Python and OS, the wrapper features used, error classes with
counts and the Recant function they came from, and the friction log's summary. It never holds prompts, responses, tool
inputs or outputs, notes text, questions, keys, file paths, hostnames or user names: every string in it passes an
allow-list of field names and patterns (`scrub`), and a dict keyed by anything else (a scope, a session) becomes a list
of its values. The form's answers are the one exception, and they are the developer's own words, shown back before the
bundle is written. The command prints the whole bundle and asks before writing it; it ends by saying where to send it."""
from __future__ import annotations

import io
import json
import platform
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import friction, known_hosts
from .version import info as version_info

ISSUES = "https://github.com/recant-memory/recant/issues/new"

# The allow-list. Keys: only the field names Recant's reports, the cost summary and the bundle itself write
# — a name a developer chose (a scope, a job, a customer) is never one. A dict keyed by data (`LISTED_BY_VALUE`) becomes
# a list of its values; error classes are kept only when Python, the SDK or Recant defines them (else "HostError").
# Strings pass by their field's rule, never by their shape. A test fails the build when a report gains an unlisted key.
FIELDS = frozenset("""
A B C P absent accepted accuracy against_history already_batched_calls answer_call answers_usd arm arms as_measured
attempts background_jobs batch bench_arms bill_usd both_succeeded break_cost break_tokens bundle by_kind cache_lifetime
cache_read cache_write calibration_tasks calls calls_max calls_median calls_per_task change_to_bill change_to_bill_at_most
client_mode clients bench_clients commands commit commit_from config_changes config_hash correct correct_words_only accuracy_words_only interval_words_only cost cost_usd delta difference
discarded discarded_usd entries equivalence equivalent errors estimate estimated_usd failed_answers failed_calls
failed_calls_usd failed_commands failed_tasks failures features first_command_ts first_report_ts first_task_usd friction
gaps grader growth_per_call_median header held help_calls history id input input_per_mtok interval jobs kind latency
levers machine memory_measured memory_served_median memory_v2 met minutes_to_first_report mixed_versions model models n
net none not_equivalent notes eval_notes_arm opted_in order_seed os output output_per_call_median output_per_mtok over_60_min
over_switch_point p50 p95 paired_difference_per_task paired_task_difference_s pairs pairs_with_a_failure pass_rate
per_call per_correct_usd per_passed_task_relative per_question_usd per_task predicted_usd predictions
predictions_anchored_in predictions_sha256 predicted_usd_at_freeze price_tags priced prices python quality questions questions_sha256 ratio
read_x realized_usd reasoning reasoning_share_of_output recall recant recant_overhead refused refused_reasons relative report
relative_interval remaining_calls_at rent_added rent_start reports retries run_id runs salts save saves_usd scopes scoring
seconds sessions sessions_carried setup shape share share_predicted source split start_context_median start_read
start_write store_copied store_from_copy streaming tasks_paired tasks_run text tokens tokens_est tokens_per_scope
total_spend_usd ttft uncached under_5_min unpriced unpriced_calls unpriced_models update usd values verdict version warm_start_share warnings
without_first_task write_added write_on_request_path write_path_s write_path_usd write_x written 5_to_60_min
coverage calls_seen calls_unseen share_calls usd_seen usd_unseen share_usd unseen_unpriced watched unseen provider host
requests with_usage usd_list usage wrapper_errors state row batch_factor cache_write_long above over
5 10 25 50 100
""".split()) | {"10% fewer calls", "10% smaller start context", "10% less added per call", "10% less output"}
LISTED_BY_VALUE = frozenset({"scopes", "tokens_per_scope", "refused_reasons", "jobs", "opted_in", "errors", "prices"})

_HEX = re.compile(r"^[0-9a-f]{7,64}$")
_VERSION = re.compile(r"^\d+\.\d+(\.\d+)?([a-z]+\d*)?$")
_MODEL = re.compile(r"^claude-[a-z0-9.\-]+$")
# A public model name: a known family, optionally behind a router's vendor prefix. A host's own model name
# ("house-model-3") is not one, and is left out.
_PUBLIC_MODEL = re.compile(r"^~?([a-z0-9-]+/)?(claude|gpt|o\d|chatgpt|gemini|gemma|glm|mistral|mixtral|codestral|magistral|"
                           r"deepseek|grok|llama|qwen|kimi|command|anthropic\.claude)[a-z0-9.:\-]*$")
_OTHER_HOST = re.compile(r"^other:[0-9a-f]{12}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][0-9:.]+([+-]\d{2}:\d{2}|Z)?)?$")
_RUN_ID = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{4}$")
_ARM = re.compile(r"^(off|safe|safe\+(reasoning|effort)\.agent)$")
LEVERS = frozenset({"advice.trim_tool_output", "breakpoint.add", "fix.duplicate_background", "fix.segment_changed",
                    "fix.tool_loop", "fold.window_only", "handoff.session", "postreply.write", "ttl.prefix",
                    "memory.core_budget", "memory.slice_budget", "memory.memory_budget", "cost per task"})
_BY_JOB = re.compile(r"^(batch|effort|reasoning|fix\.output_cap)\.(.+)$")
_BY_KEY = re.compile(r"^(fix\.metered_not_billed|fix\.unmetered)\.(.+)$")
ENUMS = {"commit_from": {"source checkout", "install record", "release build", "source checkout, uncommitted changes", "unknown"},
         "kind": {"safe", "your_call", "applied", "bench", "eval_memory", "fix_cause", "advice", "finding"},
         "state": {"applied", "available", "needs_evidence", "reverted", "finding", "host_advice", "known", "no_slack",
                   "not_worth_building", "checked", "opted_in"},
         "os": {"Darwin", "Linux", "Windows"}, "machine": {"arm64", "x86_64", "AMD64", "aarch64"},
         "client_mode": {"sync", "async"}, "bench_clients": {"sync", "async"}, "arm": {"A", "B"}, "bundle": {"recant-feedback"},
         "arms": {"none", "history", "notes", "memory_v2"}, "scoring": {"mechanical: the ANSWER line"},
         "verdict": {"held", "loss observed", "interval too wide"}, "store_copied": {"tenants", "versions", "machinery"},
         "prices": {"your table"}, "predictions_anchored_in": {"the manifest and the friction log",
                                                                "the manifest only: this machine's friction log has no entry for the run"}}


def lever_id(s: str) -> Optional[str]:
    """A lever id, its job or key suffix kept only when it is Recant's default ("agent"); otherwise "<job>"/"<key>"."""
    if s in LEVERS:
        return s
    m = _BY_JOB.match(s)
    if m:
        return s if m.group(2) == "agent" else f"{m.group(1)}.<job>"
    m = _BY_KEY.match(s)
    if m:
        return f"{m.group(1)}.<key>"
    m = re.match(r"^fix\.unwrapped\.(.+)$", s)                    # a provider id from the known list is public
    return s if m and m.group(1) in known_hosts.PROVIDERS else None


def _rules():
    hexed = lambda s: s if _HEX.match(s) or s == "unknown" else None          # noqa: E731
    return {"version": lambda s: s if _VERSION.match(s) else None, "python": lambda s: s if _VERSION.match(s) else None,
            "commit": hexed, "config_hash": hexed, "predictions_sha256": hexed, "questions_sha256": hexed,
            "model": lambda s: s if _MODEL.match(s) or _PUBLIC_MODEL.match(s) else None,
            "provider": lambda s: s if s in known_hosts.PROVIDERS or s == "none" else None,
            "host": lambda s: s if _OTHER_HOST.match(s) or s == "none" or known_hosts.provider_of_host(s) else None, "models": lambda s: s if _MODEL.match(s) else None,
            "unpriced_models": lambda s: s if _MODEL.match(s) or _PUBLIC_MODEL.match(s) else None,
            "written": lambda s: s if _DATE.match(s) else None, "run_id": lambda s: s if _RUN_ID.match(s) else None,
            "A": lambda s: s if _ARM.match(s) else None, "B": lambda s: s if _ARM.match(s) else None,
            "bench_arms": lambda s: s if _ARM.match(s) else None, "id": lever_id,
            "prices": lambda s: s if s == "your table" or re.match(r"^the library's list prices as of \d{4}-\d{2}-\d{2}$", s) else None,
            "source": lambda s: s if re.match(r"^list prices as of \d{4}-\d{2}-\d{2}$", s) else None}


RULES = _rules()


def _known_errors() -> frozenset:
    import builtins
    names = {n for n, v in vars(builtins).items() if isinstance(v, type) and issubclass(v, BaseException)}
    names |= {"APIError", "APIStatusError", "APIConnectionError", "APITimeoutError", "APIResponseValidationError",
              "BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError", "ConflictError",
              "UnprocessableEntityError", "RateLimitError", "InternalServerError", "OverloadedError",     # the SDK's
              "RemoteProtocolError", "LocalProtocolError", "ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout",
              "ReadError", "WriteError", "ConnectError", "NetworkError", "TimeoutException", "ProtocolError",   # httpx's
              "IncompleteReadError", "ProviderError", "BudgetExceeded", "RecordError", "CeilingReached", "ValidationError"}
    return frozenset(names)


KNOWN_ERRORS = _known_errors()


def error_name(s: str) -> str:
    """An error class (optionally `grade:` or `… in recant.<function>`) as the bundle may hold it."""
    m = re.match(r"^(grade:)?([A-Za-z_][A-Za-z0-9_]*)( in (recant(\.([a-z_][a-z0-9_]*|<[a-z]+>))+|unknown))?$", s or "")
    if not m:
        return "HostError"
    return (m.group(1) or "") + (m.group(2) if m.group(2) in KNOWN_ERRORS else "HostError") + (m.group(3) or "")


def safe_string(s: str, field: Optional[str] = None) -> bool:
    return scrub_string(s, field) is not None


def scrub_string(s: str, field: Optional[str]) -> Optional[str]:
    if field in RULES:
        return RULES[field](s)
    if field in ENUMS:
        return s if s in ENUMS[field] else None
    return None


def scrub(x, *, field: Optional[str] = None, dropped: Optional[list] = None):
    """The bundle's projection: numbers, booleans and None as they are; a string only when its field's rule passes it; a
    dict's keys only when they are Recant's field names (an unknown key is dropped, counted, never kept); a dict keyed
    by data becomes a list of its values; error classes as `error_name` gives them."""
    dropped = dropped if dropped is not None else []
    if isinstance(x, bool) or x is None or isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        out = scrub_string(x, field)
        if out is None:
            dropped.append(1)
        return out
    if isinstance(x, (list, tuple)):
        return [scrub(v, field=field, dropped=dropped) for v in x]
    if isinstance(x, dict):
        if field in ("errors", "wrapper_errors"):
            merged: dict = {}
            for k, v in x.items():
                k2 = error_name(str(k))
                merged[k2] = merged.get(k2, 0) + (v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0)
            return merged
        if field in LISTED_BY_VALUE:
            return {"values": [scrub(v, dropped=dropped) for v in x.values()]}
        out = {}
        for k, v in x.items():
            if isinstance(k, str) and k in FIELDS:
                out[k] = scrub(v, field=k, dropped=dropped)
            else:
                dropped.append(1)
        return out
    dropped.append(1)
    return None


def _reports(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        p = d / "report.json"
        if p.exists():
            r = json.loads(p.read_text())
            r.pop("verdict", None)                                  # sentences; their numbers are in the rest of the report
            out.append({"kind": "bench" if "cost" in r or "estimate" in r else "eval_memory", "report": r})
    return out


def features(reports: list[dict]) -> dict:
    f = {"bench_clients": sorted({r["report"].get("header", {}).get("client_mode") for r in reports} - {None}),
         "streaming": any(((r["report"].get("latency") or {}).get(a) or {}).get("ttft", {}).get("n", 0) for r in reports for a in "AB"),
         "eval_notes_arm": any("notes" in (r["report"].get("header", {}).get("arms") or []) for r in reports),
         "bench_arms": sorted({v for r in reports for v in ((r["report"].get("header", {}).get("arms") or {}).values()
                                                         if isinstance(r["report"].get("header", {}).get("arms"), dict) else [])})}
    return f


def build(*, bench_dirs: list[Path], eval_dirs: list[Path], cost_json: Optional[dict] = None, form: Optional[dict] = None) -> tuple[dict, int]:
    reports = _reports(bench_dirs) + _reports(eval_dirs)
    dropped: list = []
    body = {"bundle": "recant-feedback", "written": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "recant": version_info(),
            "python": platform.python_version(), "os": platform.system(), "machine": platform.machine(),
            "features": features(reports), "reports": [r for r in reports], "cost": cost_json, "friction": friction.summary()}
    clean = scrub(body, dropped=dropped)
    if form:
        clean["form"] = {k: str(v) for k, v in form.items()}          # the developer's own words, shown back before writing
    return clean, len(dropped)


QUESTIONS = [
    ("minutes_to_first_report", "Minutes from install to your first report"),
    ("had_to_read_code_or_guess", "Where did you have to read Recant's code, guess, or search? (one line each)"),
    ("errors_hit", "Which errors did you hit?"),
    ("could_not_measure", "What did you want to measure that the kit couldn't?"),
    ("leave_it_on", "Would you leave Recant on? yes / no / not yet, and why"),
    ("anything_else", "Anything else"),
]


def ask_form(ask: Callable = input, say: Callable = print) -> dict:
    s = friction.summary() or {}
    errs: dict = {}
    for k, v in (s.get("errors") or {}).items():
        errs[error_name(k)] = errs.get(error_name(k), 0) + v
    hit = [f"{k} ×{v}" for k, v in errs.items()]
    hit += [f"{error_name(k)} ×{v} inside a wrapped call" for k, v in (s.get("wrapper_errors") or {}).items()]
    pre = {"minutes_to_first_report": s.get("minutes_to_first_report"), "errors_hit": ", ".join(hit)}
    say("Six questions, about two minutes; every answer is optional (Enter skips; a pre-filled answer is kept on Enter).")
    out = {}
    for key, q in QUESTIONS:
        hint = f" [{pre[key]}]" if pre.get(key) not in (None, "") else ""
        a = ask(f"{q}{hint}: ").strip()
        if a or hint:
            out[key] = a or str(pre[key])
    return out


def run(*, bench_dirs, eval_dirs, cost_json=None, out_dir: Path = Path("."), form: bool = True,
        ask: Callable = input, say: Callable = print) -> Optional[Path]:
    """Asks before writing, always (no flag skips it)."""
    answers = ask_form(ask, say) if form else None
    bundle, dropped = build(bench_dirs=bench_dirs, eval_dirs=eval_dirs, cost_json=cost_json, form=answers)
    text_ = json.dumps(bundle, indent=1, ensure_ascii=False)
    say(text_)
    say(f"({dropped} strings left out by the allow-list: text, paths and names never go in the bundle)")
    if ask("Write this bundle? [y/N] ").strip().lower() not in ("y", "yes"):
        say("Nothing written.")
        return None
    p = Path(out_dir) / f"recant-feedback-{datetime.now(timezone.utc):%Y-%m-%d}.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bundle.json", text_ + "\n")
    p.write_bytes(buf.getvalue())
    say(f"Written: {p}. Recant sent nothing. To send it, open a new issue at {ISSUES} and attach the file.")
    return p

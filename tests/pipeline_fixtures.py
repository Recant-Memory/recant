"""A synthetic corpus and a scripted model, so every harness job runs end to
end with no network. The script misbehaves on purpose in a few places."""
import json
import re

from recant.provider import ProviderError, Usage, text
from recant.provider.base import Raw

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


def build_corpus(root):
    """36 dailies over six months; each has decisions, a cause, a reversal and filler."""
    (root / "notes/.hidden/talk").mkdir(parents=True)
    for m in MONTHS:
        for day in range(1, 7):
            d = f"{m}-{day:02d}"
            lines = [f"# {d}", "filler one", f"DECISION: on {d} we chose plan {day} for the scanner",
                     "filler two", "filler three", f"CAUSE: on {d} the outage was caused by cache key {day}",
                     "filler four", f"DECISION: on {d} the timeout became {day * 10} seconds",
                     "filler five", "filler six", "filler seven", "filler eight", "filler nine", "filler ten",
                     f"REVERSED: on {d} we stopped believing the vendor claim number {day}",
                     "filler eleven", "filler twelve", "filler thirteen", "filler fourteen"]
            (root / f"notes/{d}.md").write_text("\n".join(lines) + "\n")
    (root / "notes/.hidden/talk/2026-01-01.txt").write_text("[s#L1] User: hello\n[s#L2] Assistant: hi\n")
    return root


def _raw(payload, usage=Usage(input=200, output=50)):
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return Raw((text(body),), "end_turn", usage)


def _user(req):
    return req.messages[0].content[0].text


def scripted(model, req):
    user = _user(req)
    if req.job == "nominate":
        path = re.search(r"Episode: (\S+)", user).group(1)
        if path.endswith("2026-01-02.md"):
            return _raw("this is not json")
        if path.endswith("2026-01-03.md"):
            return ProviderError("server", "500", retryable=True)
        spans = []
        for n, line in re.findall(r"^(\d+): (.*)$", user, flags=re.M):
            for prefix, marker in (("DECISION:", "rule_or_decision"), ("CAUSE:", "root_cause"), ("REVERSED:", "reversal")):
                if line.startswith(prefix):
                    spans.append({"marker": marker, "start_line": int(n), "end_line": int(n), "quote": line})
        if path.endswith("2026-01-04.md"):
            spans += [{"marker": "root_cause", "start_line": 2, "end_line": 2, "quote": "a quote that is not there"},
                      {"marker": "invented_marker", "start_line": 2, "end_line": 2, "quote": "filler one"},
                      {"marker": "root_cause", "start_line": 900, "end_line": 901, "quote": "x"}]
        return _raw({"spans": spans})
    if req.job in ("draft_marked", "draft_unmarked"):
        m = re.search(r"The moment is lines (\d+)-(\d+)", user)
        start, end = int(m.group(1)), int(m.group(2))
        if "No marker fired" in user:
            if start == 17:
                return _raw({"skip": True, "item_type": "yield", "question": "", "candidate_answer": "",
                             "answer_start_line": 0, "answer_end_line": 0, "superseded_answer": "", "search_terms": []})
            line = re.search(rf"^{start}: (.*)$", user, flags=re.M).group(1)
            return _raw({"skip": False, "item_type": "yield", "question": f"What is on line {start}?",
                         "candidate_answer": f"The line reads: {line}", "answer_start_line": start,
                         "answer_end_line": end, "superseded_answer": "", "search_terms": [line]})
        line = re.search(rf"^{start}: (.*)$", user, flags=re.M).group(1)
        if "timeout became 30" in line:
            return _raw({"skip": False, "item_type": "yield", "question": "q", "candidate_answer": "a",
                         "answer_start_line": 9000, "answer_end_line": 9001, "superseded_answer": "", "search_terms": []})
        reversal = line.startswith("REVERSED")
        return _raw({"skip": False, "item_type": "supersession" if reversal else "yield",
                     "question": f"What happened here: {line[:40]}?", "candidate_answer": f"It says that {line.split(': ', 1)[1]}",
                     "answer_start_line": start, "answer_end_line": end,
                     "superseded_answer": "the vendor claim" if reversal else "", "search_terms": line.split()[-2:]})
    if req.job == "draft_noise":
        a = int(re.search(r"moment is lines (\d+)-", user).group(1))
        day = re.search(r"Episode: notes/(\S+)\.md", user).group(1)
        return _raw({"skip": False, "question": f"What colour was the dashboard on {day} near line {a}?",
                     "search_terms": ["dashboard", "colour"]})
    if req.job == "draft_seeded":
        if "unanswerable" in user:
            return _raw({"answerable": False, "candidate_answer": "", "episode_path": "", "answer_start_line": 0,
                         "answer_end_line": 0, "search_terms": ["ruleset", "version"]})
        path = re.search(r"=== (\S+)", user).group(1)
        return _raw({"answerable": True, "candidate_answer": "Plan 1 was chosen for the scanner.", "episode_path": path,
                     "answer_start_line": 3, "answer_end_line": 3, "search_terms": ["plan", "scanner"]})
    if req.job == "model_review":
        full, user = user, re.search(r"Candidate answer: (.*)", user).group(1)
        if "timeout became 20" in user:
            return _raw({"outcome": "fix", "reason": "wrong_answer", "fixed_answer": "The timeout became 20 seconds.",
                         "hard": False, "note": "tightened"})
        if "timeout became 40" in user:
            return _raw({"outcome": "discard", "reason": "bad_question", "fixed_answer": "", "hard": False, "note": ""})
        if "timeout became 50" in user:
            return _raw({"outcome": "accept", "reason": "not-a-reason", "fixed_answer": "", "hard": False, "note": ""})
        if "timeout became 60" in user:
            return Raw((), "refusal", Usage(input=10))
        return _raw({"outcome": "accept", "reason": "confirmed", "fixed_answer": "",
                     "hard": "cache key 1" in user, "note": ""})
    if req.job == "judge":
        ref = re.search(r"Reference answer: (.*)", user).group(1)
        ans = re.search(r"Answer to score: (.*)", user).group(1)
        return _raw({"correct": ref.split()[-1] == ans.split()[-1], "rationale": "last words compared"})
    raise AssertionError(f"unscripted job {req.job}")

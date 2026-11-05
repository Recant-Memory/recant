# Changelog

What changed in each version, and the measurements behind the numbers the README, the defaults and the code
comments use. Each measurement says what it was run on; none of it is a promise about your traffic. Measure your own
with `recant bench` and `recant eval memory`.

## 0.3.2

From an outside developer's first run of the whole kit, on their own two-agent app (a streamed async researcher on
Sonnet 5.5 with tools, a sync writer on Haiku 5.5; 40 tasks; November 2026). Each item is something they hit.

- **The wrapper takes a tool loop as the SDK writes it.** A request whose `messages` hold the SDK's own response
  blocks (`{"role": "assistant", "content": r.content}`) raised `TypeError: Object of type ToolUseBlock is not JSON
  serializable` on the second call of every wrapped tool loop. The wrapper now reads the request in the form the SDK
  sends it: with no lever in force, the bytes on the wire are the raw SDK's, byte for byte, on all six call paths;
  a request holding anything but plain data and the SDK's own models (an iterator, a tuple, a file, a sentinel) is sent
  as the host's own objects, untouched and with no lever, so the SDK prepares it itself;
  with a safe lever, they differ only in its cache markers (an opted-in effort or reasoning lever changes those settings,
  as you asked).
- **Recant's own failure does not reach your code.** If preparing a call, reading its response or a stream event,
  recording it, opening or closing its run, or reading the switch fails inside Recant, the call goes out as you built
  it, you get every event and the response, and the failure is counted in the friction log (`wrapper_errors`), which
  the feedback form pre-fills. The coverage watch never fails an HTTP request.
- **Prices.** The list (the vendor's pricing page, read 2026-11-05) gains Claude Haiku 5.5, with its two rate cards:
  $0.10/$0.50 per million tokens up to a 100,000-token prompt, $0.50/$2.50 over it, the prompt counting cache reads and
  writes, each request priced on its own. Without it a host using Haiku 5.5 got its calls left out of the bill and a
  bench verdict of "cost not measured". The Sonnet 5.5 row read cache hits at 0.10x input ($0.20); the page reads
  0.05x ($0.10). The list also gains Mythos 5.1, Fable 5, Opus 4.8, 4.7 and 4.6, and Sonnet 4.6 from the same page.
  `recant bench` names any model the first task called that has no price, before it runs the rest.
- **Your own price table** is documented (README, "Your own prices"), checked row by row with a one-line error naming
  the model and the field (it printed a traceback), and may carry a second card (`above`, `over`). `recant cost --json`
  prints each model's row in the same format.
- **A run's name is its job.** `client.run("researcher")` and `client.run("writer")` are reported and sized apart
  (`effort.writer`); an opted-in lever reaches only its own job. Before, every call was wrap's job, "agent".
- **A bench report at other prices** restates both sides of every prediction: the frozen part is each lever's share,
  and cost per task's base, the calibration's measured cost, is repriced at the report's table. Re-pricing after the
  run had moved the per-task lines to 111% and 122% of prediction; on the same run they read 98% and 104%.
- **Coverage.** A share below 100% never prints as 100% (99.8%); an unpriced model is named as left out of the cost
  share. `recant feedback --store` now carries the store's unseen traffic by provider and public model; it carried only
  the bench's and the eval's. Each lever in the bundle carries its state beside its kind.
- **Reports say what they mean.** The bill's reasoning share and the job's (of output tokens) are labelled; "per task"
  latency is per run; "tool calls 0" on a wrapped host says tools are not metered; the billing line says the
  provider's bill is not reconciled; the levers header states what a weekly figure is scaled from.
- **`recant eval memory`** leads with each arm's words-only score, adds a total spend line, the answer call's p95 and
  Recant's own time per call. The question drafter writes the fewest words that identify a value as the key, and asks
  only about facts the person stated: 6 of the developer's 12 misses were right answers scored against keys such as
  holding a word the answer need not repeat (a project's name with "Project" in front). The scoring rule is unchanged.
- **Progress.** `recant bench` prints a line per task run; `recant eval memory` says when it writes a scope and how many
  answers are in. Each ran 18 to 27 minutes in silence.
- **`--help`** says what each flag takes, and its default where it has one. `recant --version` from PyPI says in words that the commit is
  not recorded; it printed `(unknown, unknown)`.

## 0.3.1

The first release on PyPI, published from the public repository by trusted publishing. The library's code is 0.3.0's,
with internal references taken out of its comments; no behaviour changed.

## 0.3.0

- **Coverage: what Recant can't see.**
  - `recant.coverage.watch()` counts model-API requests that went past Recant. It keeps counts only, and hashes an unknown host's domain.
  - Every report has a Coverage line.
  - A bench verdict is qualified below 90% of calls or of list-priced cost, and gives no overall saving below 50%.
  - `recant coverage --usage` matches your own per-run totals against what Recant metered.
- **A tool's own traffic.** `recant.metering.from_claude_code` and `from_agent_sdk` meter Claude Code's and the Claude Agent SDK's end-of-run totals in their own report rows: billed (0 on a subscription) and priced at list. No lever applies to them.
  - On two Claude Code 2.1 runs the metered totals equalled the tool's own report in every token class and in cost.
  - The per-step usage on the assistant messages adds up to the same input and cache totals. Its output count is a placeholder: 35 output tokens against the run's 148 on a three-call run.
- **The feedback bundle** carries the coverage counts by provider id and public model name.
- **Known limits** are stated in the README.

## 0.2.x: `recant bench`, `recant eval memory`, `recant feedback`

**What changed in 0.2.x:**
- **0.2.0** added the bench kit: paired arms with a cache salt and a fresh store each, predictions frozen and hashed before the compared tasks, and failures counted in their arm. It also added `recant off`, the latency figures and the friction log.
- **0.2.1–0.2.3** added `recant eval memory`, resumable. The value-match score now sits beside the words-only score.
- **0.2.4**: a task's pass or fail is its grader's. A failed call counts its billed usage, but a host that retries and finishes is not failed for it.
- **0.2.5**: each model gets its own form of "reasoning off". Sonnet 5.5 takes `{"type": "between_tools"}`. Opus 5.5 and Fable 5.1 refuse both forms, so the advisor no longer offers the lever for them.

**Measurements, all in October and November 2026, on the library's own examples and public benchmarks:**

- **A/A** (safe against safe, separate salts, `examples/cost_quickstart.py`, 30 pairs):
  - cost −1.0% a task (95% −4.7% to +2.9%);
  - the latency interval contains zero.
- **Off against safe** (the same example, 10 calibration tasks and 30 pairs):
  - −44.4% a passed task (95% −46.0% to −42.7%);
  - `breakpoint.add` realized 98% of its frozen prediction;
  - the tasks are too easy to show a quality loss (100% in both arms).
- **τ-bench** (retail and airline, `claude-sonnet-5`, 10 calibration tasks and 150 pairs):
  - −73.0% a passed task (95% −77.1% to −67.8%);
  - equivalence held on all 4,120 calls, so the model's input was identical apart from cache markers;
  - quality, as measured: −1.3 points (95% −8.0 to +5.3). That interval is too wide to settle quality either way.
- **The reasoning lever** (τ-bench, `claude-sonnet-5-5`, 150 pairs, safe against safe plus reasoning off):
  - reasoning fell from 277 to 23 tokens a task;
  - cost +3.1% a task (95% −0.6% to +6.7%);
  - quality −1.3 points (95% −7.3 to +4.7), so not held.
  - The lever stays off for tool-calling work like that.
- **The breakpoint predictor's shared head** (0.2.0): before it, the prediction for `breakpoint.add` under-predicted two measured runs by 23–24%:
  - sync, $1.406 realized against $1.148 predicted;
  - async streamed, $1.3084 against $1.0533.
  - The cause: a run's first call reads the system block that other runs on the same setting cached. The per-call data is `tests/data/breakpoint_acceptance.json`.
- **Silent streams** (one host's network path, Sonnet 5.5, five streams in flight):
  - Without Recant in the path, 26 of 163 streamed calls went silent for 30 seconds or more after their first event, and 6 were closed by the upstream after about 480 seconds.
  - With Recant, 11 of about 470 broke.
  - None of 480 non-streamed calls did.
  - The same stalls occurred with no proxy in the path.
- **Memory on synthetic principals' sessions** (`recant eval memory`: six principals, 120 sessions, 255 questions from their answer key, answered on `claude-opus-5-5`):

  | arm | words only | value match |
  |---|---|---|
  | carried history | 86.3% | 100% |
  | managed notes | 86.3% | 100% |
  | memory v2 | 57.3% | 77.3% |

  - Memory v2 answered none of the 53 questions whose answer came only from a tool result. Managed notes kept all 53.

## 0.1.0: the drop-in wrapper, the cost advisor, managed notes

**What changed:**
- `recant.wrap` for the Anthropic SDK, sync and async, streamed or not;
- `recant cost`, with the safe levers applied and checked;
- managed notes as the default memory, and memory v2 as the fallback.

**The defaults, and what is behind each:**

- **The serve budget** (`serve.core_tokens: 25000`, `serve.slice_tokens: 12500`). A starting point, not a measured optimum: a cached core large enough for a principal's standing facts, and a retrieved slice half its size so a session's served memory stays bounded. Your sessions decide the right sizes; measure them with `recant eval memory` and set both in your own kernel.
- **The notes threshold** (`serve.notes_default_tokens: 95000`). On synthetic principals, notes and memory cost the same a session at 95k counted tokens of notes. Below that, notes are cheaper. Memory measured $0.742 a session where the one-slice-a-session estimate said $0.3–0.5, because every call's slice stays in the conversation and is read back by the calls after it. The switch prices that.
- **Memory v2's two passes** (`extraction.one_pass: false`). On two calibration principals (24 days, 20 stated facts and rules):
  - one pass held 17 and 18 of 20, against two passes' 19 and 18;
  - one pass wrote probe questions as tasks;
  - two passes held 94 of 100 over five runs.

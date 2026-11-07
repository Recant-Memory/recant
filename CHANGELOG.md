# Changelog

What changed in each version, and the measurements behind the numbers the README, the defaults and the code
comments use. Each measurement says what it was run on; none of it is a promise about your traffic. Measure your own
with `recant bench` and `recant eval memory`.

## 0.4.0

The first step made short: wrap the client, run the app, `recant cost`. Plus fixes from a third outside developer's
run of the whole kit on 0.3.3 (November 2026), and a stricter scorer.

- **No configuration.** `recant.wrap(client)` needs no store and no tenant: the store is `$RECANT_HOME` or `./.recant/`
  in the working directory, made when the client is wrapped, with a `.gitignore` that ignores it all, and the tenant is
  "default". Every command finds the same store with no flags (`--store`/`--tenant` still override).
- **The report leads with three lines:** what you spent, on how many calls over what period; what Recant can save of
  it with no change to what the model sees, in dollars on the calls it measured (a weekly rate only once the traffic
  spans a day); and what that rests on. Below about 20 calls it says it is too early and sizes nothing. Calls that
  failed count in what was spent, for the usage they reported, and models with no price are named, never dropped; a
  saving is sized on its job's bill since the job's settings last changed. `recant cost --json` carries the lines as
  `summary`.
- **`recant apply`** turns on the safe savings that are ready (the nightly pass's own rule: a lever that changes only
  cache markers, at least 20 runs of its job); `--dry-run` shows what each would change in a request and changes
  nothing; a lever not yet ready says why. A call made outside any `client.run` counts as a run of its own here: the
  wrapper keeps one run open per client for the life of the process, so a service that never opens a run would have
  waited for ever.
- **Calls outside any `client.run` are threaded by conversation.** With only `recant.wrap`, every call of a process
  lands in one run, and with several conversations in flight their calls interleave: a developer's wrap-only app (40
  tasks, four at a time) had 3 of 64 calls seen to extend the one before, and was told there was no safe saving. Each
  such call is now given the conversation it continues, found by content (the longest earlier request whose messages
  are a prefix of its own, as the cache itself finds one; a request that branches off an earlier point of a
  conversation starts one of its own, so no reuse is claimed that the cache may not give), and that run closes when
  the process exits, so the coverage watch writes too (it had read "the watch was not on").
- **A breakpoint where it pays.** `breakpoint.add` is sized per job and applied only to the jobs where its reads
  outweigh the write premium: a job of one call per run (a writer handed a fresh brief each time) pays to write and
  never reads, and is now left out and sent with no marker. The lever's line gives each job's part; its share, its
  20-run readiness and its later check are all of the jobs it is applied to.
- **Under a day of traffic,** the report's header says how long the traffic ran inside the window, and the levers
  table gives dollars as measured, not "/wk" figures scaled from minutes. The latency line names calls over 30 s,
  which a p95 can hide (four 62-second calls in 84, each the SDK retrying inside the call).
- **The cache minimum.** A prefix shorter than the model's minimum (512 tokens on the 5.x models, up to 4,096 on
  older ones; `recant.costshape.CACHE_MIN_TOKENS`, from the vendor's documentation read 2026-11-06; an unlisted model
  is taken at 4,096) is sent with its marker and silently not cached. `breakpoint.add` is no longer sized on such
  calls: before, a job of short prompts was offered a saving that could not come.
- **`recant doctor`** checks the version, the SDK, the store, the switch, metering, prices and coverage, one line each
  with what to do about each problem (calls to another provider are named, not a fault: the wrapper is Anthropic's
  only); it makes no directory and writes no friction log (on an existing store, its
  writable check creates one new temporary file and removes it), and exits 1 while anything needs fixing.
- **`--json` on every command,** each output naming its schema (`"schema": "recant-cost/1"`, ...; within a version,
  fields are added, never renamed or removed). `recant feedback --json` prints the bundle and writes nothing; writing
  it still asks first, piped answers work, and a closed stdin reads as blank answers and writes nothing (it had been
  reported as a fault in Recant). Progress goes to stderr.
- **Errors are one line** on stderr saying what to change, usage mistakes included: exit 1 or 2 for a problem with
  the input or the setup, exit 3 for a fault in Recant itself, judged by where it was raised (`RECANT_DEBUG=1` shows
  the traceback). A hook (`--run`, `--grade`, `--client`,
  `--load`) that is not found says which, and how its module is looked up.
- **For coding agents:** `AGENTS.md` (the recipe, and how to run the tests), a Claude Code skill in `skills/recant/`,
  `llms.txt` opening with the recipe, and the recipe at the foot of `recant --help`. The README opens with it, and
  PyPI now shows the README.
- **The scorer counts single letters and digits** (`recant.notes.parts`/`holds`, used by `recant eval memory`'s
  words-only check and by the notes evidence rule). Before, every word under two characters was dropped, so a reply
  of "B minus" scored right against the key "A minus", and "3" against "5". Now every single letter and digit counts
  as a whole word ("b" is not found inside "but", nor "5" inside "15"); only a lowercase "a" or "i", as written, is
  left out as an article or a pronoun, and an uppercase single letter always counts. A value must also pass the rule
  before 0.4.0, so the check is stricter by construction ("X Y" is still not found in "Y X", nor "a 5" in "5"), and a
  test holds it to the old rule over a grid of values and texts. The notes evidence check, which uses it, can only
  get stricter. Rescored with the new rule, the 1,020 saved answers of the memory measurement on synthetic
  principals' sessions (0.2.x, below; 255 questions, four arms) change no verdict: words only 20, 220, 220, 146 and value match 20, 255, 255, 197 of 255
  (none, history, notes, memory v2), as before; 28 of its keys hold a single character, most of them absent
  questions' "I don't know", which their own check scores.
- From the reruns on the fixed candidate: `recant apply` names the jobs a lever is applied to and those it leaves
  off, and why; the coverage line says what the coverage watch is; the output-cap line says when thinking filled the
  output; `recant eval memory --estimate` prints progress; the bench report's setup section is a list, not raw JSON;
  `recant feedback --help` lists the six questions; and the first screen gives `client.run` its `session=`.
- From the developer's run:
  - a suggested shorter key drops the question's words from its start only (trimming the end had left "EUR 50,000
    excluding");
  - the bench's Quality section prints the verdict's own quality line, with the interval under "for reference";
  - "failed tasks" are now "runs that raised", and a task that ran and scored 0 is said to count in the pass rate;
  - a lever's prediction line says it is the saving over the arm's paired tasks;
  - the eval's total spend counts its own estimate and the question drafting under the same `--out`, itemised;
  - `recant coverage --usage` gives no cost share when unpriced traffic went unseen, and says how to count it;
  - the feedback form says its minutes count from the first `recant` command;
  - the README says how to check the pass-through's bytes yourself, and that Claude Haiku 5.5 thinks by default (a
    job on it with no `thinking` parameter shows a reasoning share, and thinking counts against `max_tokens`).

## 0.3.3

From a second outside developer's run of the whole kit on 0.3.2, on the same kind of app (November 2026). Each item is
something they hit; two are 0.3.2 fixes that did not take.

- **`recant coverage --usage`** read a file in the Anthropic SDK's own field names (`input_tokens`, …) as "no tokens
  reported", and a time with a `+01:00` offset just after local midnight as "Recant saw 0%": the file's lines were
  summed by their local day, Recant's records by UTC's. It now reads the SDK's names, refuses a file with no token
  count, and puts every line on its UTC day.
- **`recant bench --client`** called the function with an argument nothing documented, and a function taking none
  crashed with a traceback. It may take none or `is_async`, and a wrong one is a one-line error.
- **A safe arm's quality rests on equivalence.** Its levers change only cache markers; with the input identical on
  every call, the verdict says so and gives the pass rate as context. One task in 30 pairs that failed in the safe arm
  had read "quality NOT held". An arm with an opted-in lever is still held to the interval.
- **Drafted keys** (0.3.2's prompt change did not take: the drafter still wrote a key with a word the answer need not
  repeat, and the developer trimmed 6 of 26 by hand): `--make-questions` now lists each key longer than two words, or
  holding words its question already says, with a suggested shorter key to take or leave; it changes no key by itself,
  since a word the question says, or a leading "A", can be part of the value. A drafted question whose quote is itself a question is dropped, and
  absent questions are held to a third of a scope's kept questions, rounded up.
- **Who is asking.** A session line may carry `name` (default: the scope after its colon), and every arm's answering
  prompt says who is asking; memory v2 had held a person's facts without knowing the asker was that person.
- **`recant eval memory --estimate`** runs every arm on the first scope only and prints the whole run's spend. With
  fewer than 10 scopes, the verdict says its intervals (which resample whole scopes) mean little; over 3 they had read
  100% to 100%.
- **Weekly figures** (0.3.2's header did not explain them): the levers header gives the window's own scale factor, and
  a share names the jobs it is of.
- **Progress** reaches a log file or CI line by line (it was block-buffered when stdout was not a terminal).
- **`results.jsonl`** keeps the start of each task's `repr(result)`, so a failed task can be looked at; it stays local.
- **The feedback bundle** numbers two jobs' levers (`effort.<job 1>`, `effort.<job 2>`) instead of writing both as
  `effort.<job>`, and the terminal says where the allow-list left strings out.
- **A relative `store=`** is fixed when you wrap, so a later change of directory does not move it.
- **`recant --version` from PyPI** names the commit the release was built from (`release build`; deferred from 0.3.2).
- The README now says what `session=` is for, what a line of `sessions.jsonl` is, how a run carries into
  `asyncio.to_thread`, and that an `--estimate` directory resumes.

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

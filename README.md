# Recant Memory

**See what your Anthropic calls cost, and what you'd save, on your normal traffic. One line to add.**

```
pip install "recant[anthropic]"
```

```python
import anthropic, recant

client = recant.wrap(anthropic.Anthropic())        # or anthropic.AsyncAnthropic(); wrap each client your app builds
# ...run your app as you normally do: sync, async, streaming and tool loops all work; nothing else changes...
```

```
recant cost      # what you spent, and what Recant would save with no change to what the model sees
recant apply     # turn those savings on (recant apply --dry-run shows each change first)
recant cost      # after 50 runs or 7 days: what you actually saved, against the prediction
```

- **No configuration.** The store is `./.recant/` in the directory your app runs from (or `$RECANT_HOME`), with its
  own `.gitignore`; run the `recant` commands from the same directory. It holds counts, token figures, timings and
  hashes of what was sent, never the text of a prompt, a reply or a tool result. Nothing leaves your machine.
- **Two or more agents?** Optional: `with client.run("researcher", session=conversation_id):` around one agent's
  calls and `client.run("writer", session=conversation_id)` around another's, on whichever wrapped client each uses
  (`async with` on an async one), and the report sizes each apart. `session` is the user's conversation: the gaps
  between its runs size the cache-lifetime levers.
- **Safe by construction.** `recant apply` changes only cache markers: the model is sent the same input. `recant off`
  makes the wrapper a pure pass-through again.
- **Honest from the first report.** After about 20 calls it says what you spent and what it can save, in dollars on
  the calls it measured; with fewer it says it is too early. A model with no price is named, never dropped.
- **For coding agents.** `recant doctor` checks the setup in one go; every command takes `--json`. `AGENTS.md` has the
  recipe, and `skills/recant/` a Claude Code skill (install below).

When you want proof beyond the safe savings, `recant bench` compares your app with and without Recant on your own
tasks, and `recant eval memory` measures memory on your own sessions (both below).

### Using it from a coding agent

Ask Claude Code or Codex to "add Recant (pip package `recant`) to this app and show me what it would save". To give
Claude Code the recipe as a skill, from your project:

```
mkdir -p .claude/skills/recant
curl -fsSL https://raw.githubusercontent.com/recant-memory/recant/main/skills/recant/SKILL.md -o .claude/skills/recant/SKILL.md
```

For Codex, or any agent that reads `AGENTS.md`, copy the first section of Recant's `AGENTS.md` ("Adding Recant to an
app") into your project's `AGENTS.md`. `llms.txt` lists the whole surface in one file.

## What Recant is

Recant makes an agent's context cost less, for the same or better answers. It has three parts, and you can use any
one of them alone:

1. **A cost saver.** Recant meters every model call your agent makes and finds what it is paying for that it need
   not: requests that could be read from the cache, cache lifetimes, output caps, effort, tool output, loops. It
   prices each lever in dollars on your own traffic and applies the safe ones on `recant apply` — those that change
   only cache markers, never what the model sees — then checks each one's realized saving against its prediction and
   reverts one that loses money or trips its guard (one that saves less than predicted stays on, its check saying so).
   Everything else is reported with its price and left to you.
2. **Managed memory.** By default, notes: the agent writes keyed, dated facts with one tool; Recant checks each
   against the conversation it came from, keeps the history, and renders the notes into the cached prefix. The
   fallback, for a scope that outgrows notes, is extracted memory: a store with a review lifecycle, a task ledger,
   extraction from your sessions, and a budget fitter.
3. **The measurement.** The same log sizes the levers, checks them after they are applied, and holds the eval
   harness that says whether the memory beats grep on your data.

Recant is a library, not a service or an agent framework: your agent loop, your model client and your keys stay
yours. It calls nothing on its own and enforces no spend limit.

## Install

From PyPI:

```
pip install "recant[anthropic]"
```

Or from GitHub, pinned to a tag:

```
pip install "recant[anthropic] @ git+https://github.com/recant-memory/recant@v0.4.2"
```

`recant --version` prints the version and the commit it was built from: `release build` from PyPI (0.3.3 on),
`install record` from GitHub, `source checkout` from a clone. Every report carries both.
`[anthropic]` adds the Anthropic SDK, which `recant.wrap` and the reference client use; leave it off if your agent
uses its own client. From a source checkout: `pip install "./recant[anthropic]"`, and `pip install "./recant[test]"`
then `pytest recant` runs the tests. Python 3.11+. Dependencies: pydantic, pyyaml, python-ulid.

## What leaves your host

Nothing. Recant writes files under the store directory you give it and sends nothing anywhere: your model calls go
from your client as before. The log keeps counts, token figures, timings and hashes of what was sent — never the text
of a prompt, a reply or a tool result. Your notes and memories are markdown files in the same store.

## Quickstart 1, in detail: wrapping your client

```python
import anthropic, recant

client = recant.wrap(anthropic.Anthropic())            # store and tenant optional: store="dir", tenant="name"
with client.run("agent", session="chat-1"):            # optional: name the agent and the conversation
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=300, system=SYSTEM, messages=msgs)
```

- **Two or more agents, or clients.** Wrap each client you have, `Anthropic` and `AsyncAnthropic` alike; they share
  the default store. Put `with client.run("researcher", session=conversation_id):` around one agent's calls and
  `client.run("writer", ...)` around another's, and `recant cost` reports and sizes each apart. A call outside any run
  is wrap's own job (`recant.wrap(..., job=...)`, default "agent").
- **`session`** is the conversation a run belongs to. The advisor reads the gaps between runs of one session to size
  the cache-lifetime and hand-off levers, so give it the user's conversation id (the same for each agent's run in it),
  not a fresh id per task.
- **Threads and asyncio.** A run is a context (a `ContextVar`): `asyncio.to_thread` carries it into the thread, and a
  run opened inside a thread works as well; concurrent runs are metered apart.
- **Tool loops** can put the response's content blocks back in `messages` as the SDK returns them
  (`{"role": "assistant", "content": r.content}`); the wrapper sends them as the SDK would. A request holding anything
  but plain dicts, lists, scalars and the SDK's own models (an iterator, a tuple, a set, a file, an omission sentinel)
  is sent as your own objects, untouched, with no lever applied to it.
- **Recant never breaks a call.** If Recant's own code fails on a call (preparing it, reading its response or stream,
  opening or closing its run, reading the switch), the call goes out as you built it, you get the response, and the
  failure is counted in the friction log; the coverage watch never fails an HTTP request. `recant.wrap(...)` itself
  raises on a store it cannot create.
- **The store** is fixed when you wrap: a relative path is resolved then.
- **Settings.** A lever is priced from its job's last change of settings: the model, effort, output cap, tools, or the
  system prompt's fixed part. Prompts that share at least the model's cacheable minimum from the start are one fixed
  part, judged by what they all share, so a per-request name or brief after a fixed opening is content, not a change;
  a prompt that opens in two or more conversations is a fixed prompt, so replacing it, or adding to or cutting it
  where every prompt of the job is fixed, is a change. A prompt that opens in one conversation only and shares no
  cacheable start with another is content, and never a change. The report says where the rule cannot see: a job whose
  prompts are read as content, and a job whose prompts carry a per-request part after a fixed part (a change at the
  end of that fixed part may not be detected); for both, the realized-against-predicted check is the guard. A job
  whose per-request part comes first is also named: no conversation can reuse another's prompt, and moving that part
  below the fixed instructions is a saving the report sizes, across conversations, when all of them share an ending
  long enough to cache.
- **Checking the pass-through yourself.** Hash each request body as the SDK sends it, run once with `recant off` and
  once with `recant on` before `recant apply`: the digests match, and match the unwrapped client's. After `recant
  apply` the only difference is `cache_control` fields.

  ```python
  import hashlib, anthropic, recant
  def show(request):
      print(hashlib.sha256(request.content).hexdigest()[:16], request.url.path)
  client = recant.wrap(anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(event_hooks={"request": [show]})))
  ```
- **Thinking you didn't ask for.** Some models think by default: Claude Haiku 5.5 uses adaptive thinking unless the
  request sends `thinking: {"type": "disabled"}`, so a job on it with no `thinking` parameter shows a reasoning share
  in `recant cost`, and its thinking counts against `max_tokens` (a low cap can leave no room for the text).

Then, after a few dozen calls:

```
recant cost                      # spent, can save, measured on; the bill split; the levers, each priced in dollars
recant apply --dry-run           # what each safe lever would change in a request
recant apply                     # turn them on
recant cost --nightly            # schedule it: apply what newly pays, check what was applied, revert what loses money
```

Schedule the last line nightly. An applied lever is carried out on the wrapper's next `create`; the request it sends
differs from yours only in cache markers, and each call is checked for that (a request that would differ in anything
else is sent as you built it). After 50 runs or seven days the nightly pass prices the same calls both ways and
records the lever's realized saving against its prediction (plain `recant cost` shows the same check, unrecorded);
one that loses money or trips its guard is reverted, and one that saves less than predicted stays on. `recant cost
--revert breakpoint.add` reverts one by hand; `--all` lists levers that save nothing in the window.
`examples/cost_quickstart.py` is this, runnable: a support agent with no caching set up, and the command sequence.

The wrapper takes `anthropic.Anthropic` or `anthropic.AsyncAnthropic` and covers `messages.create`, `create(stream=True)`,
the `messages.stream(...)` helper and `messages.count_tokens`. A streamed call is recorded from its final message's
usage when the stream ends (or is closed); thinking tokens are recorded, so the report shows thinking's share of the
output. On the async client `async with client.run(...)` works too, and each run keeps its own log, so tasks in flight
at once are metered apart. `examples/cost_quickstart_async.py` is the same agent, async and streamed:

```python
client = recant.wrap(anthropic.AsyncAnthropic())
async with client.run("agent", session="task-1"):
    async with client.messages.stream(model="claude-sonnet-5-5", max_tokens=1024, system=SYSTEM, messages=msgs) as s:
        text = await s.get_final_text()
```

**Timeouts and retries are yours, not Recant's.** Recant never times out or retries a call. The SDK's default read
timeout is 600 seconds, so it does not cut a stream that goes silent mid-body, and a stream that breaks mid-body is not
retried (the SDK retries a request only until its response starts). Such silences happen: on one host's network path
in October 2026 (Sonnet 5.5, five streams in flight), 26 of 163 streamed calls went silent for 30 seconds or more after
their first event, and 6 were closed by the upstream after about 480 seconds of silence — with the raw SDK client and no
Recant in the path (11 of about 470 with Recant; 0 of 480 non-streamed calls; `CHANGELOG.md`, 0.2.x). Set your own idle timeout — the examples
use `timeout=anthropic.Timeout(120.0, connect=10.0)` — and decide what a failed call means for your task. Recant
records a broken stream with its error and the usage it had reported, so the cost report still counts it.

With another client, or your own loop around the SDK, meter each call yourself and carry the safe levers out with one
function:

```python
from recant import StoreLayout, costshape, levers
from recant.metering import Meter

layout = StoreLayout("store"); layout.ensure("demo")
levers.declare_host(layout, "demo")                     # once: this host carries out the safe levers
meter = Meter(layout, "demo")

meter.run("agent")                                      # one run per task or conversation
params = levers.apply(my_params, costshape.overlay(layout, "demo"))      # cache markers where the advisor put them
r = my_client.create(**params)
meter.record(job="agent", model=params["model"], session="task-1", system=my_params.get("system"),
             messages=my_params["messages"], tools=my_params.get("tools"), max_tokens=params["max_tokens"],
             usage={"input": r.usage.input_tokens, "output": r.usage.output_tokens,
                    "cache_read": r.usage.cache_read_input_tokens, "cache_write": r.usage.cache_creation_input_tokens})
meter.close()
```

`levers.apply` takes and returns an Anthropic-shaped request; other providers' shapes wait until a host needs them.

**Your own prices.** Prices default to the library's dated list prices (the vendor's page, read 2026-11-05):
Fable 5.1, Mythos 5.1, Fable 5, Opus 5.5, Opus 5, Opus 4.8, 4.7 and 4.6, Sonnet 5.5, Sonnet 5, Sonnet 4.6, and Haiku
5.5 (its two cards) and 4.5, by model ID (`recant.costshape.LIST_PRICES`); `recant cost` warns when the list is 90 days
old. A model with no
price is reported in tokens only, and `recant bench` names any it saw in the first task before it runs the rest. `--prices
prices.json` takes your own table, row by row over the list: one JSON object keyed by model ID, each row in USD per
million tokens, every field but `input` and `output` optional:

```json
{"my-model": {"input": 1.0, "output": 5.0, "cache_read": 0.10, "cache_write": 1.25, "cache_write_long": 2.0, "batch_factor": 0.5},
 "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125,
                      "above": 100000, "over": {"input": 0.50, "output": 2.50, "cache_read": 0.05, "cache_write": 0.625}}}
```

`cache_read` and `cache_write` default to the input price (no discount assumed), `cache_write_long` (the 1-hour write)
to `cache_write`, `batch_factor` to 1. `above` and `over` are a second card for a request whose prompt (uncached,
cache reads and cache writes together) is over `above` tokens. `recant cost --json` prints each priced model's row in
this format.

## What Recant can't see

Recant never presents a result as covering your traffic when part of it went past Recant. Every report — `recant cost`,
`recant bench`, `recant eval memory` — has a Coverage line, such as (an illustration, not a measurement): "Recant saw 96% of model
calls (unseen: 41 calls to openrouter.ai, model glm-latest)".

- **In your process.** `recant.wrap` turns on a watch (`recant.coverage.watch()`; `wrap(..., watch=False)` leaves it
  off, and `recant off` stops it counting). It counts requests to known model-API hosts that did not go through Recant:
  Anthropic, OpenAI, OpenRouter, Gemini and Vertex, Bedrock, Azure OpenAI, Mistral, Groq, Together, Fireworks, DeepSeek,
  xAI and Z.ai (the list is `recant/known_hosts.py`, one line a host). It keeps counts only: the host, the provider, the
  model the request names, how many requests, and the usage numbers a non-streamed response reports. `recant cost`
  names them on its Coverage line ("unseen: 3 calls to api.openai.com, model gpt-4o-mini"). A POST to any
  other host whose body looks like a model call is counted as `other`, with the domain hashed. It hooks the HTTP library
  the SDKs share (httpx, or httpx2), so it does not see a client built on another one (boto3's Bedrock client,
  `requests`), nor subprocesses such as Claude Code.
- **Against your own totals.** `recant coverage --usage runs.jsonl` matches your own
  per-run log (JSON lines: `ts`, an ISO time, with its offset or taken as UTC; `model`; `tokens_in`, the uncached input,
  cache reads and writes apart; `tokens_out`; `cache_read`; `cache_write`, or the Anthropic SDK's own names for the
  four, `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`; optional `billed`:
  `api_key` or `subscription`; other fields are ignored; a file with no token count is refused) against what Recant metered on the same days,
  by model and provider: tokens and list-priced cost reported, and the share Recant saw. A line can be a call, a task or
  a day: a usage file has no request ids, so lines are not matched one by one, only summed by UTC day and model. `recant.billing` does the same against a provider's own usage report when you supply a source for it
  (`BillingSource`), and `recant cost` then lists traffic billed but never metered.
- **In a bench.** Below 90% of calls or of list-priced cost, the verdict is qualified ("this result covers only <share> of your
  traffic; the rest went to <host>"). Below 50%, it gives no overall saving, only the covered part's.

## Claude Code and the Claude Agent SDK

These tools make their own model calls, so `recant.wrap` does not see them. Each reports its own totals at the end of
a run, per model, and Recant meters those:

```python
from recant.metering import Meter, from_claude_code, from_agent_sdk

meter = Meter(layout, "demo")
from_claude_code(result_json, meter, billing="subscription")   # `claude -p ... --output-format json`
from_agent_sdk(result_message, meter, billing="api_key", messages=assistant_messages)   # the SDK's ResultMessage
```

The totals are the tool's own figures (its cost is an estimate at list price), and the call count comes from the
distinct message ids when you pass the run's assistant messages. The per-step usage on those messages repeats the
input and cache counts, but its output count is a placeholder, so Recant takes the run's totals. The report shows this
traffic in its own rows, billed (0 on a subscription) and priced at list. On a subscription the point is tokens against
the plan's limits, not dollars. No lever applies: these tools manage their own requests and caching. `recant bench` can
still compare two configurations of such a host from these records. Claude Code's OpenTelemetry export
(`claude_code.api_request` events, per request) carries the same counts; Recant has no receiver for it.

## Quickstart 2: managed notes (the default memory)

One notes block per scope — the user's, a team's, the org's — that the agent writes itself, as keyed, dated entries,
and that Recant checks, keeps the history of, and renders into the cached prefix. Offer the two tools, hand their
calls to Recant, and put the block in the system prompt at a session start:

```python
from recant import Store, StoreLayout, episodes, notes
from recant.machinery import Machinery

layout = StoreLayout("store"); layout.ensure("demo")
Machinery.seed_defaults(layout); m = Machinery.load(layout)
store = Store(layout, list(m.kernel.corpus.allowlist) + list(episodes.HOT_PATTERNS))

# a session start: the notes block for the prefix (list prices and the client's own token counter by default)
block = notes.session_start(layout, "demo", store, principal_id="me", serve=m.kernel.serve, today="2026-01-06",
                            block="user:me", model="claude-sonnet-5-5", client=client)
system = MY_CHARACTER + "\n\n" + notes.INSTRUCTION + ("\n\n" + block["text"] if block else "")
session = episodes.open_session(layout, "demo", "2026-01-06")      # append each turn: the evidence the check reads
session.append("user", said, who="me")

# each turn: offer notes.TOOL_SPECS; a call to one of them goes to Recant, its answer back to the model
r = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, system=system, tools=notes.TOOL_SPECS, messages=msgs)
for call in (b for b in r.content if b.type == "tool_use"):
    out = notes.handle(call.name, call.input, layout=layout, tenant="demo", today="2026-01-06", principal_id="me",
                       lines=episodes.read_session(layout, "demo", session.episode_path), episode_path=session.episode_path)
```

`examples/notes_agent.py` is a complete agent on the raw SDK: two sessions, facts noted in the first and answered from
the prefix in the second (`--offline` runs it without a key). `session_start` returns None when the scope has moved to
memory (below); serve the memory then.

The evidence rule, in code: a principal note's value must appear in the principal's own words in the session (a quote
is optional); a tool note names a call made in the session, and where the host keeps a bounded part of tool output in
the record (`notes.record_output`) the value must be in it. An absence is never a note. Two writers to one key within two
hours, with different values, are served as unresolved until a later write settles it. Shared blocks render
byte-identically across conversations, so their cache is shared; the host says who may write them.

A scope whose notes outgrow the threshold (`serve.notes_default_tokens`, 95,000 counted tokens until the host has
measured its own prices) migrates into extracted memory with no model call, one way, and is served from there; a host
whose agent cannot keep notes sets `serve.mode: normal`.

## Quickstart 3: extracted memory (the fallback)

Two configuration files describe the machinery: the **artifact** (what a learning loop may tune: markers, prompts,
models per job, the trust table) and the **kernel** (what nothing automated may touch: thresholds, the corpus
allowlist, the serve budget). Recant ships working defaults (`recant/defaults/`): host-neutral extraction and nomination
prompts, real model IDs (Sonnet 5, reasoning off), and the default budgets (a 25,000-token core, a 12,500-token slice).
`Machinery.seed_defaults(layout)` installs them; copy the two files to change any value. The default nomination does not
mark requests for work ("remind me to…"): extraction finds facts, decisions and causes; put tasks in the ledger from your
agent's own tools (`Ledger`), or add a marker for them.

```python
from pathlib import Path
import recant
from recant import Store, Ledger, StoreLayout, episodes, budget
from recant.log import LogWriter
from recant.machinery import Machinery
from recant.store.review import memory_queue, decide_memory
from recant.store.search import load_entries

TENANT = "00000000-0000-0000-0000-000000000001"

# 1. open a store and seed it with Recant's default machinery (or your own two files: Machinery.seed(layout, a, k))
layout = StoreLayout(Path("store")); layout.ensure(TENANT)
Machinery.seed_defaults(layout)
m = Machinery.load(layout)
allow = list(m.kernel.corpus.allowlist) + list(episodes.HOT_PATTERNS)       # memories may anchor in your sessions
store, ledger = Store(layout, allow), Ledger(layout, allow)

# 2. write: your agent appends each turn as it happens; when a session starts and ends is yours
session = episodes.open_session(layout, TENANT, "2026-01-05")
session.append("user", "We decided: the scanner ships on Fridays.", who="me", when="2026-01-05T09:00:00+00:00")
session.append("assistant", "Noted.")

# 3. extract the day with your client: anything with complete(request) -> Completion (see ModelClient below).
#    Schedule it yourself. Proposals go to the review queue, tasks to the ledger; the result says what it used.
from recant.clients.anthropic import AnthropicClient               # the [anthropic] extra; ANTHROPIC_API_KEY
result = recant.extract(layout, TENANT, "2026-01-05", machinery=m, client=AnthropicClient(), store=store, ledger=ledger,
                        principal_id="me")

# 4. review: what is waiting, then a decision — a person or your own reviewer
with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
    for row in memory_queue(store, TENANT, principal_id="me"):
        decide_memory(store, TENANT, log, row["id"], to_state="probation", reason="confirmed",
                      reviewer={"kind": "human", "id": "me"}, reasons=m.artifact.review_reasons)

# 5. fit to a budget: two text blocks, byte-identical for identical store state; you place the cache breakpoints
entries = load_entries(store, TENANT, "as_reviewed", m.artifact.trust_rank)
served = budget.fit(entries, family=None, serve=m.kernel.serve, view="as_reviewed", question="what did we decide about the cache?")
system_prompt = my_character + "\n\n" + served["core_text"] + "\n\n" + served["slice_text"]
```

`examples/memory_v2_agent.py` is this quickstart as a runnable file, with the client as the one thing you replace.
A past corpus can go in instead of live sessions: `episodes.ingest(layout, TENANT, corpus_dir, allowlist)` copies
what the allowlist admits, and `recant.harness.nominate` and `recant.harness.extract.Extractor` run over it.

## Measure it yourself

Three commands measure Recant on your own agent, against not using it, and send us numbers. Each writes `report.md` and
`report.json` into its directory; nothing leaves your host.

**Cost and quality on your own tasks: `recant bench`.**

```
recant bench --tasks tasks.jsonl --run mypkg.agent:run_task --grade mypkg.agent:grade [--n 40] [--calibrate 10]
```

- You write `run_task(task, client) -> result`, making every model call through `client` (pass the same client to
  every agent of a multi-agent host, and name each agent's calls with `client.run("<agent>")`), and optionally
  `grade(task, result) -> bool | float`. Without a grader the report shows cost and latency and says quality was not
  measured. `tasks.jsonl` is one task per line, any JSON.
- `client` is a wrapped `anthropic.AsyncAnthropic` when `run_task` is `async def`, a wrapped `anthropic.Anthropic`
  otherwise (`--client mypkg:make_client` builds your own raw client instead: `make_client()`, or `make_client(is_async)` if it
  takes an argument, returning an `anthropic.Anthropic` or `AsyncAnthropic`). The bench hands one client, so a host
  whose agents use both is benched on one: with an async `run_task`, the sync agent awaits its calls on the async client
  (`await client.messages.create(...)`). The report's header says which (`client_mode`).
- It prices the run from the first task and asks before going on (`--yes` skips the question; `--estimate` stops there,
  and the same `--out` later resumes from it, its task counted as the first calibration task), and prints a line per
  task run as it goes. `--run` and `--grade` name `module:function`; the module is imported with the current directory
  on the path.
  The first `--calibrate` tasks run without Recant and size the levers; the predictions are written to
  `predictions.json` and hashed before any compared task runs, and a changed file fails the report.
- Every other task runs twice, without Recant (arm A: your request as built, plus the arm's cache salt; Recant only reads the
  response, so its own time per call is reported for arm A too) and with it (arm B: `--mode safe`, the safe levers), the order
  alternating task by task. The calibration tasks run as arm C, without Recant, in their own store. Each arm has its own cache salt (a line of
  the same length at the start of the system prompt, so no arm reads another's prompt cache) and its own fresh store.
  A task's pass or fail is your grader's; a task whose `run_task` raised counts as failed in its arm and stays in the
  comparison. A failed call counts its billed cost and is reported in its arm; if your code retries it and the task
  finishes, the task is not failed for it.
- The report: cost per task and per passed task with 95% paired intervals (each arm's cold first task shown apart),
  tokens by class, quality (for a safe arm, whose levers change only cache markers, quality rests on the equivalence
  check, the pass rate shown as context; for an arm with an opted-in lever, held means no observed loss and the 95%
  interval no worse than −3 points), latency per
  call and per task (a task's time includes your own timeouts and retries), time to first token, Recant's own time per
  call, each lever's realized saving against its frozen prediction (met when within a quarter), failures and
  equivalence per arm, and plain verdict lines. What is frozen is each lever's share of the bill; its dollars, and the
  cost per task it predicts, are at the report's prices, so `recant bench --out DIR --report --prices prices.json`
  restates both sides of every prediction at your table. The bench's measured saving is on its own tasks, run its
  own way (paired, in alternating order), so it can differ from what `recant cost` predicts on your everyday traffic.
- `--arm-a safe --arm-b safe` is an A/A run: both arms write caches, each under its own salt, and the difference's
  interval should contain zero. `--arm-a safe --arm-b safe+reasoning.agent` measures one your-call lever against the
  safe set alone. An existing `--out` directory resumes, counting only the attempt that finished each task; a different
  Recant version is refused unless `--allow-mixed`.
- `results.jsonl` in the bench directory keeps each task run's score, error and the first 2,000 characters of
  `repr(result)`, so a task that failed in one arm can be looked at (it stays local: the feedback bundle reads only the
  report).
- Every call `client` makes while a task runs counts toward that task, whatever session, run, thread or asyncio task
  your code puts it in. Both arms run in your process, one after the other: anything `run_task` keeps in memory
  between tasks (a cache of tool results, a loaded index) carries from one arm to the other, so keep such state per
  task, or clear it at the start of `run_task`.

**Memory on your own sessions: `recant eval memory`.**

```
recant eval memory --sessions sessions.jsonl --make-questions          # drafts ./questions.jsonl: read and edit it
recant eval memory --sessions sessions.jsonl --questions questions.jsonl --estimate   # the first scope only: the spend
recant eval memory --sessions sessions.jsonl --questions questions.jsonl
```

- `sessions.jsonl`: one line per session, `{"scope": "user:ana", "ts": "<the session's start, ISO>", "name": "Ana",
  "messages": [...]}`, Anthropic-shaped messages in order; or `--load mypkg.export:load_sessions` returning the same
  list. `scope` is whose memory it is (one scope per person, team or org); `name` (optional, default the scope after its
  colon) is how the person is named, and every arm's answering prompt says who is asking. Put in `messages` what was
  said, not what your app adds to the prompt (an earlier session's facts restated would leak into the later one).
- The drafted questions are recall (a fact stated earlier, quoted from the person's own turn), update (only the newest
  value is right) and absent (nothing says; the right answer is "I don't know"). A question whose quote is not the
  person's own words in the session it names is dropped. `--questions` names the file (default `./questions.jsonl`);
  the drafting calls are metered under `--out` (default `eval/<UTC stamp>/`). The file is hashed when scoring starts.
- Edit the keys before scoring: a recall or update answer is right when its `ANSWER:` line holds every word of the key
  (or, beside it, the same date or amount in another format), so a key is the fewest words that identify the value:
  `Juniper`, not `the Juniper account`. The drafter asks only about the person's own turns, never a tool's result, so
  facts found by tools are not tested here (see Known limits for why that matters to memory v2).
- Four arms answer every question: none, carried history (the ceiling), managed notes, memory v2. The report gives
  accuracy per arm and per kind with 95% intervals, each arm against carried history, cost per question and per correct
  answer with the write path, the total spent, tokens, latency (the answer call's p50 and p95, Recant's own time per
  call), notes size per scope against the 95k switch point, and what the evidence check refused. The verdict leads with
  each arm's words-only score; with fewer than 10 scopes it says the intervals (which resample whole scopes) mean
  little. Scoring is mechanical: the answer's words on the reply's `ANSWER:` line. An absent
  question scores the abstention, the same in every arm that does not invent an answer.

**Tell us: `recant feedback`.** Asks six optional questions (two minutes: minutes from install to your first
report; where you had to read Recant's code or guess; the errors you hit; what you wanted to measure and couldn't;
whether you would leave it on; anything else; piped answers work, one per line), prints the whole bundle and writes
`recant-feedback-<date>.zip` on your yes: the reports' numbers, versions, OS, the features you used, error classes and
counts, the coverage of the store you name (`--store`, `--tenant`) by provider and public model, and your setup
friction (every `recant` command, and any failure of Recant's own code inside a wrapped call, logs locally to
`~/.recant/friction.jsonl`, or `$RECANT_HOME/friction.jsonl`). Every string Recant writes into it passes an
allow-list: no prompts, replies, tool data, notes, questions, keys, paths or names, and no hostname but a known model
API's (an unknown host is hashed). Your answers to the six questions are not filtered: they go in as you typed them,
shown to you with the rest before anything is written. Recant
sends nothing; attach the file to a new issue on the repository.

**One switch: `recant off`.** `recant off --store DIR` (or `RECANT=off`) makes the wrapper a pure pass-through from the
next call: your request byte for byte, nothing written. `recant off --meter` meters and changes nothing on the request;
`recant on` reverses it; `recant status` says which is in force. The switch is a file in the store
(`recant.switch.json`); `RECANT=off`, `meter` or `on` in the environment overrides it for that process.

## Your model client

```python
class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> Completion: ...
```

`request.model` is already resolved; `request.system`, `request.messages`, `request.tools`, `request.max_tokens`,
`request.output_schema` are the library's own types (`recant.provider`). Return `Completion(content, stop_reason,
usage, request_id=None)` with `usage` as input / output / cache_read / cache_write tokens, or raise `ProviderError`.
Optional, found by presence: `count_tokens(request) -> int`; a batch path, `submit(requests) -> handle` and
`poll(handle) -> None | iterable of (custom_id, Completion | ProviderError)`. A job that asks for batch on a client
without one is sent live. `recant.clients.anthropic.AnthropicClient` is the reference implementation.

Dollars: pass `prices={model_id: recant.provider.Prices(...)}` to `recant.extract` (or to `harness.Provider`) and
every logged call carries `cost_usd`; pass nothing and the log carries tokens with `cost_usd: null`. The ceiling, if
you want one, is yours — wrap the client, or build a `harness.Provider` subclass and pass that as `client`; the
library never refuses a call.

## What to log for each lever

`recant cost` sizes a lever only from records that carry what it needs. A host that keeps its own model client logs
through `recant.metering.Meter` (`record` for each model call, `tool` for each tool call); one whose calls go through
Recant's provider gets the per-call fields without doing anything. Run totals alone size only the first row; everything
else is an estimate, an upper bound, or not sized at all.

| lever or finding | from run totals? | the minimum record |
|---|---|---|
| bill split, prices | yes | per run: model, uncached, cache write, cache read, output; billed cost to learn prices |
| prefix and memory section sizes | estimate | per call: the served sections' tokens |
| hand-off against carried history | upper bound | per run: history tokens at its first call, calls, whether the cache was warm |
| cache lifetime | estimate | per run: start, end, session; cache writes split by lifetime |
| batching background work | upper bound | per run: a background flag, and when its output is next needed |
| cache breaks, breakpoints | no | per call: cache read and write, segment hashes, whether the request only appends |
| output caps | no | per call: output tokens, stop reason, the cap |
| reasoning and effort per job | no | per call: reasoning tokens, job, the effort sent |
| tool-output trimming, re-fetch | no | per tool call: bytes fetched and sent, target hashes |
| tool loops | no | per tool call: name, arguments (hashed by Recant), whether it failed |
| fold rule | no | per call: context size; compaction events |
| fewer calls, growth per call | no | per call: context size |
| unmetered spend | no | the provider's usage report (admin access) |

## Formats

Every file Recant writes has a schema in `recant/schemas/`: the memory record, the task record, the eval item, the
log record, the machinery artifact and kernel. One JSON object per log line; markdown with frontmatter per memory.
A session file is lines of `[<session>#L<n>] User|Assistant|Tool: <text>`, a continuation line labelled with two
spaces (`EPISODE_FORMAT_VERSION`, grammar in `recant/store/hot.py`). Since version 2 a tool line whose call itself
failed is written `Tool: ✗ <name>: …` (`append("tool", text, failed=True)`): extraction keeps no memory whose only
evidence is the agent's own failed call, its retries, or the agent's own account of one.

## Known limits

- **Extracted memory keeps no fact found only in tool output.** On one test (120 sessions and 255 questions from six
  synthetic principals; `CHANGELOG.md`, 0.2.x), memory v2 answered none of the 53 questions whose answer came from a tool result;
  managed notes kept all 53, with the call they came from. Use notes for tool-sourced facts.
- **The reasoning lever reaches little.** Turning reasoning off is offered only for a model that can carry it out
  (Sonnet 5.5 takes `between_tools`; Opus 5.5 and Fable 5.1 cannot turn it off). On one τ-bench run (Sonnet 5.5, 150
  pairs; `CHANGELOG.md`, 0.2.x) it cut reasoning from 277 to 23 tokens a task and cost +3.1% a task (95% −0.6% to +6.7%), with quality −1.3
  points (95% −7.3 to +4.7): it stays off for work like that.
- **Notes mode is not in `recant bench` yet.** The wrapper does not carry managed notes, so there is no notes arm.
- **One client per bench.** A host whose agents use a sync and an async client is benched on one of them (above).
- **One provider.** The wrapper and the levers are Anthropic's. A second provider's wrapper is built when coverage
  reports show a host relying on that provider for meaningful traffic: a share of a real user's bill large enough to
  change a bench verdict, not a guess. `recant feedback` carries the coverage counts by provider and public model name
  (hashed hosts stay hashed), and that is how we learn which provider comes next.

## What it is not

Not an agent framework: no loop and no channels; it ships the notes tools' definitions and their handler, and keeps
the session files you append to, but when a session starts or ends is your agent's. Not a vector database: retrieval
is lexical (BM25 over the entries the core left out) until an eval on your data says otherwise. Does not enforce spend
limits: it finds and applies savings, and a ceiling, if you want one, goes around your client. Not a hosted service.

## Changes

`CHANGELOG.md`: what changed in each version, and the measurements behind the numbers here and in the defaults.

## Licence

Apache-2.0. See `LICENSE`.

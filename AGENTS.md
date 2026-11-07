# AGENTS.md: Recant for coding agents

Two cases: you are adding Recant to an app, or you are changing Recant itself.

## Adding Recant to an app

Recant works on any Python code that calls the Anthropic API through the `anthropic` SDK. It meters the calls, says
what they cost and what it would save with no change to what the model sees, and applies those savings (cache
markers only) when asked.

1. Install: `pip install "recant[anthropic]"`, or add `recant[anthropic]` to the project's dependencies.
2. Find every place the app builds an Anthropic client (`grep -rn "Anthropic(" .`) and wrap each one:

   ```python
   import anthropic, recant
   client = recant.wrap(anthropic.Anthropic())          # or recant.wrap(anthropic.AsyncAnthropic())
   ```

   Change nothing else. `messages.create`, `create(stream=True)`, `messages.stream(...)`, `count_tokens` and tool
   loops that put `r.content` back in `messages` all work as before.
3. Optional, when the app has two or more agents: around each agent's calls put
   `with client.run("<agent name>", session=<conversation id>):` (`async with` in async code). Calls outside a run
   are reported under the job "agent".
4. Run the app the way it normally runs. There is nothing to configure: the store is `./.recant/` (or
   `$RECANT_HOME`), made when the client is wrapped, with its own `.gitignore`.
5. `recant cost` (or `recant cost --json`, whose `summary` holds the same three lines). Report its first three lines to
   the user as they are. Below about 20 calls it says it is too early: run more of the app's normal traffic, don't
   invent traffic to inflate the number.
6. `recant apply --dry-run`, then `recant apply` if the user wants the savings on.
7. If anything is off: `recant doctor` names each problem and what to do about it.

Rules:

- **Never change prompts, models, `max_tokens`, effort, thinking or tools to save cost.** Recant changes cache markers
  only. The other levers in the report carry a dollar size and need `recant bench` evidence and the owner's decision.
- Don't read or edit files under `.recant/`; use the commands. Don't commit `.recant/`.
- `recant off` makes the wrapper a pure pass-through (nothing metered, nothing changed); `recant on` reverses it.
- Every command takes `--json` and prints one object with a `schema` field (`recant-cost/1`, `recant-apply/1`,
  `recant-doctor/1`, ...). Errors are one line on stderr; exit 3 means a fault in Recant itself, which is worth
  reporting (`recant feedback` writes a numbers-only bundle; nothing is sent).

`llms.txt` lists the whole surface; `README.md` explains it.

## Changing Recant itself

- Python 3.11+. From the repository root: `pip install -e ".[anthropic,test]"`, then `pytest` runs the suite (no
  network, no key; tests that need an optional package skip themselves).
- `src/recant/wrap.py` is the wrapper; `costshape.py` the report, the levers and the prices; `cli.py` the commands.
- The wrapper must never break the host's call: a failure in Recant's own code sends the call as built and is
  counted in the friction log (`tests/test_wrap_never_breaks_the_host.py`). Keep that true.
- The library sends nothing anywhere and keeps no prompt or reply text in its log; a change that would is out of scope.
- Every user-visible number states what it rests on. Add a test with each change; a test that asserts nothing is
  worse than none.

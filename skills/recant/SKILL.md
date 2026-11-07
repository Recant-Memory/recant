---
name: recant
description: Add Recant to a Python app that calls the Anthropic API, and report what its calls cost and what Recant would save with no change to what the model sees. Use when the user asks to measure, cut or explain the cost of their Anthropic/Claude API calls, or to add Recant.
---

# Recant: what an Anthropic app's calls cost, and what it would save

Works on any Python code that calls the Anthropic API through the `anthropic` SDK.

## Steps

1. Install into the project's environment: `pip install "recant[anthropic]"` (and add `recant[anthropic]` to its
   dependency file).
2. Find each place an Anthropic client is built: `grep -rn "Anthropic(" --include=*.py .`
3. Wrap each one and change nothing else:

   ```python
   import anthropic, recant
   client = recant.wrap(anthropic.Anthropic())          # or recant.wrap(anthropic.AsyncAnthropic())
   ```

   Streaming (`create(stream=True)`, `messages.stream(...)`), `count_tokens` and tool loops work unchanged. For two
   or more agents, optionally put `with client.run("<agent>", session=<conversation id>):` around each one's calls.
4. Run the app the way it normally runs (its own entry point, tests or demo). No configuration: the store is
   `./.recant/`, made when the client is wrapped, with its own `.gitignore`.
5. Run `recant cost --json` and give the user the three lines in `summary` as they are. Below about 20 calls it says
   it is too early; say so and suggest running more of the app's normal traffic.
6. Offer `recant apply --dry-run`, then `recant apply`, which turns on the safe savings (cache markers only).
7. If a step fails, run `recant doctor --json` and act on each check whose `ok` is false (its `fix` says how).

## Rules

- Never change prompts, models, `max_tokens`, effort, thinking or tools to save cost. Report those levers with their
  dollar size and leave them to the user; `recant bench` measures whether quality holds.
- Don't read Recant's source or the files under `.recant/` to get numbers: the commands print them.
- Don't commit `.recant/`.
- `recant off` makes the wrapper a pure pass-through; `recant on` turns it back.

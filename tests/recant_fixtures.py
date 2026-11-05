import textwrap

from ulid import ULID


TENANT = "t1"
ALLOWLIST = ["notes/????-??-??.md", "notes/.hidden/talk/*.txt"]

ARTIFACT_YAML = textwrap.dedent("""\
    format: 3
    version_note: test fixture
    markers:
      - {name: rule_or_decision, description: a rule or decision is stated}
      - {name: root_cause, description: a cause is established}
      - {name: reversal, description: an earlier conclusion is reversed}
    extractor:
      first_pass:   {model: model-large, prompt_version: e1, prompt: "Write the marked moment up as one memory, or say why it is not one."}
      steady_state: {model: model-small, prompt_version: null, prompt: null}
      tasks:        {model: model-large, prompt_version: t1, prompt: "Write the moment up as one task, or say why it is not one."}
    nomination: {model: model-small, prompt_version: n1, prompt: "Find marked moments.\n{markers}"}
    judge:      {model: model-small, prompt_version: j1, prompt: "Score the answer against the reference."}
    reader:     {model: model-medium}
    dedup_threshold: 0.6
    ranking_weights: null
    fitness_weights: null
    replay_sampling: full
    routing: {1: model-small, 2: model-medium, 3: model-large}
    trust_table:
      - [[stated, principal], [directive, principal]]
      - [[observed, principal]]
      - [[stated, agent], [directive, agent]]
      - [[observed, agent]]
      - [[concluded, principal]]
      - [[concluded, agent]]
    contestation_threshold: 1
    review_reasons: [contradiction, staleness, premise_collapse, bad_question, wrong_answer, not_a_memory, confirmed, citation]
    """)

KERNEL_YAML = textwrap.dedent("""\
    format: 4
    gate: {margin_pp: 4.5, floor_pp: 3.0, suite_size: 20}
    noise_share: [0.25, 0.30]
    heldout: {slice: 4, reserve: 4}
    activation: {consolidation_rate: 30, rollback: 20, review_agreement: 59}
    shadow_window_days: null
    exploration_floors: null
    corpus:
      allowlist: ["notes/????-??-??.md", "notes/.hidden/talk/*.txt"]
      full_context: ["notes/????-??-??.md"]
    eval:
      blind_first_sample: 6
      blind_hard_cap: 2
      judge_agreement_floor: 0.9
      noise_draft_ratio: 0.5
      context_lines: 2
      dedupe_question_overlap: 0.5
      unmarked: {share: 0.1, window_lines: 4, oversample: 3}
      drafting: {tier: 2, prompt_version: d1, prompt: Draft a question and a candidate answer.}
      noise:    {tier: 2, prompt_version: x1, prompt: Draft a plausible question the corpus does not answer.}
      seeded:   {tier: 2, prompt_version: s1, prompt: Answer the supplied question from the episodes.}
      dedupe:   {tier: 2, prompt_version: p1, prompt: Group drafts that ask for the same fact.}
      review:   {tier: 3, prompt_version: r1, prompt: Review the drafted item.}
      held_review: {tier: 3, prompt_version: h1, prompt: Does any search hit answer the question.}
      reader: {prompt_version: q1, prompt: Answer from the material or say NOTHING RELEVANT., max_tokens: 200}
      arms: {grep: {calls: 3, lines_per_call: 5, context_lines: 1}, context_cache_ttl: 5m, service: {rounds: 3, entries_per_search: 5, chars_per_search: 4000}}
    tasks: {stale_after_days: 30, stale_days: 3, presumed_dropped_days: 14}
    extraction: {run_dedupe_overlap: 0.7, agent_lines_never_stated: true}
    cost_check: {reps: 3, level: 0.95, resamples: 500, seed: 0, gate: {cost_cut_min: 0.15, cache_hit_drop_max_pp: 1.0}}
    serve:
      memory_budget_tokens: 2000
      chars_per_token: 4.0
      core_tokens: 1200
      slice_tokens: 800
      scopes:
        - {name: ops, match: [outage, cache], channels: ["#ops"]}
        - {name: product, match: [vendor, plan, dashboard], channels: []}
        - {name: manner, match: [manner], channels: [], global: true}
    """)


def make_record(**over):
    rec = {
        "id": str(ULID()),
        "claim": "A signed key file is the standard login method for the scanner.",
        "kind": "stated",
        "actor": {"class": "principal", "id": "owner"},
        "anchor": {"quote": "use the key file, always", "episode": "notes/2026-01-02.md#L3-L4",
                   "channel": None, "message_id": None, "timestamp": "2026-01-02"},
        "scope": "scanner",
        "supersession_condition": "Until the vendor ships token auth.",
        "derived_from": ["notes/.hidden/talk/2026-01-02.txt#L10-L12"],
        "status": {"state": "proposed", "reason": None, "failure_class": None, "superseded_by": None},
    }
    rec.update(over)
    return rec

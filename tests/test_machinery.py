import pytest
import yaml

from recant.machinery import Machinery, MachineryError
from recant_fixtures import ARTIFACT_YAML, KERNEL_YAML

# The keys a loop may diff, and the keys nothing automated may touch.
TUNABLE = {"format", "version_note", "markers", "extractor", "nomination", "judge", "reader", "dedup_threshold",
           "ranking_weights", "fitness_weights", "replay_sampling", "routing", "trust_table",
           "contestation_threshold", "review_reasons"}
PROTECTED = {"format", "eval", "gate", "noise_share", "heldout", "activation", "shadow_window_days",
             "exploration_floors", "corpus", "tasks", "serve", "extraction", "cost_check"}      # prices, ceilings, runtime: the consumer's (format 4)


def write(tmp_path, artifact=ARTIFACT_YAML, kernel=KERNEL_YAML):
    a, k = tmp_path / "a.yaml", tmp_path / "k.yaml"
    a.write_text(artifact if isinstance(artifact, str) else yaml.safe_dump(artifact))
    k.write_text(kernel if isinstance(kernel, str) else yaml.safe_dump(kernel))
    return a, k


def test_the_two_files_hold_exactly_their_own_keys(machinery):
    assert set(type(machinery.artifact).model_fields) == TUNABLE
    assert set(type(machinery.kernel).model_fields) == PROTECTED
    assert TUNABLE & PROTECTED == {"format"}


@pytest.mark.parametrize("key", sorted(PROTECTED - {"format"}))
def test_a_protected_key_in_the_artifact_fails_to_load(tmp_path, key):
    art, ker = yaml.safe_load(ARTIFACT_YAML), yaml.safe_load(KERNEL_YAML)
    art[key] = ker[key]
    with pytest.raises(MachineryError, match=key):
        Machinery.from_files(*write(tmp_path, artifact=art))


@pytest.mark.parametrize("key", ["markers", "routing", "trust_table", "judge"])
def test_a_tunable_key_in_the_kernel_fails_to_load(tmp_path, key):
    art, ker = yaml.safe_load(ARTIFACT_YAML), yaml.safe_load(KERNEL_YAML)
    ker[key] = art[key]
    with pytest.raises(MachineryError, match=key):
        Machinery.from_files(*write(tmp_path, kernel=ker))


@pytest.mark.parametrize("key", sorted((TUNABLE | PROTECTED) - {"format"}))
def test_no_key_may_be_missing(tmp_path, key):
    art, ker = yaml.safe_load(ARTIFACT_YAML), yaml.safe_load(KERNEL_YAML)
    (art if key in art else ker).pop(key)
    with pytest.raises(MachineryError, match=key):
        Machinery.from_files(*write(tmp_path, artifact=art, kernel=ker))


def test_trust_is_a_lookup_over_every_pair(machinery):
    a = machinery.artifact
    assert [a.trust_rank(k, c) for k, c in [("stated", "principal"), ("directive", "principal"),
            ("observed", "principal"), ("stated", "agent"), ("directive", "agent"), ("observed", "agent"),
            ("concluded", "principal"), ("concluded", "agent")]] == [1, 1, 2, 3, 3, 4, 5, 6]
    assert a.trust_rank("observed", "agent") < a.trust_rank("concluded", "principal")


def test_a_trust_table_missing_a_pair_or_repeating_one_is_refused(tmp_path):
    art = yaml.safe_load(ARTIFACT_YAML)
    art["trust_table"][2] = [["stated", "agent"]]  # the agent directive has no rank
    with pytest.raises(MachineryError, match="trust_table"):
        Machinery.from_files(*write(tmp_path, artifact=art))
    art = yaml.safe_load(ARTIFACT_YAML)
    art["trust_table"].append([["stated", "agent"]])
    with pytest.raises(MachineryError, match="trust_table"):
        Machinery.from_files(*write(tmp_path, artifact=art))


def test_routing_is_tiers_one_to_three_only(tmp_path):
    art = yaml.safe_load(ARTIFACT_YAML)
    art["routing"][0] = "embedding-model"
    with pytest.raises(MachineryError, match="routing"):
        Machinery.from_files(*write(tmp_path, artifact=art))


def test_seeding_never_overwrites_the_store(layout, machinery, tmp_path):
    art = yaml.safe_load(ARTIFACT_YAML)
    art["version_note"] = "a newer seed"
    a, k = write(tmp_path, artifact=art)
    assert Machinery.seed(layout, a, k) is False
    assert Machinery.load(layout).artifact.version_note == "test fixture"


def test_missing_files_stop_the_load(layout):
    with pytest.raises(MachineryError, match="missing"):
        Machinery.load(layout)


def test_a_format_one_file_is_refused_with_the_keys_it_lacks(tmp_path):
    art = yaml.safe_load(ARTIFACT_YAML)
    del art["format"]
    art["markers"] = ["rule_or_decision"]
    with pytest.raises(MachineryError, match=r"format 1.*reads format 3"):
        Machinery.from_files(*write(tmp_path, artifact=art))
    ker = yaml.safe_load(KERNEL_YAML)
    del ker["format"], ker["eval"]
    with pytest.raises(MachineryError, match="eval"):
        Machinery.from_files(*write(tmp_path, kernel=ker))
    ker = yaml.safe_load(KERNEL_YAML)
    ker["format"], ker["ceilings_usd"] = 3, {"default": 1.0}             # a format-3 kernel: the keys that moved out are named
    with pytest.raises(MachineryError, match=r"format 3.*reads format 4.*moved out: \['ceilings_usd'\]"):
        Machinery.from_files(*write(tmp_path, kernel=ker))


def test_a_prompt_and_its_version_travel_together(tmp_path):
    art = yaml.safe_load(ARTIFACT_YAML)
    art["judge"]["prompt"] = None
    with pytest.raises(MachineryError, match="together"):
        Machinery.from_files(*write(tmp_path, artifact=art))


def test_the_artifact_hash_covers_the_prompt_text(tmp_path, machinery):
    a, k = write(tmp_path, artifact=ARTIFACT_YAML.replace("Score the answer", "Score the answer strictly"))
    assert Machinery.from_files(a, k).artifact_hash != machinery.artifact_hash


def test_the_drafting_fingerprint_moves_with_what_produces_drafts_and_nothing_else(tmp_path, machinery):
    base = machinery.drafting_fingerprint()
    judge_only = Machinery.from_files(*write(tmp_path, artifact=ARTIFACT_YAML.replace("Score the answer", "Score it")))
    assert judge_only.drafting_fingerprint() == base          # the judge does not draft
    for change in [dict(artifact=ARTIFACT_YAML.replace("a cause is established", "a root cause is reached")),
                   dict(artifact=ARTIFACT_YAML.replace("Find marked moments.", "Find the marked moments.")),
                   dict(kernel=KERNEL_YAML.replace("Draft a question and", "Draft one question and"))]:
        assert Machinery.from_files(*write(tmp_path, **change)).drafting_fingerprint() != base
    review_only = Machinery.from_files(*write(tmp_path, kernel=KERNEL_YAML.replace("Review the drafted item.", "Review it.")))
    assert review_only.drafting_fingerprint() == base


def test_install_replaces_and_keeps_what_it_replaced(layout, machinery, tmp_path):
    a, k = write(tmp_path, kernel=KERNEL_YAML.replace("stale_days: 3", "stale_days: 4"))
    archived = Machinery.install(layout, a, k)
    assert len(archived) == 1 and archived[0].startswith("kernel.yaml.")
    assert (layout.machinery_dir / "history" / archived[0]).read_text() == KERNEL_YAML
    assert Machinery.load(layout).kernel.tasks.stale_days == 4
    assert Machinery.install(layout, a, k) == []

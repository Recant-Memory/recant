"""The one quality standard : held = no observed loss and the 95% interval's lower end no worse than
−3 points; intervals are a paired bootstrap by cluster, a cluster resampled whole."""
from recant import quality as q


def test_held_needs_no_observed_loss_and_the_lower_end_above_the_floor():
    assert q.held(0.0, -0.03) and q.held(0.01, -0.029)
    assert not q.held(-0.001, 0.0)                  # any observed loss fails, however tight
    assert not q.held(0.02, -0.031)


def test_clusters_are_resampled_whole():
    # 20 clusters of 10 items: b wins all of a cluster or loses all of it. Pooled as independent items the interval
    # would be ~sqrt(10) narrower; resampled by cluster it is as wide as 20 coin flips.
    items = [(c, 0, 1 if c % 2 else 0) for c in range(20) for _ in range(10)] + \
            [(c, 1, 0) for c in range(20) if c % 2 == 0 for _ in range(10)]
    res = q.paired_bootstrap(items, n=4000, seed=1)
    assert res["clusters"] == 20 and res["items"] == 300
    lo, hi = res["interval"]
    assert lo < res["delta"] < hi and hi - lo > 0.3


def test_the_verdict_and_the_sample_that_would_confirm_it():
    tight = q.paired_bootstrap([(c, 1, 1) for c in range(30)] + [(0, 0, 1)], n=2000)
    assert q.verdict(tight) == "held" and q.clusters_needed(tight) == 30
    wide = {"delta": 0.006, "interval": (-0.036, 0.049), "items": 100, "clusters": 50}
    assert q.verdict(wide) == "interval too wide"
    assert q.clusters_needed(wide) == 69              # 50 × (0.042 / 0.036)², rounded up
    assert q.verdict({"delta": -0.006, "interval": (-0.05, 0.04), "clusters": 50}) == "loss observed"
    assert q.clusters_needed({"delta": -0.006, "interval": (-0.05, 0.04), "clusters": 50}) is None


def test_levers_that_leave_the_input_unchanged_are_checked_for_equivalence_not_gated():
    assert q.exempt("breakpoint.add") and q.exempt("ttl.prefix") and q.exempt("batch.extract")
    assert not q.exempt("fold.window_only") and not q.exempt("memory.slice_budget") and not q.exempt("handoff.session")
    built = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
    marked = [{"role": "system", "content": [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]},
              {"role": "user", "content": "hi"}]
    assert q.equivalent(built, marked)
    assert not q.equivalent(built, [marked[0], {"role": "user", "content": "hi "}])            # one byte of content
    assert not q.equivalent(built, marked[:1])                                                   # a message dropped
    two = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}]}]
    assert q.equivalent(two, [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert not q.equivalent(two, [{"role": "user", "content": "ab"}])                           # blocks merged: not a marker

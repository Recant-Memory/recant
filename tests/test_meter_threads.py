"""A host's threads sharing one Meter (a multi-agent host on threads): the log a call outside any run
opens is opened once, so no record goes to a log another thread closed."""
import threading

from recant.log import read_run
from recant.metering import Meter
from recant_fixtures import TENANT


def test_threads_recording_at_once_lose_nothing(layout):
    m = Meter(layout, TENANT)
    start = threading.Barrier(16)
    errors = []

    def one(i):
        start.wait()
        try:
            m.record(job="agent", model="claude-sonnet-5-5", usage={"input": 10, "output": 1}, session=f"s{i}",
                     messages=[{"role": "user", "content": str(i)}])
        except Exception as exc:                                     # noqa: BLE001
            errors.append(exc)
    ts = [threading.Thread(target=one, args=(i,)) for i in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    m.close()
    recs = [r for p in layout.log_dir(TENANT).glob("*.jsonl") for r in read_run(p, typed=False) if r["kind"] == "model_call"]
    assert not errors and len(recs) == 16 and len(list(layout.log_dir(TENANT).glob("*.jsonl"))) == 1

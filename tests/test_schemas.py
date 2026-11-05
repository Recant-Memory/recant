import json

from recant import schemas


def test_the_pinned_schema_files_match_the_models():
    built = schemas.build()
    assert set(built) == {"memory-record", "task-record", "eval-item", "log-record", "machinery-artifact", "machinery-kernel"}
    for name, schema in built.items():
        path = schemas.HERE / f"{name}.schema.json"
        assert path.exists(), f"run `python -m recant.schemas` to write {path.name}"
        assert path.read_text() == schemas.render(schema), f"{path.name} is stale; regenerate it"
        assert json.loads(path.read_text())["x-format-version"] >= 1


def test_the_memory_schema_is_closed():
    schema = schemas.build()["memory-record"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"id", "claim", "kind", "actor", "anchor", "scope",
                                         "supersession_condition", "derived_from", "status", "key",
                                         "value_status"}   # key and value_status: optional, additive
    assert set(schema["$defs"]["Actor"]["properties"]) == {"class", "id"}

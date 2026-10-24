from pathlib import Path

import pytest
from curatorkit.config import PipelineConfig
from curatorkit.connectors.jsonl import JSONLReader
from curatorkit.gates.schema import SchemaGate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,expected",
    [("qa_generation", 6), ("multiturn_generation", 6), ("preference_generation", 8)],
)
def test_curator_reads_seed_records_before_generation(name, expected):
    config = PipelineConfig.from_yaml(ROOT / f"configs/curatorkit/{name}.yaml")
    reader = config.readers[0]
    rows, reader_rejected = JSONLReader(
        path=ROOT / reader.path,
        format=reader.format,
        field_mapping=reader.field_mapping,
    ).read()
    assert not reader_rejected
    assert len(rows) == expected
    gate_config = next(g for g in config.gates if g.type == "schema")
    passed, rejected = SchemaGate(
        required_fields=gate_config.required_fields,
        min_tokens=gate_config.min_tokens,
        max_tokens=gate_config.max_tokens,
    ).run(rows)
    assert not rejected
    assert len(passed) == expected

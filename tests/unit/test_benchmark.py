"""Benchmark hygiene, not benchmark scores. Models are stubbed for these checks."""

import importlib.util
import json
from pathlib import Path

from agent_runtime import AgentConfig

spec = importlib.util.spec_from_file_location(
    "needle_benchmark",
    Path(__file__).parents[2] / "examples" / "benchmark.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_only_cleans_its_own_temporary_directories(tmp_path, monkeypatch):
    keep = tmp_path / "unrelated.txt"
    keep.write_text("keep me")
    visited = []

    def run(task, config):
        root = Path(config.workspace_root)
        assert root.parent == tmp_path
        assert root.name.startswith("needle-calc-")
        visited.append(root)
        return {"status": "COMPLETED", "success": True, "steps": 1, "invalid": 0, "seconds": 0.0}

    monkeypatch.setattr(benchmark, "run_native", run)
    monkeypatch.setattr(benchmark, "run_lifted", run)
    output = tmp_path / "results.jsonl"
    benchmark.main(["--tasks", "calc", "--workroot", str(tmp_path), "--out", str(output)])
    assert keep.read_text() == "keep me"
    assert len(visited) == 2 and all(not path.exists() for path in visited)
    assert len(output.read_text().splitlines()) == 2


def test_native_baseline_uses_same_tools_and_reports_result(tmp_path, monkeypatch):
    class Provider:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def _request(self, path, payload):
            self.calls += 1
            assert "tools" in payload  # only this benchmark baseline uses native structured calls
            message = (
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"expression": "2*(15+3)"}),
                            },
                        }
                    ],
                }
                if self.calls == 1
                else {"content": "36"}
            )
            return {"choices": [{"message": message}]}

    monkeypatch.setattr(benchmark, "OpenAICompatibleReasoningModel", Provider)
    record = benchmark.run_native("calc", AgentConfig(workspace_root=str(tmp_path)))
    assert record["status"] == "COMPLETED" and record["success"]
    assert record["steps"] == 1 and record["invalid"] == 0


def test_time_checker_is_not_just_a_colon(tmp_path):
    assert not benchmark._check("time", "An answer: no clock used", tmp_path)
    assert benchmark._check("time", "2026-09-05T10:30:00+00:00", tmp_path)

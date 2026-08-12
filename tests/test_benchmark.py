import json
import importlib.util
import subprocess
import sys

import pytest


def test_benchmark_thread_budget_smoke(tmp_path):
    output = tmp_path / "benchmark.json"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_performance.py",
            "--sequence-count",
            "100",
            "--target-counts",
            "1",
            "--total-threads",
            "1,2,4",
            "--numba-threads",
            "1,2,4",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    configurations = {
        (
            item["total_threads"],
            item["numba_threads"],
            item["joblib_workers"],
        )
        for item in payload["results"]
    }
    assert {(1, 1, 1), (2, 1, 1), (2, 2, 1), (4, 1, 1)} <= configurations
    assert output.is_file()


@pytest.mark.parametrize("returncode", [-9, 137, 143])
def test_resource_killed_worker_is_recorded_not_promoted_to_benchmark_failure(
    monkeypatch, returncode
):
    spec = importlib.util.spec_from_file_location(
        "benchmark_performance", "benchmarks/benchmark_performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class Arguments:
        sequence_count = 100
        coverage = False
        skewed_rows = False
        phase_timings = False

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, "", ""),
    )

    sample = module._sample_in_subprocess(
        Arguments(), 1, 1, 1, 2.0, "cache_miss_hot_jit"
    )

    assert sample == {
        "mode": "cache_miss_hot_jit",
        "status": "resource_limited",
        "returncode": returncode,
    }


def test_cache_storage_preflight_scales_from_a_one_target_measurement():
    spec = importlib.util.spec_from_file_location(
        "benchmark_performance", "benchmarks/benchmark_performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._estimated_cache_bytes_for_targets(200, 1) == 200
    assert module._estimated_cache_bytes_for_targets(200, 64) == 6_500
    assert module._resource_limited_samples(
        ("cache_miss_hot_jit",),
        reason="temporary_storage",
        required_bytes=6_500,
        available_bytes=1_000,
    )[0]["status"] == "resource_limited"


def test_benchmark_progress_output_is_always_parseable_json(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "benchmark_performance", "benchmarks/benchmark_performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    output = tmp_path / "benchmark.json"

    module._write_benchmark_output(output, [{"target_count": 1}])

    assert json.loads(output.read_text())["results"] == [{"target_count": 1}]

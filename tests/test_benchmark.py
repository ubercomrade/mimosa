import json
import subprocess
import sys


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
    assert {(1, 1, 1), (2, 1, 2), (2, 2, 1), (4, 1, 4)} <= configurations
    assert output.is_file()

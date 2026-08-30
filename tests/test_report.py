from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inferqual.report import render_html, write_bundle


def sample_report() -> dict[str, object]:
    return {
        "model": {"name": "TinyVisionNet", "input_shape": [1, 3, 224, 224]},
        "results": [
            {
                "id": "pytorch-cpu-fp32",
                "label": "PyTorch CPU",
                "device": "CPU",
                "precision": "FP32",
                "cosine_similarity": 1.0,
                "latency": {
                    "p50_ms": 1.25,
                    "p95_ms": 1.295,
                    "throughput_per_second": 800.0,
                },
            },
            {
                "id": "onnxruntime-cpu-int8",
                "label": "ONNX Runtime INT8",
                "device": "CPU",
                "precision": "INT8",
                "cosine_similarity": 0.99998,
                "latency": {
                    "p50_ms": 0.71,
                    "p95_ms": 0.719,
                    "throughput_per_second": 1408.451,
                },
            },
        ],
        "selected": "onnxruntime-cpu-int8",
        "comparison": {
            "baseline": "pytorch-cpu-fp32",
            "speedup_vs_pytorch_cpu": 1.761,
        },
    }


class ReportTests(unittest.TestCase):
    def test_report_escapes_result_values(self) -> None:
        report = sample_report()
        report["results"][1]["label"] = '<script data-x="1">bad()</script>'  # type: ignore[index]

        rendered = render_html(report)

        self.assertNotIn('<script data-x="1">', rendered)
        self.assertIn("&lt;script data-x=&quot;1&quot;&gt;bad()&lt;/script&gt;", rendered)

    def test_bundle_writes_the_result(self) -> None:
        report = sample_report()
        with tempfile.TemporaryDirectory() as directory:
            json_path, html_path = write_bundle(report, Path(directory))

            self.assertEqual(json.loads(json_path.read_text()), report)
            self.assertIn("Benchmark result", html_path.read_text())


if __name__ == "__main__":
    unittest.main()

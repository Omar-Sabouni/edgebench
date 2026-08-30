"""End-to-end PyTorch, ONNX Runtime, INT8, and CUDA benchmarking."""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from inferqual.metrics import fidelity, select_fastest, summarize_latency
from inferqual.model import DEFAULT_SEED, INPUT_SHAPE, build_model, make_input, make_inputs
from inferqual.report import write_bundle

DeviceTarget = Literal["auto", "cpu", "cuda"]


class BenchmarkError(RuntimeError):
    """Raised when a requested benchmark cannot produce a useful result."""


@dataclass(frozen=True, slots=True)
class Settings:
    device: DeviceTarget = "auto"
    warmup: int = 10
    iterations: int = 50
    threads: int = 1

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if isinstance(self.warmup, bool) or not 0 <= self.warmup <= 1_000:
            raise ValueError("warmup must be between 0 and 1,000")
        if isinstance(self.iterations, bool) or not 1 <= self.iterations <= 10_000:
            raise ValueError("iterations must be between 1 and 10,000")
        if isinstance(self.threads, bool) or not 1 <= self.threads <= 256:
            raise ValueError("threads must be between 1 and 256")


class _TorchRuntime:
    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self._device = torch.device(device)
        self._model = copy.deepcopy(model).to(self._device).eval()
        suffix = "cuda" if self._device.type == "cuda" else "cpu"
        self.id = f"pytorch-{suffix}-fp32"
        self.label = f"PyTorch {suffix.upper()}"
        self.device = suffix.upper()
        self.precision = "FP32"

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @torch.inference_mode()
    def run(self, inputs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        tensor = torch.from_numpy(inputs).to(self._device)
        output = self._model(tensor)
        if not isinstance(output, torch.Tensor):
            raise BenchmarkError("the built-in model did not return one tensor")
        return np.asarray(output.detach().cpu().numpy(), dtype=np.float32)


class _OrtRuntime:
    def __init__(
        self,
        model_path: Path,
        *,
        provider: str,
        precision: str,
        threads: int,
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [provider]
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self.device = "CUDA" if provider == "CUDAExecutionProvider" else "CPU"
        name = precision if precision == "INT8" else self.device
        self.id = f"onnxruntime-{self.device.lower()}-{precision.lower()}"
        self.label = f"ONNX Runtime {name}"
        self.precision = precision

    def synchronize(self) -> None:
        # Session.run returns host output and is synchronous for this latency scope.
        return None

    def run(self, inputs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        outputs = self._session.run(None, {self._input_name: inputs})
        if len(outputs) != 1:
            raise BenchmarkError("ONNX model returned an unexpected output count")
        return np.asarray(outputs[0], dtype=np.float32)


_Runtime = _TorchRuntime | _OrtRuntime


class _CalibrationReader(CalibrationDataReader):
    def __init__(self, inputs: list[npt.NDArray[np.float32]]) -> None:
        self._inputs = inputs
        self._index = 0

    def get_next(self) -> dict[str, npt.NDArray[np.float32]] | None:
        if self._index >= len(self._inputs):
            return None
        value = {"input": self._inputs[self._index]}
        self._index += 1
        return value

    def rewind(self) -> None:
        self._index = 0


def _short_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return (
        f"{error.__class__.__name__}: {message[:240]}" if message else error.__class__.__name__
    )


def _export_onnx(model: torch.nn.Module, sample: npt.NDArray[np.float32], path: Path) -> None:
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch.onnx.export(
                model,
                (torch.from_numpy(sample),),
                path,
                input_names=["input"],
                output_names=["output"],
                opset_version=18,
                dynamo=True,
                optimize=True,
                external_data=False,
                verbose=False,
            )
        onnx.checker.check_model(onnx.load(path))
    except Exception as exc:
        raise BenchmarkError(f"ONNX export failed: {_short_error(exc)}") from exc
    finally:
        logging.disable(previous_logging)


def _quantize_onnx(source: Path, destination: Path) -> None:
    preprocessed = destination.with_name(".model.preprocessed.onnx")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            quant_pre_process(source, preprocessed, skip_symbolic_shape=True)
            quantize_static(
                preprocessed,
                destination,
                _CalibrationReader(make_inputs(DEFAULT_SEED + 10_000, 12)),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
            )
        onnx.checker.check_model(onnx.load(destination))
    except Exception as exc:
        raise BenchmarkError(f"INT8 quantization failed: {_short_error(exc)}") from exc
    finally:
        preprocessed.unlink(missing_ok=True)


def _outputs(
    runtime: _Runtime, inputs: list[npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    return np.concatenate([runtime.run(case) for case in inputs], axis=0)


def _measure(
    runtime: _Runtime,
    sample: npt.NDArray[np.float32],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        runtime.run(sample)
    runtime.synchronize()
    samples_ms: list[float] = []
    for _ in range(iterations):
        runtime.synchronize()
        started = perf_counter_ns()
        runtime.run(sample)
        runtime.synchronize()
        samples_ms.append((perf_counter_ns() - started) / 1_000_000)
    return summarize_latency(samples_ms)


def _reference_outputs(
    model: torch.nn.Module, inputs: list[npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    with torch.inference_mode():
        values = [np.asarray(model(torch.from_numpy(case)).numpy()) for case in inputs]
    return np.concatenate(values).astype(np.float32, copy=False)


def _available_runtimes(
    model: torch.nn.Module,
    output: Path,
    settings: Settings,
    unavailable: list[dict[str, str]],
) -> list[_Runtime]:
    runtimes: list[_Runtime] = []
    providers = set(ort.get_available_providers())

    def add(identifier: str, factory: Callable[[], _Runtime]) -> None:
        try:
            runtimes.append(factory())
        except Exception as exc:
            unavailable.append({"id": identifier, "reason": _short_error(exc)})

    def ort_runtime(path: Path, provider: str, precision: str) -> _OrtRuntime:
        return _OrtRuntime(
            path, provider=provider, precision=precision, threads=settings.threads
        )

    fp32_path = output / "model.onnx"
    _export_onnx(model, make_input(DEFAULT_SEED, 50_000), fp32_path)
    if settings.device in {"auto", "cpu"}:
        runtimes.append(_TorchRuntime(model, "cpu"))
        add(
            "onnxruntime-cpu-fp32",
            lambda: ort_runtime(fp32_path, "CPUExecutionProvider", "FP32"),
        )

        int8_path = output / "model.int8.onnx"
        try:
            _quantize_onnx(fp32_path, int8_path)
            add(
                "onnxruntime-cpu-int8",
                lambda: ort_runtime(int8_path, "CPUExecutionProvider", "INT8"),
            )
        except Exception as exc:
            unavailable.append({"id": "onnxruntime-cpu-int8", "reason": _short_error(exc)})

    wants_cuda = settings.device == "cuda" or (
        settings.device == "auto"
        and (torch.cuda.is_available() or "CUDAExecutionProvider" in providers)
    )
    if wants_cuda:
        if torch.cuda.is_available():
            add("pytorch-cuda-fp32", lambda: _TorchRuntime(model, "cuda:0"))
        elif settings.device == "cuda":
            unavailable.append(
                {"id": "pytorch-cuda-fp32", "reason": "CUDA is not available to PyTorch"}
            )

        if "CUDAExecutionProvider" in providers:
            add(
                "onnxruntime-cuda-fp32",
                lambda: ort_runtime(fp32_path, "CUDAExecutionProvider", "FP32"),
            )
        elif settings.device == "cuda":
            unavailable.append(
                {
                    "id": "onnxruntime-cuda-fp32",
                    "reason": "CUDAExecutionProvider is not installed",
                }
            )
    return runtimes


def run_benchmark(output: Path, settings: Settings | None = None) -> dict[str, Any]:
    """Run the benchmark and write its JSON and HTML results."""
    settings = settings or Settings()
    output = output.expanduser()
    if settings.device == "cuda" and not (
        torch.cuda.is_available() or "CUDAExecutionProvider" in ort.get_available_providers()
    ):
        raise BenchmarkError("no CUDA backend is available")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise BenchmarkError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(settings.threads)
    try:
        model = build_model(DEFAULT_SEED)
        fidelity_inputs = make_inputs(DEFAULT_SEED + 1_000, 8)
        benchmark_input = make_input(DEFAULT_SEED + 2_000)
        reference = _reference_outputs(model, fidelity_inputs)
        unavailable: list[dict[str, str]] = []
        runtimes = _available_runtimes(model, output, settings, unavailable)
        if settings.device == "cuda" and not any(
            runtime.device == "CUDA" for runtime in runtimes
        ):
            raise BenchmarkError("no CUDA backend is available")
        if not runtimes:
            raise BenchmarkError("no benchmark backend is available")

        results: list[dict[str, Any]] = []
        for runtime in runtimes:
            try:
                candidate = _outputs(runtime, fidelity_inputs)
                fidelity_result = fidelity(reference, candidate, precision=runtime.precision)
                latency_result = _measure(
                    runtime, benchmark_input, settings.warmup, settings.iterations
                )
                results.append(
                    {
                        "id": runtime.id,
                        "label": runtime.label,
                        "device": runtime.device,
                        "precision": runtime.precision,
                        "status": "passed" if fidelity_result["passed"] else "failed",
                        "fidelity": {
                            "max_abs_error": fidelity_result["max_abs_error"],
                            "cosine_similarity": fidelity_result["cosine_similarity"],
                            "passed": fidelity_result["passed"],
                        },
                        "latency": {
                            "p50_ms": latency_result["p50_ms"],
                            "p95_ms": latency_result["p95_ms"],
                            "throughput_per_second": latency_result["throughput_per_second"],
                        },
                    }
                )
            except Exception as exc:
                unavailable.append({"id": runtime.id, "reason": _short_error(exc)})
        if not results:
            raise BenchmarkError("all benchmark backends failed")
        try:
            selected = select_fastest(results)
        except ValueError as exc:
            raise BenchmarkError(str(exc)) from exc

        report: dict[str, Any] = {
            "model": {
                "name": "TinyVisionNet",
                "input_shape": list(INPUT_SHAPE),
            },
            "results": [
                {
                    "id": result["id"],
                    "label": result["label"],
                    "device": result["device"],
                    "precision": result["precision"],
                    "cosine_similarity": result["fidelity"]["cosine_similarity"],
                    "latency": result["latency"],
                }
                for result in results
                if result["status"] == "passed"
            ],
            "selected": selected,
        }
        baseline = next(
            (result for result in results if result["id"] == "pytorch-cpu-fp32"), None
        )
        winner = next(result for result in results if result["id"] == selected)
        if baseline is not None and baseline["status"] == "passed":
            report["comparison"] = {
                "baseline": baseline["id"],
                "speedup_vs_pytorch_cpu": round(
                    baseline["latency"]["p50_ms"] / winner["latency"]["p50_ms"], 3
                ),
            }
        write_bundle(report, output)
        return report
    finally:
        torch.set_num_threads(previous_threads)

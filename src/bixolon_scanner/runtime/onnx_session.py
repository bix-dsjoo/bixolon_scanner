"""ONNX Runtime provider selection and session execution primitives."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from ..contracts.errors import (
    ModelExecutionError,
    ProviderExecutionError,
    ProviderInitializationError,
)

ExecutionProvider: TypeAlias = Literal[
    "cuda",
    "cpu",
    "directml",
    "openvino",
    "openvino_gpu",
]


def _openvino_model_cache_directory(
    cache_root: Path,
    device: str,
    model_path: Path,
) -> Path:
    resolved_model = model_path.resolve()
    stat = resolved_model.stat()
    identity = f"{resolved_model}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in resolved_model.stem
    )
    return (cache_root.resolve() / device.lower() / f"{safe_stem}-{digest}").resolve()


class OrtRunner:
    """Own an ONNX Runtime session and its provider-specific resources."""

    def __init__(
        self,
        model_path: Path,
        provider: ExecutionProvider,
        cuda_dll_dir: Path | None = None,
        *,
        enable_cuda_graph: bool = False,
        cuda_graph_output_shapes: dict[str, tuple[int, ...]] | None = None,
        cpu_intra_op_threads: int = 0,
        openvino_cache_dir: Path | None = None,
    ):
        if cpu_intra_op_threads < 0:
            raise ValueError("CPU intra-op thread count must be non-negative")
        try:
            import onnxruntime as ort

            self._dll_directory = None
            self._openvino_dll_directory = None
            self._openvino_dlls: list[object] = []
            self._cuda_dlls: list[object] = []
            if provider in {"openvino", "openvino_gpu"} and hasattr(os, "add_dll_directory"):
                bundled = getattr(sys, "_MEIPASS", None)
                spec = None if bundled else importlib.util.find_spec("openvino")
                libraries = (
                    Path(bundled)
                    if bundled
                    else Path(spec.origin).parent / "libs"
                    if spec is not None and spec.origin
                    else None
                )
                if libraries is not None and libraries.is_dir():
                    self._openvino_dll_directory = os.add_dll_directory(str(libraries.resolve()))
                    if os.name == "nt" and (libraries / "openvino.dll").is_file():
                        # ORT's provider loader does not consistently use AddDllDirectory.
                        # Keep the absolute-path dependency loaded for this session lifetime.
                        self._openvino_dlls.append(ctypes.WinDLL(str(libraries / "openvino.dll")))
            if provider == "cuda" and cuda_dll_dir is not None:
                cuda_dll_dir = cuda_dll_dir.resolve()
                if not cuda_dll_dir.is_dir():
                    raise ProviderInitializationError
                if hasattr(os, "add_dll_directory"):
                    self._dll_directory = os.add_dll_directory(str(cuda_dll_dir))
                if os.name == "nt":
                    # Keep explicit handles alive for the session lifetime. ONNX Runtime's
                    # preload helper releases its local ctypes handles and its optional CUDA
                    # plugin can then report a false NVRTC lookup failure in an isolated app
                    # bundle even though the primary CUDA EP remains available.
                    dependency_order = (
                        "cudart64_13.dll",
                        "cublasLt64_13.dll",
                        "cublas64_13.dll",
                        "cufft64_12.dll",
                        "nvJitLink_130_0.dll",
                        "nvrtc-builtins64_130.dll",
                        "nvrtc64_130_0.dll",
                        "zlibwapi.dll",
                        "cudnn64_9.dll",
                        "cudnn_ops64_9.dll",
                        "cudnn_cnn64_9.dll",
                        "cudnn_adv64_9.dll",
                        "cudnn_graph64_9.dll",
                        "cudnn_heuristic64_9.dll",
                        "cudnn_engines_precompiled64_9.dll",
                        "cudnn_engines_runtime_compiled64_9.dll",
                    )
                    for filename in dependency_order:
                        path = cuda_dll_dir / filename
                        if path.is_file():
                            self._cuda_dlls.append(ctypes.WinDLL(str(path)))
                else:
                    ort.preload_dlls(directory=str(cuda_dll_dir))
            available = ort.get_available_providers()
            provider_name = {
                "cuda": "CUDAExecutionProvider",
                "cpu": "CPUExecutionProvider",
                "directml": "DmlExecutionProvider",
                "openvino": "OpenVINOExecutionProvider",
                "openvino_gpu": "OpenVINOExecutionProvider",
            }[provider]
            if provider_name not in available:
                raise ProviderInitializationError
            options = ort.SessionOptions()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                if provider in {"openvino", "openvino_gpu"}
                else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            if provider == "cpu":
                # The detector, primary, detail and verifier own separate pools.
                # Let idle pools sleep while another stage uses the CPU.
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            if provider == "cpu" and cpu_intra_op_threads > 0:
                options.intra_op_num_threads = cpu_intra_op_threads
            if provider == "directml":
                # DirectML requires sequential execution and does not support ORT's
                # memory-pattern optimization for sessions with dynamic input shapes.
                options.enable_mem_pattern = False
            if provider == "openvino_gpu":
                # A GPU diagnostic must fail if ORT cannot assign the complete graph
                # to OpenVINO. Otherwise an unsupported node could make a nominal GPU
                # profile silently run on the generic CPU EP.
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.log_severity_level = 3
            provider_options: dict[str, str] = {}
            if provider == "cuda":
                provider_options["use_tf32"] = "0"
            if enable_cuda_graph and provider == "cuda":
                provider_options["enable_cuda_graph"] = "1"
            if provider == "directml":
                provider_options["device_id"] = "0"
            if provider in {"openvino", "openvino_gpu"}:
                device = "GPU" if provider == "openvino_gpu" else "CPU"
                provider_options["device_type"] = device
                device_config: dict[str, str] = {}
                if provider == "openvino_gpu":
                    device_config.update(
                        {
                            "PERFORMANCE_HINT": "LATENCY",
                            "NUM_STREAMS": "1",
                            "INFERENCE_PRECISION_HINT": "f32",
                        }
                    )
                if provider == "openvino":
                    device_config.update(
                        {
                            "PERFORMANCE_HINT": "LATENCY",
                            "NUM_STREAMS": "1",
                            "INFERENCE_PRECISION_HINT": "f32",
                        }
                    )
                    if cpu_intra_op_threads > 0:
                        device_config["INFERENCE_NUM_THREADS"] = str(cpu_intra_op_threads)
                if openvino_cache_dir is not None:
                    cache_dir = _openvino_model_cache_directory(
                        openvino_cache_dir,
                        device,
                        model_path,
                    )
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    device_config["CACHE_DIR"] = str(cache_dir)
                    device_config["CACHE_MODE"] = "OPTIMIZE_SPEED"
                if device_config:
                    provider_options["load_config"] = json.dumps(
                        {device: device_config}, separators=(",", ":")
                    )
            providers = (
                [(provider_name, provider_options)]
                if provider in {"cuda", "directml", "openvino", "openvino_gpu"}
                else [provider_name]
            )
            self.session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=providers
            )
            if self.session.get_providers()[0] != provider_name:
                raise ProviderInitializationError
            self.cuda = provider == "cuda"
            self.provider = provider
            self.accelerated = provider in {
                "cuda",
                "directml",
                "openvino",
                "openvino_gpu",
            }
            self.cuda_graph = self.cuda and enable_cuda_graph
            self.cuda_graph_output_shapes = cuda_graph_output_shapes or {}
            self._graph_binding = None
            self._graph_input_values: dict[str, object] = {}
            self._graph_output_values: list[object] = []
            self._graph_signature: tuple[tuple[str, tuple[int, ...]], ...] | None = None
        except ProviderInitializationError:
            self._close_initialization_handles()
            raise
        except Exception as exc:
            self._close_initialization_handles()
            raise ProviderInitializationError from exc

    def _close_initialization_handles(self) -> None:
        getattr(self, "_openvino_dlls", []).clear()
        for name in ("_dll_directory", "_openvino_dll_directory"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def run(self, output_names: list[str], input_name: str, tensor: np.ndarray) -> list[np.ndarray]:
        return self.run_inputs(output_names, {input_name: tensor})

    def run_inputs(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        try:
            if not self.cuda:
                return self.session.run(output_names, inputs)
            if self.cuda_graph:
                return self._run_cuda_graph(output_names, inputs)
            binding = self.session.io_binding()
            for input_name, tensor in inputs.items():
                binding.bind_cpu_input(input_name, tensor)
            for output_name in output_names:
                binding.bind_output(output_name, "cuda")
            self.session.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()
        except Exception as exc:
            if getattr(self, "provider", "cpu") in {"cuda", "openvino_gpu", "directml"}:
                raise ProviderExecutionError from exc
            raise ModelExecutionError from exc

    def close(self) -> None:
        """Release provider resources before replacing a live runtime adapter."""

        self._graph_binding = None
        self._graph_input_values.clear()
        self._graph_output_values.clear()
        self.session = None
        self._cuda_dlls.clear()
        self._close_initialization_handles()

    def _run_cuda_graph(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        import onnxruntime as ort

        signature = tuple((name, tuple(tensor.shape)) for name, tensor in inputs.items())
        if self._graph_binding is None:
            output_metadata = {value.name: value for value in self.session.get_outputs()}
            if any(
                name not in output_metadata
                or (
                    name not in self.cuda_graph_output_shapes
                    and any(
                        not isinstance(dimension, int) for dimension in output_metadata[name].shape
                    )
                )
                for name in output_names
            ):
                raise ModelExecutionError
            self._graph_signature = signature
            self._graph_binding = self.session.io_binding()
            for name, tensor in inputs.items():
                value = ort.OrtValue.ortvalue_from_shape_and_type(
                    tensor.shape, tensor.dtype, "cuda", 0
                )
                self._graph_input_values[name] = value
                self._graph_binding.bind_ortvalue_input(name, value)
            for name in output_names:
                metadata = output_metadata[name]
                if metadata.type != "tensor(float)":
                    raise ModelExecutionError
                value = ort.OrtValue.ortvalue_from_shape_and_type(
                    self.cuda_graph_output_shapes.get(name, tuple(metadata.shape)),
                    np.float32,
                    "cuda",
                    0,
                )
                self._graph_output_values.append(value)
                self._graph_binding.bind_ortvalue_output(name, value)
        elif signature != self._graph_signature:
            raise ModelExecutionError
        for name, tensor in inputs.items():
            self._graph_input_values[name].update_inplace(tensor)
        self.session.run_with_iobinding(self._graph_binding)
        return [value.numpy() for value in self._graph_output_values]


def select_provider(
    mode: Literal["auto", "cuda", "cpu", "directml", "openvino", "openvino_gpu"],
    cuda_dll_dir: Path | None = None,
) -> ExecutionProvider:
    """Resolve the requested provider without silently downgrading explicit acceleration."""

    if mode == "cpu":
        return "cpu"
    dll_directory = None
    try:
        if mode in {"auto", "cuda"} and cuda_dll_dir is not None and os.name == "nt":
            resolved_cuda_dir = cuda_dll_dir.resolve()
            if not resolved_cuda_dir.is_dir():
                raise ProviderInitializationError
            if hasattr(os, "add_dll_directory"):
                dll_directory = os.add_dll_directory(str(resolved_cuda_dir))
        import onnxruntime as ort

        available = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in available
        has_directml = "DmlExecutionProvider" in available
        has_openvino = "OpenVINOExecutionProvider" in available
    except Exception as exc:
        raise ProviderInitializationError from exc
    finally:
        if dll_directory is not None:
            dll_directory.close()
    if mode == "cuda" and not has_cuda:
        raise ProviderInitializationError
    if mode == "directml" and not has_directml:
        raise ProviderInitializationError
    if mode in {"openvino", "openvino_gpu"} and not has_openvino:
        raise ProviderInitializationError
    if mode == "directml":
        return "directml"
    if mode in {"openvino", "openvino_gpu"}:
        return mode
    return "cuda" if has_cuda else "cpu"


__all__ = ["ExecutionProvider", "OrtRunner", "select_provider"]

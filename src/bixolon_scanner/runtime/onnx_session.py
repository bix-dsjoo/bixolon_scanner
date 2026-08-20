"""ONNX Runtime provider selection and session execution primitives."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Literal

import numpy as np

from ..contracts.errors import ModelExecutionError, ProviderInitializationError


class OrtRunner:
    """Own an ONNX Runtime session and its provider-specific resources."""

    def __init__(
        self,
        model_path: Path,
        provider: Literal["cuda", "cpu"],
        cuda_dll_dir: Path | None = None,
        *,
        enable_cuda_graph: bool = False,
        cuda_graph_output_shapes: dict[str, tuple[int, ...]] | None = None,
        cpu_intra_op_threads: int = 0,
    ):
        if cpu_intra_op_threads < 0:
            raise ValueError("CPU intra-op thread count must be non-negative")
        try:
            import onnxruntime as ort

            self._dll_directory = None
            self._cuda_dlls: list[object] = []
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
            provider_name = (
                "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
            )
            if provider_name not in available:
                raise ProviderInitializationError
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            if provider == "cpu" and cpu_intra_op_threads > 0:
                options.intra_op_num_threads = cpu_intra_op_threads
            options.log_severity_level = 3
            provider_options = {"use_tf32": "0"}
            if enable_cuda_graph and provider == "cuda":
                provider_options["enable_cuda_graph"] = "1"
            providers = (
                [(provider_name, provider_options)] if provider == "cuda" else [provider_name]
            )
            self.session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=providers
            )
            if self.session.get_providers()[0] != provider_name:
                raise ProviderInitializationError
            self.cuda = provider == "cuda"
            self.cuda_graph = self.cuda and enable_cuda_graph
            self.cuda_graph_output_shapes = cuda_graph_output_shapes or {}
            self._graph_binding = None
            self._graph_input_values: dict[str, object] = {}
            self._graph_output_values: list[object] = []
            self._graph_signature: tuple[tuple[str, tuple[int, ...]], ...] | None = None
        except ProviderInitializationError:
            raise
        except Exception as exc:
            raise ProviderInitializationError from exc

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
            raise ModelExecutionError from exc

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


def select_provider(mode: Literal["auto", "cuda", "cpu"]) -> Literal["cuda", "cpu"]:
    """Resolve the requested provider without silently downgrading explicit CUDA."""

    if mode == "cpu":
        return "cpu"
    try:
        import onnxruntime as ort

        has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as exc:
        raise ProviderInitializationError from exc
    if mode == "cuda" and not has_cuda:
        raise ProviderInitializationError
    return "cuda" if has_cuda else "cpu"


__all__ = ["OrtRunner", "select_provider"]

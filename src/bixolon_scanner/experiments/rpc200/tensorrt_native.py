from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from ...contracts.errors import ModelExecutionError, ProviderInitializationError
from ...contracts.model_package import ModelPackage
from ...runtime.onnx import OnnxClassifier, OnnxDetector


class _CudaRuntime:
    _HOST_TO_DEVICE = 1
    _DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        try:
            self.library = ctypes.CDLL("libcudart.so")
        except OSError as exc:
            raise ProviderInitializationError from exc
        self.library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.library.cudaMalloc.restype = ctypes.c_int
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaFree.restype = ctypes.c_int
        self.library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.cudaMemcpyAsync.restype = ctypes.c_int
        self.library.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.library.cudaStreamCreate.restype = ctypes.c_int
        self.library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaStreamDestroy.restype = ctypes.c_int
        self.library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.library.cudaStreamSynchronize.restype = ctypes.c_int

    @staticmethod
    def _check(status: int) -> None:
        if status != 0:
            raise ModelExecutionError

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(self.library.cudaMalloc(ctypes.byref(pointer), size))
        if pointer.value is None:
            raise ModelExecutionError
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        self._check(self.library.cudaFree(ctypes.c_void_p(pointer)))

    def copy_to_device(self, pointer: int, tensor: np.ndarray, stream: int) -> None:
        self._check(
            self.library.cudaMemcpyAsync(
                ctypes.c_void_p(pointer),
                ctypes.c_void_p(tensor.ctypes.data),
                tensor.nbytes,
                self._HOST_TO_DEVICE,
                ctypes.c_void_p(stream),
            )
        )

    def copy_to_host(self, tensor: np.ndarray, pointer: int, stream: int) -> None:
        self._check(
            self.library.cudaMemcpyAsync(
                ctypes.c_void_p(tensor.ctypes.data),
                ctypes.c_void_p(pointer),
                tensor.nbytes,
                self._DEVICE_TO_HOST,
                ctypes.c_void_p(stream),
            )
        )

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self._check(self.library.cudaStreamCreate(ctypes.byref(stream)))
        if stream.value is None:
            raise ModelExecutionError
        return int(stream.value)

    def destroy_stream(self, stream: int) -> None:
        self._check(self.library.cudaStreamDestroy(ctypes.c_void_p(stream)))

    def synchronize(self, stream: int) -> None:
        self._check(self.library.cudaStreamSynchronize(ctypes.c_void_p(stream)))


class TensorRTRunner:
    """Small experiment-only TensorRT runner with the same interface as OrtRunner."""

    def __init__(self, engine_path: Path):
        try:
            import tensorrt as trt

            self._trt = trt
            self._cuda = _CudaRuntime()
            self._logger = trt.Logger(trt.Logger.ERROR)
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if self._engine is None:
                raise ProviderInitializationError
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise ProviderInitializationError
            self._stream = self._cuda.create_stream()
            self._buffers: dict[str, tuple[int, int]] = {}
            self._names = [
                self._engine.get_tensor_name(index) for index in range(self._engine.num_io_tensors)
            ]
        except ProviderInitializationError:
            raise
        except Exception as exc:
            raise ProviderInitializationError from exc

    def close(self) -> None:
        for pointer, _ in self._buffers.values():
            self._cuda.free(pointer)
        self._buffers.clear()
        if getattr(self, "_stream", None) is not None:
            self._cuda.destroy_stream(self._stream)
            self._stream = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _buffer(self, name: str, size: int) -> int:
        current = self._buffers.get(name)
        if current is not None and current[1] >= size:
            return current[0]
        if current is not None:
            self._cuda.free(current[0])
        pointer = self._cuda.malloc(size)
        self._buffers[name] = (pointer, size)
        return pointer

    def run(self, output_names: list[str], input_name: str, tensor: np.ndarray) -> list[np.ndarray]:
        profile = self._engine.get_tensor_profile_shape(input_name, 0)
        max_batch = int(profile[2][0]) if profile else int(tensor.shape[0])
        if tensor.shape[0] > max_batch:
            chunks = [
                self.run(output_names, input_name, tensor[start : start + max_batch])
                for start in range(0, tensor.shape[0], max_batch)
            ]
            return [
                np.concatenate([chunk[index] for chunk in chunks], axis=0)
                for index in range(len(output_names))
            ]
        return self.run_inputs(output_names, {input_name: tensor})

    def run_inputs(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        try:
            trt = self._trt
            for name, value in inputs.items():
                expected_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
                tensor = np.ascontiguousarray(value, dtype=expected_dtype)
                if not self._context.set_input_shape(name, tensor.shape):
                    raise ModelExecutionError
                inputs[name] = tensor

            unresolved = self._context.infer_shapes()
            if unresolved:
                raise ModelExecutionError

            outputs: dict[str, np.ndarray] = {}
            for name in self._names:
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    tensor = inputs[name]
                    pointer = self._buffer(name, tensor.nbytes)
                    self._cuda.copy_to_device(pointer, tensor, self._stream)
                else:
                    shape = tuple(int(value) for value in self._context.get_tensor_shape(name))
                    if any(value < 0 for value in shape):
                        raise ModelExecutionError
                    dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
                    tensor = np.empty(shape, dtype=dtype)
                    outputs[name] = tensor
                    pointer = self._buffer(name, tensor.nbytes)
                if not self._context.set_tensor_address(name, pointer):
                    raise ModelExecutionError

            if not self._context.execute_async_v3(self._stream):
                raise ModelExecutionError
            for name, tensor in outputs.items():
                self._cuda.copy_to_host(tensor, self._buffers[name][0], self._stream)
            self._cuda.synchronize(self._stream)
            return [outputs[name] for name in output_names]
        except ModelExecutionError:
            raise
        except Exception as exc:
            raise ModelExecutionError from exc


def build_tensorrt_adapters(
    model_package: ModelPackage,
    detector_engine: Path | None,
    classifier_engine: Path | None,
):
    if getattr(model_package.metadata, "count_verifier", None) is not None:
        raise ProviderInitializationError

    if detector_engine is None:
        detector = OnnxDetector(
            model_package.detector_path, model_package.metadata.detector, "cuda"
        )
    else:
        detector = OnnxDetector.__new__(OnnxDetector)
        detector.metadata = model_package.metadata.detector
        detector.runner = TensorRTRunner(detector_engine)
        detector.version = detector.metadata.version

    if classifier_engine is None:
        classifier = OnnxClassifier(
            model_package.classifier_path, model_package.metadata.classifier, "cuda"
        )
    else:
        classifier = OnnxClassifier.__new__(OnnxClassifier)
        classifier.metadata = model_package.metadata.classifier
        classifier.runner = TensorRTRunner(classifier_engine)
        classifier.version = classifier.metadata.version

    detector.warmup()
    classifier.warmup()
    if detector_engine is not None and classifier_engine is not None:
        provider = "tensorrt"
    elif detector_engine is not None:
        provider = "tensorrt-detector"
    else:
        provider = "tensorrt-classifier"
    return detector, classifier, provider

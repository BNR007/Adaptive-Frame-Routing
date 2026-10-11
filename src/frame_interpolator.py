"""Frame-interpolation interfaces and real RIFE ONNX midpoint inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
import logging
from typing import Any, Callable, Protocol

import cv2
import numpy as np
import onnxruntime as ort

from resource_validation import ResourceLease
from regional_transform import (
    RegionProvenance,
    RegionalConfig,
    RegionalResult,
    RegionRoute,
    RegionalTransformController,
)


SUPPORTED_INTERPOLATOR_LAYOUTS = ("nchw", "nhwc")
SUPPORTED_INTERPOLATOR_PROVIDERS = ("cpu", "directml")


class FrameInterpolatorError(RuntimeError):
    """Raised when a frame interpolator cannot initialize or be used."""


class RIFEInterpolatorError(FrameInterpolatorError):
    """Raised for unsupported or unavailable RIFE ONNX infrastructure."""


class FrameInterpolator(ABC):
    """Reusable lifecycle interface for optional frame interpolation."""

    produces_intermediate_frame = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize resources required by the interpolator."""

    @abstractmethod
    def interpolate(self, frame_a: Any, frame_b: Any) -> Any:
        """Return the frame selected or produced between two input frames."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release owned resources. Implementations must be idempotent."""


class NoOpInterpolator(FrameInterpolator):
    """Lifecycle-aware interpolator that returns the current frame unchanged."""

    def __init__(self) -> None:
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        self._initialized = True

    def interpolate(self, frame_a: Any, frame_b: Any) -> Any:
        del frame_a
        if not self._initialized:
            raise FrameInterpolatorError("NoOpInterpolator is not initialized.")
        return frame_b

    def shutdown(self) -> None:
        self._initialized = False


@dataclass(frozen=True, slots=True)
class RIFETensorMetadata:
    name: str
    dtype: str
    shape: tuple[object, ...]
    layout: str


class _SessionValue(Protocol):
    name: str
    shape: list[object]
    type: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_SessionValue]: ...

    def get_outputs(self) -> list[_SessionValue]: ...

    def get_providers(self) -> list[str]: ...

    def run(self, output_names, input_feed): ...


SessionFactory = Callable[..., _InferenceSession]


def _normalize_choice(
    value: str,
    field_name: str,
    supported_values: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip().lower()
    if normalized not in supported_values:
        raise ValueError(
            f"Unsupported {field_name}: {normalized}. Expected one of: "
            f"{', '.join(supported_values)}."
        )
    return normalized


def _static_dimension(value: object, tensor_name: str) -> int | None:
    if isinstance(value, (int, np.integer)):
        dimension = int(value)
        if dimension <= 0:
            raise RIFEInterpolatorError(
                f"{tensor_name} contains non-positive dimension {dimension}."
            )
        return dimension
    if value is None or isinstance(value, str):
        return None
    raise RIFEInterpolatorError(
        f"{tensor_name} contains unsupported dimension metadata: {value!r}."
    )


class RIFEInterpolator(FrameInterpolator):
    """Strict midpoint interpolation for two-input RIFE ONNX image models."""

    produces_intermediate_frame = True

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_layout: str = "nchw",
        output_layout: str = "nchw",
        provider: str = "cpu",
        device_id: int = 0,
        inference_width: int | None = None,
        inference_height: int | None = None,
        temporal_stabilization: bool = False,
        regional_transform: bool = False,
        regional_budget_ms: float | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.model_path = model_path
        self.input_layout = _normalize_choice(
            input_layout,
            "RIFE input layout",
            SUPPORTED_INTERPOLATOR_LAYOUTS,
        )
        self.output_layout = _normalize_choice(
            output_layout,
            "RIFE output layout",
            SUPPORTED_INTERPOLATOR_LAYOUTS,
        )
        self.provider = _normalize_choice(
            provider,
            "RIFE execution provider",
            SUPPORTED_INTERPOLATOR_PROVIDERS,
        )
        if isinstance(device_id, bool) or not isinstance(device_id, int):
            raise TypeError("RIFE DirectML device ID must be an integer.")
        if device_id < 0:
            raise ValueError("RIFE DirectML device ID must be zero or greater.")
        self.device_id = device_id
        if (inference_width is None) != (inference_height is None):
            raise ValueError(
                "RIFE inference width and height must either both be set or both be omitted."
            )
        for value, name in (
            (inference_width, "RIFE inference width"),
            (inference_height, "RIFE inference height"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")
        self.inference_width = inference_width
        self.inference_height = inference_height
        if not isinstance(temporal_stabilization, bool):
            raise TypeError("Temporal stabilization must be a boolean.")
        self.temporal_stabilization = temporal_stabilization
        if not isinstance(regional_transform, bool):
            raise TypeError("Regional transformation must be a boolean.")
        if regional_budget_ms is not None and (
            isinstance(regional_budget_ms, bool)
            or not isinstance(regional_budget_ms, (int, float))
            or not np.isfinite(regional_budget_ms)
            or regional_budget_ms <= 0.0
        ):
            raise ValueError("Regional deadline budget must be positive when set.")
        self.regional_transform = regional_transform
        self.regional_budget_ms = (
            None if regional_budget_ms is None else float(regional_budget_ms)
        )
        self.regional_controller = (
            RegionalTransformController(RegionalConfig())
            if regional_transform
            else None
        )
        self.last_regional_result: RegionalResult | None = None
        self._input_alignment = 32
        self._session_factory = session_factory or ort.InferenceSession
        self._session: _InferenceSession | None = None
        self.available_providers: tuple[str, ...] = ()
        self.active_providers: tuple[str, ...] = ()
        self.input_metadata: tuple[RIFETensorMetadata, ...] = ()
        self.output_metadata: RIFETensorMetadata | None = None
        self._input_names: tuple[str, str] | None = None
        self._output_name: str | None = None
        self._input_shapes: tuple[tuple[object, ...], tuple[object, ...]] | None = None
        self._output_shape: tuple[object, ...] | None = None
        self._profile_samples = []
        self._profile_started_at = time.perf_counter()
        self.last_preprocessing_ms: float | None = None
        self.last_inference_ms: float | None = None
        self.last_postprocessing_ms: float | None = None
        self.last_total_ms: float | None = None
        self.last_flow_ms: float = 0.0
        self.last_generation_method = "rife_midpoint"
        self.input_dimensions: tuple[int, int] | None = None
        self.padded_input_dimensions: tuple[int, int] | None = None
        self.output_dimensions: tuple[int, int] | None = None
        self.last_stabilized_fraction = 0.0
        self.last_interpolation_confidence = 1.0
        self.last_duplicate_bypass = False
        # These arrays are private scratch space used only during the synchronous
        # interpolate call. Reusing them avoids allocating and clearing several
        # megabytes for every generated frame, which otherwise competes with
        # capture and presentation under live GPU load.
        self._inference_frame_buffers: tuple[np.ndarray, np.ndarray] | None = None
        self._input_tensor_buffers: tuple[np.ndarray, np.ndarray] | None = None
        self._full_midpoint_buffer: np.ndarray | None = None
        self._full_fallback_buffer: np.ndarray | None = None
        self._full_confidence_buffer: np.ndarray | None = None
        self._full_stable_buffer: np.ndarray | None = None
        self._flow_cache_key: tuple[int, int, tuple[int, ...], tuple[int, ...]] | None = None
        self._flow_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._validation_session_lease = ResourceLease("onnx_sessions")

    @property
    def initialized(self) -> bool:
        return self._session is not None

    @staticmethod
    def _validate_image_tensor(
        value: _SessionValue,
        layout: str,
        tensor_name: str,
    ) -> tuple[object, ...]:
        if getattr(value, "type", None) != "tensor(float)":
            raise RIFEInterpolatorError(
                f"{tensor_name} must be float32; got {getattr(value, 'type', None)!r}."
            )
        shape = getattr(value, "shape", None)
        if not isinstance(shape, (list, tuple)) or len(shape) != 4:
            rank = len(shape) if isinstance(shape, (list, tuple)) else "unknown"
            raise RIFEInterpolatorError(
                f"{tensor_name} must be a rank-4 {layout.upper()} image tensor; "
                f"got rank {rank}."
            )
        shape = tuple(shape)
        static_shape = tuple(
            _static_dimension(dimension, tensor_name) for dimension in shape
        )
        if static_shape[0] not in (None, 1):
            raise RIFEInterpolatorError(
                f"{tensor_name} must use batch size 1 or a dynamic batch dimension."
            )
        channel_index = 1 if layout == "nchw" else 3
        if static_shape[channel_index] not in (None, 3):
            raise RIFEInterpolatorError(
                f"{tensor_name} must contain exactly 3 image channels for "
                f"{layout.upper()}."
            )
        return shape

    @staticmethod
    def _spatial_dimensions(
        shape: tuple[object, ...],
        layout: str,
        tensor_name: str,
    ) -> tuple[int | None, int | None]:
        indices = (2, 3) if layout == "nchw" else (1, 2)
        return tuple(
            _static_dimension(shape[index], tensor_name) for index in indices
        )

    @staticmethod
    def _validate_matching_dimensions(
        first: tuple[int | None, int | None],
        second: tuple[int | None, int | None],
        description: str,
    ) -> None:
        for first_dimension, second_dimension in zip(first, second):
            if (
                first_dimension is not None
                and second_dimension is not None
                and first_dimension != second_dimension
            ):
                raise RIFEInterpolatorError(
                    f"RIFE {description} dimensions must match; got "
                    f"{first} and {second}."
                )

    def initialize(self) -> None:
        if self._session is not None:
            return
        if self.model_path is None or not str(self.model_path).strip():
            raise RIFEInterpolatorError(
                "RIFE initialization requires an ONNX model path."
            )
        resolved_path = Path(self.model_path).expanduser()
        if not resolved_path.exists():
            raise RIFEInterpolatorError(f"RIFE ONNX model does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise RIFEInterpolatorError(
                f"RIFE ONNX model path is not a file: {resolved_path}"
            )

        try:
            available_providers = tuple(ort.get_available_providers())
        except Exception as error:
            raise RIFEInterpolatorError(
                f"Unable to query ONNX Runtime providers for RIFE: {error}"
            ) from error
        self.available_providers = available_providers
        requested_provider = (
            "CPUExecutionProvider"
            if self.provider == "cpu"
            else "DmlExecutionProvider"
        )
        if requested_provider not in available_providers:
            raise RIFEInterpolatorError(
                f"Requested {requested_provider} is unavailable for RIFE. "
                f"Available providers: {', '.join(available_providers) or 'none'}."
            )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if self.provider == "directml":
            # Avoid a large idle CPU pool competing with capture/OpenCV and the game.
            session_options.intra_op_num_threads = 1
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers: list[Any] = [
                ("DmlExecutionProvider", {"device_id": self.device_id})
            ]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            session = self._session_factory(
                str(resolved_path),
                sess_options=session_options,
                providers=providers,
            )
            active_providers = tuple(session.get_providers())
            inputs = session.get_inputs()
            outputs = session.get_outputs()
        except Exception as error:
            raise RIFEInterpolatorError(
                f"Unable to load RIFE ONNX model '{resolved_path}': {error}"
            ) from error
        get_meta = getattr(session, "get_modelmeta", None)
        metadata = get_meta().custom_metadata_map if get_meta is not None else {}
        alignment = int(metadata.get("urfts.input_alignment", "32"))
        if alignment < 16 or alignment > 512 or alignment & (alignment - 1):
            raise RIFEInterpolatorError("Unsupported RIFE input alignment.")
        self._input_alignment = alignment
        if requested_provider not in active_providers:
            raise RIFEInterpolatorError(
                f"Requested {requested_provider} is not active for RIFE. Active "
                f"providers: {', '.join(active_providers) or 'none'}."
            )
        if len(inputs) != 2:
            raise RIFEInterpolatorError(
                "RIFE infrastructure supports exactly two image inputs; "
                f"the model exposes {len(inputs)} inputs."
            )
        if len(outputs) != 1:
            raise RIFEInterpolatorError(
                "RIFE infrastructure supports exactly one image output; "
                f"the model exposes {len(outputs)} outputs."
            )

        input_shapes = tuple(
            self._validate_image_tensor(
                value,
                self.input_layout,
                f"RIFE input {index + 1}",
            )
            for index, value in enumerate(inputs)
        )
        output_shape = self._validate_image_tensor(
            outputs[0],
            self.output_layout,
            "RIFE output",
        )
        first_spatial = self._spatial_dimensions(
            input_shapes[0], self.input_layout, "RIFE input 1"
        )
        second_spatial = self._spatial_dimensions(
            input_shapes[1], self.input_layout, "RIFE input 2"
        )
        output_spatial = self._spatial_dimensions(
            output_shape, self.output_layout, "RIFE output"
        )
        self._validate_matching_dimensions(
            first_spatial,
            second_spatial,
            "input spatial",
        )
        self._validate_matching_dimensions(
            first_spatial,
            output_spatial,
            "input/output spatial",
        )

        self._session = session
        self.active_providers = active_providers
        self.input_metadata = tuple(
            RIFETensorMetadata(
                name=value.name,
                dtype=value.type,
                shape=shape,
                layout=self.input_layout,
            )
            for value, shape in zip(inputs, input_shapes)
        )
        self.output_metadata = RIFETensorMetadata(
            name=outputs[0].name,
            dtype=outputs[0].type,
            shape=output_shape,
            layout=self.output_layout,
        )
        self._validation_session_lease.acquire()
        self._input_names = (inputs[0].name, inputs[1].name)
        self._output_name = outputs[0].name
        self._input_shapes = (input_shapes[0], input_shapes[1])
        self._output_shape = output_shape

    def warmup(self, iterations: int = 5) -> float:
        """Compile and exercise the configured ONNX graph before live capture."""
        if isinstance(iterations, bool) or not isinstance(iterations, int):
            raise TypeError("RIFE warmup iterations must be an integer.")
        if iterations < 1:
            raise ValueError("RIFE warmup iterations must be positive.")
        if (
            self._session is None
            or self._input_names is None
            or self._output_name is None
        ):
            raise RIFEInterpolatorError("RIFEInterpolator is not initialized.")
        if self.inference_width is None or self.inference_height is None:
            return 0.0

        height = self.inference_height
        width = self.inference_width
        padded_height = (
            (height + self._input_alignment - 1) // self._input_alignment
        ) * self._input_alignment
        padded_width = (
            (width + self._input_alignment - 1) // self._input_alignment
        ) * self._input_alignment
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        tensor = self._prepare_input(frame, padded_height, padded_width)
        started = time.perf_counter()
        call_times_ms: list[float] = []
        try:
            for _ in range(iterations):
                call_started = time.perf_counter()
                outputs = self._session.run(
                    [self._output_name],
                    {
                        self._input_names[0]: tensor,
                        self._input_names[1]: tensor,
                    },
                )
                if len(outputs) != 1:
                    raise RIFEInterpolatorError(
                        "RIFE ONNX warmup returned an unexpected output count."
                    )
                call_times_ms.append((time.perf_counter() - call_started) * 1000.0)
        except RIFEInterpolatorError:
            raise
        except Exception as error:
            raise RIFEInterpolatorError(
                f"RIFE ONNX warmup inference failed: {error}"
            ) from error
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self.regional_controller is not None and call_times_ms:
            # Exclude graph compilation when possible, then add the measured
            # fallback-path overhead conservatively. This prevents one visible
            # live stall before the deadline controller learns that a model is
            # already too expensive for the requested frame budget.
            steady_times = call_times_ms[1:] or call_times_ms
            predicted_transform_ms = float(np.median(steady_times)) + 12.0
            self.regional_controller.record_inference_cost(predicted_transform_ms)
        logging.getLogger(__name__).info(
            "RIFE warmup: %s calls at %sx%s completed in %.2f ms",
            iterations,
            width,
            height,
            elapsed_ms,
        )
        return elapsed_ms

    @staticmethod
    def _validate_frame_pair(frame_a: Any, frame_b: Any) -> None:
        for frame, name in ((frame_a, "frame_a"), (frame_b, "frame_b")):
            if not isinstance(frame, np.ndarray):
                raise RIFEInterpolatorError(f"{name} must be a NumPy array.")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RIFEInterpolatorError(
                    f"{name} must have HxWx3 dimensions; got {frame.shape}."
                )
            if frame.dtype != np.uint8:
                raise RIFEInterpolatorError(
                    f"{name} must use uint8 pixels; got {frame.dtype}."
                )
        if frame_a.shape != frame_b.shape:
            raise RIFEInterpolatorError(
                f"RIFE input frame dimensions must match; got {frame_a.shape} "
                f"and {frame_b.shape}."
            )

    @staticmethod
    def _validate_runtime_shape(
        actual_shape: tuple[int, ...],
        metadata_shape: tuple[object, ...],
        tensor_name: str,
    ) -> None:
        if len(actual_shape) != 4:
            raise RIFEInterpolatorError(
                f"{tensor_name} must be rank 4 at runtime; got {actual_shape}."
            )
        for index, (actual, expected) in enumerate(zip(actual_shape, metadata_shape)):
            fixed_dimension = _static_dimension(expected, tensor_name)
            if fixed_dimension is not None and actual != fixed_dimension:
                raise RIFEInterpolatorError(
                    f"{tensor_name} runtime shape {actual_shape} does not match "
                    f"dimension {index}={fixed_dimension}."
                )

    def _prepare_input(
        self,
        frame: np.ndarray,
        padded_height: int,
        padded_width: int,
    ) -> np.ndarray:
        image = frame[..., ::-1].astype(np.float32) / np.float32(255.0)
        if padded_height != frame.shape[0] or padded_width != frame.shape[1]:
            image = np.pad(
                image,
                (
                    (0, padded_height - frame.shape[0]),
                    (0, padded_width - frame.shape[1]),
                    (0, 0),
                ),
                mode="constant",
            )
        if self.input_layout == "nchw":
            image = np.transpose(image, (2, 0, 1))
        return np.ascontiguousarray(image[np.newaxis, ...], dtype=np.float32)

    def _resize_for_inference(self, frame: np.ndarray) -> np.ndarray:
        if self.inference_width is None or self.inference_height is None:
            return frame
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            self.inference_width / width,
            self.inference_height / height,
        )
        if scale >= 1.0:
            return frame
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    def _resize_pair_for_inference(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Downscale an endpoint pair into reusable, independent buffers."""
        if self.inference_width is None or self.inference_height is None:
            return frame_a, frame_b
        height, width = frame_a.shape[:2]
        scale = min(
            1.0,
            self.inference_width / width,
            self.inference_height / height,
        )
        if scale >= 1.0:
            return frame_a, frame_b
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        shape = (resized_height, resized_width, 3)
        buffers = self._inference_frame_buffers
        if buffers is None or buffers[0].shape != shape:
            buffers = (
                np.empty(shape, dtype=np.uint8),
                np.empty(shape, dtype=np.uint8),
            )
            self._inference_frame_buffers = buffers
        cv2.resize(
            frame_a,
            (resized_width, resized_height),
            dst=buffers[0],
            interpolation=(
                cv2.INTER_LINEAR if self.regional_transform else cv2.INTER_AREA
            ),
        )
        cv2.resize(
            frame_b,
            (resized_width, resized_height),
            dst=buffers[1],
            interpolation=(
                cv2.INTER_LINEAR if self.regional_transform else cv2.INTER_AREA
            ),
        )
        return buffers

    def _prepare_input_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        padded_height: int,
        padded_width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert BGR endpoints directly into reusable model tensors."""
        if self.input_layout == "nchw":
            shape = (1, 3, padded_height, padded_width)
        else:
            shape = (1, padded_height, padded_width, 3)
        buffers = self._input_tensor_buffers
        if buffers is None or buffers[0].shape != shape:
            buffers = (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )
            self._input_tensor_buffers = buffers
        height, width = frame_a.shape[:2]
        scale = np.float32(1.0 / 255.0)
        for frame, tensor in zip((frame_a, frame_b), buffers):
            # Clear only when padding exists; every active image pixel is
            # overwritten below.
            if height != padded_height or width != padded_width:
                tensor.fill(0.0)
            if self.input_layout == "nchw":
                np.multiply(
                    frame[..., 2], scale, out=tensor[0, 0, :height, :width]
                )
                np.multiply(
                    frame[..., 1], scale, out=tensor[0, 1, :height, :width]
                )
                np.multiply(
                    frame[..., 0], scale, out=tensor[0, 2, :height, :width]
                )
            else:
                destination = tensor[0, :height, :width]
                np.multiply(frame[..., 2], scale, out=destination[..., 0])
                np.multiply(frame[..., 1], scale, out=destination[..., 1])
                np.multiply(frame[..., 0], scale, out=destination[..., 2])
        return buffers

    def _prepare_output(
        self,
        value: object,
        original_height: int,
        original_width: int,
        padded_height: int,
        padded_width: int,
    ) -> np.ndarray:
        tensor = np.asarray(value)
        if tensor.dtype != np.float32:
            raise RIFEInterpolatorError(
                f"RIFE output must be float32; got {tensor.dtype}."
            )
        if self._output_shape is None:
            raise RIFEInterpolatorError("RIFE output metadata is unavailable.")
        self._validate_runtime_shape(tensor.shape, self._output_shape, "RIFE output")
        if tensor.shape[0] != 1:
            raise RIFEInterpolatorError("RIFE output must contain exactly one frame.")
        channel_index = 1 if self.output_layout == "nchw" else 3
        if tensor.shape[channel_index] != 3:
            raise RIFEInterpolatorError(
                f"RIFE output must contain 3 channels for {self.output_layout.upper()}."
            )
        if not np.isfinite(tensor).all():
            raise RIFEInterpolatorError("RIFE output contains NaN or infinite values.")
        image = tensor[0]
        if self.output_layout == "nchw":
            image = np.transpose(image, (1, 2, 0))
        if image.shape[:2] != (padded_height, padded_width):
            raise RIFEInterpolatorError(
                "RIFE output dimensions must match the padded input dimensions; "
                f"expected {padded_width}x{padded_height}, got "
                f"{image.shape[1]}x{image.shape[0]}."
            )
        image = image[:original_height, :original_width]
        image = np.rint(np.clip(image, 0.0, 1.0) * np.float32(255.0)).astype(
            np.uint8
        )
        return np.ascontiguousarray(image[..., ::-1])

    @staticmethod
    def _motion_summary(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[float, float]:
        """Return mean motion and the p95 threshold class used for bypass."""
        first = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        second = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        motion = cv2.absdiff(first, second)
        mean_motion = float(cv2.mean(motion)[0])
        pixels_above_two = cv2.countNonZero(
            cv2.compare(motion, 2, cv2.CMP_GT)
        )
        # p95 <= 2 exactly means no more than five percent of pixels exceed 2.
        p95_class = 2.0 if pixels_above_two * 20 <= motion.size else 3.0
        return mean_motion, p95_class

    @staticmethod
    def _active_motion_fraction(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Return the fraction of pixels with clearly visible endpoint motion."""
        difference = cv2.absdiff(frame_a, frame_b)
        motion = cv2.max(
            cv2.max(difference[..., 0], difference[..., 1]),
            difference[..., 2],
        )
        active = cv2.compare(motion, 12, cv2.CMP_GT)
        return float(cv2.countNonZero(active)) / float(active.size)

    @staticmethod
    def _match_midpoint_color(output, midpoint, reference_mask):
        """Correct small generated-frame colour shifts using stable regions."""
        if cv2.countNonZero(reference_mask) < 16:
            return output
        generated_mean = cv2.mean(output, mask=reference_mask)
        target_mean = cv2.mean(midpoint, mask=reference_mask)
        adjustment = tuple(
            float(np.clip(target_mean[index] - generated_mean[index], -4.0, 4.0))
            for index in range(3)
        ) + (0.0,)
        if max(abs(value) for value in adjustment[:3]) < 0.5:
            return output
        return cv2.add(output, adjustment)

    @classmethod
    def _confidence_composite(cls, output, frame_a, frame_b):
        """Use model output only where a cheap midpoint confidence check passes.

        All analysis is performed at inference resolution. The returned mask is
        later reused to restore full-resolution endpoint detail without another
        expensive full-size motion analysis.
        """
        difference = cv2.absdiff(frame_a, frame_b)
        motion = cv2.max(
            cv2.max(difference[..., 0], difference[..., 1]),
            difference[..., 2],
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        low_motion = cv2.erode(cv2.compare(motion, 12, cv2.CMP_LE), kernel)
        midpoint = cv2.addWeighted(frame_a, 0.5, frame_b, 0.5, 0.0)
        output = cls._match_midpoint_color(output, midpoint, low_motion)

        generated_difference = cv2.absdiff(output, midpoint)
        residual = cv2.max(
            cv2.max(generated_difference[..., 0], generated_difference[..., 1]),
            generated_difference[..., 2],
        )
        allowed_residual = cv2.convertScaleAbs(motion, alpha=0.75, beta=8.0)
        inconsistent = cv2.compare(residual, allowed_residual, cv2.CMP_GT)
        inconsistent = cv2.dilate(inconsistent, kernel)
        fallback = cv2.max(low_motion, inconsistent)
        cv2.copyTo(midpoint, fallback, output)
        fallback_fraction = float(cv2.countNonZero(fallback)) / float(fallback.size)
        return output, fallback_fraction, fallback

    @classmethod
    def _stabilize_low_motion(cls, output, frame_a, frame_b):
        """Backward-compatible wrapper for the confidence compositor."""
        output, fallback_fraction, _mask = cls._confidence_composite(
            output, frame_a, frame_b
        )
        return output, fallback_fraction

    @staticmethod
    def _restore_static_detail(output, frame_a, frame_b):
        """Preserve exact static texture without modifying moving regions."""
        difference = cv2.absdiff(frame_a, frame_b)
        stable = cv2.inRange(difference, (0, 0, 0), (1, 1, 1))
        stable = cv2.erode(stable, np.ones((3, 3), dtype=np.uint8))
        cv2.copyTo(frame_b, stable, output)
        return output

    def _fractional_flow_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        *,
        inference_pair: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """Compute a low-resolution bidirectional flow pair once per endpoint pair."""
        key = (id(frame_a), id(frame_b), frame_a.shape, frame_b.shape)
        if key == self._flow_cache_key and self._flow_cache is not None:
            return self._flow_cache, 0.0
        first, second = (
            inference_pair
            if inference_pair is not None
            else self._resize_pair_for_inference(frame_a, frame_b)
        )
        gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        gray_second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        started = time.perf_counter()
        flow_forward = cv2.calcOpticalFlowFarneback(
            gray_first, gray_second, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_backward = cv2.calcOpticalFlowFarneback(
            gray_second, gray_first, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._flow_cache_key = key
        self._flow_cache = (first, second, flow_forward, flow_backward)
        return self._flow_cache, elapsed_ms

    def interpolate_at(self, frame_a: Any, frame_b: Any, time_position: float) -> Any:
        """Synthesize a frame at a fractional time between real endpoints.

        The checked-in RIFE ONNX exports have no timestep input and can only
        infer the midpoint. For non-midpoint positions, use bidirectional
        Farneback flow at the configured inference resolution. This keeps the
        temporal locations real and reports flow work separately from neural
        inference.
        """
        if isinstance(time_position, bool) or not isinstance(time_position, (int, float)):
            raise TypeError("Interpolation time position must be numeric.")
        position = float(time_position)
        if not np.isfinite(position) or not 0.0 < position < 1.0:
            raise ValueError("Interpolation time position must be between zero and one.")
        if abs(position - 0.5) < 1e-9:
            self.last_generation_method = "rife_midpoint"
            return self.interpolate(frame_a, frame_b)
        if self._session is None:
            raise RIFEInterpolatorError("RIFEInterpolator is not initialized.")
        self._validate_frame_pair(frame_a, frame_b)
        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        first, second = self._resize_pair_for_inference(frame_a, frame_b)
        self.last_preprocessing_ms = (time.perf_counter() - preprocess_started) * 1000.0
        self.last_inference_ms = 0.0
        self.last_duplicate_bypass = False
        self.last_regional_result = None
        analysis = None
        if self.regional_transform and self.regional_controller is not None:
            analysis = self.regional_controller.analyze(first, second)
            selected = self.regional_controller.allocate(
                analysis,
                self.regional_budget_ms,
                elapsed_ms=analysis.analysis_ms,
            )
            if not np.any(selected):
                self.regional_controller.last_admission_reason = "no_eligible_tiles"
                self.last_regional_result = self.regional_controller.compose_fallback(
                    first, second, analysis=analysis, time_position=position
                )
                self.last_generation_method = "regional_temporal_fallback"
                self.last_flow_ms = 0.0
                self.last_inference_ms = 0.0
                self.last_interpolation_confidence = 0.0
                self.last_stabilized_fraction = 1.0
                fallback_started = time.perf_counter()
                output = cv2.addWeighted(
                    frame_a, 1.0 - position, frame_b, position, 0.0
                )
                endpoint_mask = cv2.resize(
                    self.last_regional_result.endpoint_fallback,
                    frame_a.shape[1::-1],
                    interpolation=cv2.INTER_NEAREST,
                )
                nearest_endpoint = frame_a if position < 0.5 else frame_b
                cv2.copyTo(nearest_endpoint, cv2.compare(endpoint_mask, 0.5, cv2.CMP_GE), output)
                self.last_postprocessing_ms = (
                    time.perf_counter() - fallback_started
                ) * 1000.0
                self.last_total_ms = (time.perf_counter() - total_started) * 1000.0
                self.input_dimensions = (first.shape[1], first.shape[0])
                self.padded_input_dimensions = self.input_dimensions
                self.output_dimensions = (output.shape[1], output.shape[0])
                return np.ascontiguousarray(output)

        (first, second, flow_forward, flow_backward), flow_ms = self._fractional_flow_pair(
            frame_a, frame_b, inference_pair=(first, second)
        )
        self.last_preprocessing_ms = max(
            0.0, (time.perf_counter() - preprocess_started) * 1000.0 - flow_ms
        )
        self.last_flow_ms = flow_ms
        self.last_generation_method = "farneback_flow"
        height, width = first.shape[:2]
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        map_first_x = x - position * flow_forward[..., 0]
        map_first_y = y - position * flow_forward[..., 1]
        map_second_x = x - (1.0 - position) * flow_backward[..., 0]
        map_second_y = y - (1.0 - position) * flow_backward[..., 1]
        if self.regional_transform and self.regional_controller is not None:
            warped_first_small = cv2.remap(
                first, map_first_x, map_first_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_second_small = cv2.remap(
                second, map_second_x, map_second_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            candidate_small = cv2.addWeighted(
                warped_first_small, 1.0 - position,
                warped_second_small, position, 0.0,
            )
            self.last_regional_result = self.regional_controller.compose_interpolation(
                candidate_small,
                first,
                second,
                analysis=analysis,
                budget_ms=self.regional_budget_ms,
                time_position=position,
            )
            regional_result = self.last_regional_result
            self.last_interpolation_confidence = float(
                np.mean(regional_result.confidence)
            )
            self.last_stabilized_fraction = 1.0 - self.last_interpolation_confidence
            if not np.any(regional_result.pixel_confidence > 0.0):
                # Avoid full-resolution flow remapping and float-image blending
                # when the confidence gate rejects every generated pixel.
                self.last_generation_method = "farneback_rejected"
                fallback_started = time.perf_counter()
                output = cv2.addWeighted(
                    frame_a, 1.0 - position, frame_b, position, 0.0
                )
                endpoint_mask = cv2.resize(
                    regional_result.endpoint_fallback,
                    frame_a.shape[1::-1],
                    interpolation=cv2.INTER_NEAREST,
                )
                nearest_endpoint = frame_a if position < 0.5 else frame_b
                cv2.copyTo(
                    nearest_endpoint,
                    cv2.compare(endpoint_mask, 0.5, cv2.CMP_GE),
                    output,
                )
                self.last_postprocessing_ms = (
                    time.perf_counter() - fallback_started
                ) * 1000.0
                self.last_total_ms = (time.perf_counter() - total_started) * 1000.0
                self.input_dimensions = (first.shape[1], first.shape[0])
                self.padded_input_dimensions = self.input_dimensions
                self.output_dimensions = (output.shape[1], output.shape[0])
                return np.ascontiguousarray(output)
        full_height, full_width = frame_a.shape[:2]
        if self.regional_transform and self.regional_controller is not None:
            # The regional candidate is needed only in admitted motion areas.
            # Upscale that small candidate once, then combine it with a sharp
            # full-resolution endpoint fallback below. Expanding the flow and
            # remapping both 720p endpoints cost more than 200 ms in live runs.
            candidate = cv2.resize(
                candidate_small,
                (full_width, full_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            if (full_width, full_height) != (width, height):
                flow_forward = cv2.resize(
                    flow_forward, (full_width, full_height), interpolation=cv2.INTER_LINEAR
                )
                flow_backward = cv2.resize(
                    flow_backward, (full_width, full_height), interpolation=cv2.INTER_LINEAR
                )
                flow_forward[..., 0] *= full_width / width
                flow_forward[..., 1] *= full_height / height
                flow_backward[..., 0] *= full_width / width
                flow_backward[..., 1] *= full_height / height
                height, width = full_height, full_width
                x, y = np.meshgrid(
                    np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
                )
                map_first_x = x - position * flow_forward[..., 0]
                map_first_y = y - position * flow_forward[..., 1]
                map_second_x = x - (1.0 - position) * flow_backward[..., 0]
                map_second_y = y - (1.0 - position) * flow_backward[..., 1]
            warped_first = cv2.remap(
                frame_a, map_first_x, map_first_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_second = cv2.remap(
                frame_b, map_second_x, map_second_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            candidate = cv2.addWeighted(
                warped_first, 1.0 - position, warped_second, position, 0.0
            )
        post_started = time.perf_counter()
        if self.regional_transform and self.regional_controller is not None:
            regional_result = self.last_regional_result
            assert regional_result is not None
            if regional_result.frame.shape[:2] != frame_a.shape[:2]:
                confidence = cv2.resize(
                    regional_result.pixel_confidence,
                    (frame_a.shape[1], frame_a.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )[..., None]
                endpoint_fallback = cv2.resize(
                    regional_result.endpoint_fallback,
                    (frame_a.shape[1], frame_a.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                temporal_blend = cv2.addWeighted(
                    frame_a, 1.0 - position, frame_b, position, 0.0
                )
                nearest_endpoint = frame_a if position < 0.5 else frame_b
                stable_fallback = cv2.blendLinear(
                    temporal_blend,
                    nearest_endpoint,
                    (1.0 - endpoint_fallback).astype(np.float32),
                    endpoint_fallback.astype(np.float32),
                )
                confidence = confidence[..., 0].astype(np.float32)
                candidate = cv2.blendLinear(
                    stable_fallback,
                    candidate,
                    (1.0 - confidence).astype(np.float32),
                    confidence,
                )
            else:
                candidate = regional_result.frame
            self.last_interpolation_confidence = float(
                np.mean(regional_result.confidence)
            )
            self.last_stabilized_fraction = 1.0 - self.last_interpolation_confidence
        else:
            self.last_interpolation_confidence = 0.0
            self.last_stabilized_fraction = 1.0
        if not self.regional_transform:
            candidate = self._restore_static_detail(candidate, frame_a, frame_b)
        self.last_preprocessing_ms = max(0.0, self.last_preprocessing_ms or 0.0)
        self.last_postprocessing_ms = (time.perf_counter() - post_started) * 1000.0
        self.last_total_ms = (time.perf_counter() - total_started) * 1000.0
        self.input_dimensions = (first.shape[1], first.shape[0])
        self.padded_input_dimensions = (first.shape[1], first.shape[0])
        self.output_dimensions = (candidate.shape[1], candidate.shape[0])
        return np.ascontiguousarray(candidate)

    def interpolate(self, frame_a: Any, frame_b: Any) -> Any:
        if (
            self._session is None
            or self._input_names is None
            or self._output_name is None
            or self._input_shapes is None
        ):
            raise RIFEInterpolatorError("RIFEInterpolator is not initialized.")
        self.last_preprocessing_ms = None
        self.last_inference_ms = None
        self.last_postprocessing_ms = None
        self.last_total_ms = None
        self.last_flow_ms = 0.0
        self.last_generation_method = "rife_midpoint"
        self.input_dimensions = None
        self.padded_input_dimensions = None
        self.output_dimensions = None
        self.last_stabilized_fraction = 0.0
        self.last_interpolation_confidence = 1.0
        self.last_duplicate_bypass = False
        self.last_regional_result = None
        self._validate_frame_pair(frame_a, frame_b)
        total_start = time.perf_counter()
        original_height, original_width = frame_a.shape[:2]
        preprocessing_start = time.perf_counter()
        inference_frame_a, inference_frame_b = self._resize_pair_for_inference(
            frame_a, frame_b
        )
        if self.temporal_stabilization and not self.regional_transform:
            # Reuse the already downscaled inference inputs for motion analysis.
            # An extra pair of full-resolution AREA resizes cost several ms per
            # frame pair at 720p and provided no additional useful signal.
            mean_motion, p95_motion = self._motion_summary(
                inference_frame_a, inference_frame_b
            )
            if mean_motion <= 0.75 and p95_motion <= 2.0:
                output = np.ascontiguousarray(frame_b.copy())
                self.last_preprocessing_ms = (
                    time.perf_counter() - preprocessing_start
                ) * 1000.0
                self.last_inference_ms = 0.0
                self.last_postprocessing_ms = 0.0
                self.last_total_ms = (time.perf_counter() - total_start) * 1000.0
                self.last_stabilized_fraction = 1.0
                self.last_interpolation_confidence = 1.0
                self.last_duplicate_bypass = True
                self.input_dimensions = (original_width, original_height)
                self.padded_input_dimensions = (original_width, original_height)
                self.output_dimensions = (original_width, original_height)
                return output
            # A nominally still game scene can contain water, particles, HUD
            # animation, or capture noise, so it will not satisfy the strict
            # duplicate test above. When that activity occupies only a small
            # part of the image, the confidence compositor would replace almost
            # all model pixels with the endpoint midpoint anyway. Produce that
            # stable midpoint directly and avoid an otherwise wasted inference.
            active_motion_fraction = self._active_motion_fraction(
                inference_frame_a, inference_frame_b
            )
            if mean_motion <= 3.0 and active_motion_fraction <= 0.08:
                output = cv2.addWeighted(frame_a, 0.5, frame_b, 0.5, 0.0)
                self.last_preprocessing_ms = (
                    time.perf_counter() - preprocessing_start
                ) * 1000.0
                self.last_inference_ms = 0.0
                self.last_postprocessing_ms = 0.0
                self.last_total_ms = (time.perf_counter() - total_start) * 1000.0
                self.last_stabilized_fraction = 1.0
                self.last_interpolation_confidence = 0.0
                self.last_duplicate_bypass = True
                self.input_dimensions = (original_width, original_height)
                self.padded_input_dimensions = (original_width, original_height)
                self.output_dimensions = (original_width, original_height)
                return np.ascontiguousarray(output)
        inference_height, inference_width = inference_frame_a.shape[:2]
        padded_height = ((inference_height + self._input_alignment - 1) // self._input_alignment) * self._input_alignment
        padded_width = ((inference_width + self._input_alignment - 1) // self._input_alignment) * self._input_alignment
        regional_analysis = None
        skip_regional_inference = False
        regional_model_executed = False
        if self.regional_transform:
            if self.regional_controller is None:
                raise RIFEInterpolatorError(
                    "Regional transformation controller is unavailable."
                )
            regional_analysis = self.regional_controller.analyze(
                inference_frame_a, inference_frame_b
            )
            preselected = self.regional_controller.allocate(
                regional_analysis,
                self.regional_budget_ms,
                elapsed_ms=regional_analysis.analysis_ms,
            )
            preselected &= regional_analysis.routes == RegionRoute.SMOOTH_MOTION
            skip_regional_inference = not self.regional_controller.should_run_model(
                regional_analysis,
                preselected,
                self.regional_budget_ms,
                elapsed_ms=regional_analysis.analysis_ms,
            )

        if skip_regional_inference:
            output = cv2.addWeighted(
                inference_frame_a, 0.5, inference_frame_b, 0.5, 0.0
            )
            self.last_preprocessing_ms = (
                time.perf_counter() - preprocessing_start
            ) * 1000.0
            self.last_inference_ms = 0.0
            self.last_duplicate_bypass = True
        else:
            tensor_a, tensor_b = self._prepare_input_pair(
                inference_frame_a,
                inference_frame_b,
                padded_height,
                padded_width,
            )
            self._validate_runtime_shape(
                tensor_a.shape,
                self._input_shapes[0],
                "RIFE input 1",
            )
            self._validate_runtime_shape(
                tensor_b.shape,
                self._input_shapes[1],
                "RIFE input 2",
            )
            self.last_preprocessing_ms = (
                time.perf_counter() - preprocessing_start
            ) * 1000.0
            try:
                inference_start = time.perf_counter()
                outputs = self._session.run(
                    [self._output_name],
                    {
                        self._input_names[0]: tensor_a,
                        self._input_names[1]: tensor_b,
                    },
                )
                self.last_inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000.0
                regional_model_executed = self.regional_transform
            except Exception as error:
                self.last_inference_ms = None
                raise RIFEInterpolatorError(
                    f"RIFE ONNX inference failed: {error}"
                ) from error
            if len(outputs) != 1:
                raise RIFEInterpolatorError(
                    f"RIFE ONNX inference returned {len(outputs)} outputs; expected one."
                )
            output = self._prepare_output(
                outputs[0],
                inference_height,
                inference_width,
                padded_height,
                padded_width,
            )
        postprocessing_start = time.perf_counter()
        fallback_mask = None
        regional_candidate = None
        regional_confidence = None
        regional_endpoint_fallback = None
        regional_scene_cut = False
        if self.regional_transform:
            if self.regional_controller is None:
                raise RIFEInterpolatorError(
                    "Regional transformation controller is unavailable."
                )
            regional_candidate = output
            if skip_regional_inference:
                self.last_regional_result = self.regional_controller.compose_fallback(
                    inference_frame_a,
                    inference_frame_b,
                    analysis=regional_analysis,
                )
            else:
                self.last_regional_result = self.regional_controller.compose_interpolation(
                    output,
                    inference_frame_a,
                    inference_frame_b,
                    analysis=regional_analysis,
                    budget_ms=self.regional_budget_ms,
                )
            output = self.last_regional_result.frame
            regional_confidence = self.last_regional_result.pixel_confidence
            regional_endpoint_fallback = (
                self.last_regional_result.endpoint_fallback
            )
            regional_scene_cut = bool(
                self.last_regional_result.provenance.size
                and np.all(
                    self.last_regional_result.provenance
                    == RegionProvenance.REAL
                )
            )
            self.last_stabilized_fraction = 1.0 - float(
                np.mean(self.last_regional_result.confidence)
            )
            self.last_interpolation_confidence = 1.0 - self.last_stabilized_fraction
        elif self.temporal_stabilization:
            # Stabilize at the model's working resolution. Doing this after
            # enlargement creates several full-size temporary images and can
            # add over 100 ms per generated frame at 720p.
            (
                output,
                self.last_stabilized_fraction,
                fallback_mask,
            ) = self._confidence_composite(
                output, inference_frame_a, inference_frame_b
            )
            self.last_interpolation_confidence = 1.0 - self.last_stabilized_fraction
        output_was_resized = output.shape[:2] != (original_height, original_width)
        if output_was_resized and regional_candidate is not None:
            if regional_scene_cut:
                output = frame_b.copy()
            elif skip_regional_inference:
                # When no neural pixels were admitted, construct the stable
                # fallback directly at source resolution. Resizing and then
                # reconstructing a low-resolution midpoint wasted 10-20 ms.
                if (
                    self._full_midpoint_buffer is None
                    or self._full_midpoint_buffer.shape != frame_a.shape
                ):
                    self._full_midpoint_buffer = np.empty_like(frame_a)
                output = cv2.addWeighted(
                    frame_a,
                    0.5,
                    frame_b,
                    0.5,
                    0.0,
                    dst=self._full_midpoint_buffer,
                )
                output = np.ascontiguousarray(output)
                fallback_mask = None
            else:
                # The expensive confidence composition has already happened at
                # inference resolution. Enlarge that result once, then restore
                # exact endpoint detail only where confidence is effectively
                # zero. This avoids several full-frame float RGB allocations.
                output = cv2.resize(
                    self.last_regional_result.frame,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                if (
                    self._full_midpoint_buffer is None
                    or self._full_midpoint_buffer.shape != frame_a.shape
                ):
                    self._full_midpoint_buffer = np.empty_like(frame_a)
                full_midpoint = cv2.addWeighted(
                    frame_a,
                    0.5,
                    frame_b,
                    0.5,
                    0.0,
                    dst=self._full_midpoint_buffer,
                )
                assert regional_confidence is not None
                assert regional_endpoint_fallback is not None
                if (
                    self._full_confidence_buffer is None
                    or self._full_confidence_buffer.shape != frame_a.shape[:2]
                ):
                    self._full_confidence_buffer = np.empty(
                        frame_a.shape[:2], dtype=np.float32
                    )
                full_confidence = cv2.resize(
                    regional_confidence,
                    (original_width, original_height),
                    dst=self._full_confidence_buffer,
                    interpolation=cv2.INTER_LINEAR,
                )
                if (
                    self._full_fallback_buffer is None
                    or self._full_fallback_buffer.shape != frame_a.shape[:2]
                ):
                    self._full_fallback_buffer = np.empty(
                        frame_a.shape[:2], dtype=np.uint8
                    )
                endpoint_mask = cv2.resize(
                    cv2.compare(
                        regional_endpoint_fallback,
                        0.5,
                        cv2.CMP_GE,
                    ),
                    (original_width, original_height),
                    dst=self._full_fallback_buffer,
                    interpolation=cv2.INTER_NEAREST,
                )
                if (
                    self._full_stable_buffer is None
                    or self._full_stable_buffer.shape != frame_a.shape
                ):
                    self._full_stable_buffer = np.empty_like(frame_a)
                np.copyto(self._full_stable_buffer, full_midpoint)
                cv2.copyTo(frame_b, endpoint_mask, self._full_stable_buffer)
                fallback_mask = cv2.compare(
                    full_confidence,
                    self.regional_controller.config.confidence_floor,
                    cv2.CMP_LE,
                    dst=self._full_fallback_buffer,
                )
                cv2.copyTo(self._full_stable_buffer, fallback_mask, output)
                output = np.ascontiguousarray(output)
        elif output_was_resized:
            output = cv2.resize(
                output,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
            output = np.ascontiguousarray(output)
        if output_was_resized and fallback_mask is not None:
            # Reconstruct uncertain regions from the original-resolution real
            # endpoints. This avoids the visible sharp/soft alternation caused
            # by enlarging every generated pixel from the small model input.
            if (
                self._full_midpoint_buffer is None
                or self._full_midpoint_buffer.shape != frame_a.shape
            ):
                self._full_midpoint_buffer = np.empty_like(frame_a)
            if (
                self._full_fallback_buffer is None
                or self._full_fallback_buffer.shape != frame_a.shape[:2]
            ):
                self._full_fallback_buffer = np.empty(
                    frame_a.shape[:2], dtype=np.uint8
                )
            full_midpoint = cv2.addWeighted(
                frame_a,
                0.5,
                frame_b,
                0.5,
                0.0,
                dst=self._full_midpoint_buffer,
            )
            full_fallback = cv2.resize(
                fallback_mask,
                (original_width, original_height),
                dst=self._full_fallback_buffer,
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.copyTo(full_midpoint, full_fallback, output)
        elif output_was_resized:
            output = self._restore_static_detail(output, frame_a, frame_b)
        self.last_postprocessing_ms = (
            time.perf_counter() - postprocessing_start
        ) * 1000.0
        self.last_total_ms = (time.perf_counter() - total_start) * 1000.0
        if regional_model_executed and self.regional_controller is not None:
            self.regional_controller.record_inference_cost(self.last_total_ms)
        self._profile_samples.append((self.last_preprocessing_ms, self.last_inference_ms,
                                      self.last_postprocessing_ms, self.last_total_ms,
                                      self.last_interpolation_confidence))
        if time.perf_counter() - self._profile_started_at >= 5.0:
            samples = self._profile_samples
            means = [sum(row[i] for row in samples) / len(samples) for i in range(5)]
            logging.getLogger(__name__).info(
                "RIFE stages (%s calls, %sx%s inference): preprocess %.2f ms, "
                "inference %.2f ms, postprocess %.2f ms, total %.2f ms, "
                "confidence %.1f%%",
                len(samples), inference_width, inference_height,
                *means[:4], means[4] * 100.0)
            self._profile_samples = []
            self._profile_started_at = time.perf_counter()
        self.input_dimensions = (inference_width, inference_height)
        self.padded_input_dimensions = (padded_width, padded_height)
        self.output_dimensions = (output.shape[1], output.shape[0])
        return output

    def shutdown(self) -> None:
        self._validation_session_lease.release()
        self._session = None
        self.available_providers = ()
        self.active_providers = ()
        self.input_metadata = ()
        self.output_metadata = None
        self._input_names = None
        self._output_name = None
        self._input_shapes = None
        self._output_shape = None
        self.last_preprocessing_ms = None
        self.last_inference_ms = None
        self.last_postprocessing_ms = None
        self.last_total_ms = None
        self.last_flow_ms = 0.0
        self.last_generation_method = "rife_midpoint"
        self.input_dimensions = None
        self.padded_input_dimensions = None
        self.output_dimensions = None
        self.last_stabilized_fraction = 0.0
        self.last_interpolation_confidence = 1.0
        self._inference_frame_buffers = None
        self._input_tensor_buffers = None
        self._full_midpoint_buffer = None
        self._full_fallback_buffer = None
        self._full_confidence_buffer = None
        self._full_stable_buffer = None
        self.last_duplicate_bypass = False
        self.last_regional_result = None
        self._flow_cache_key = None
        self._flow_cache = None
        if self.regional_controller is not None:
            self.regional_controller.reset()

"""Tests for frame-interpolation lifecycle and RIFE ONNX infrastructure."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

import frame_interpolator
from frame_interpolator import (
    FrameInterpolator,
    FrameInterpolatorError,
    NoOpInterpolator,
    RIFEInterpolator,
    RIFEInterpolatorError,
)


class _FakeSession:
    def __init__(
        self,
        model_path: str,
        sess_options=None,
        providers=None,
        *,
        input_shapes=None,
        output_shape=None,
        tensor_type: str = "tensor(float)",
        active_providers=None,
    ) -> None:
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers or ["CPUExecutionProvider"]
        self.input_shapes = input_shapes or [
            [1, 3, "height", "width"],
            [1, 3, "height", "width"],
        ]
        self.output_shape = output_shape or [1, 3, "height", "width"]
        self.tensor_type = tensor_type
        self.active_providers = active_providers
        if self.active_providers is None:
            self.active_providers = [
                provider[0] if isinstance(provider, tuple) else provider
                for provider in self.providers
            ]

    def get_inputs(self):
        return [
            SimpleNamespace(
                name=f"frame_{index}",
                shape=shape,
                type=self.tensor_type,
            )
            for index, shape in enumerate(self.input_shapes)
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="interpolated",
                shape=self.output_shape,
                type=self.tensor_type,
            )
        ]

    def get_providers(self):
        return self.active_providers

    def run(self, output_names, input_feed):
        del output_names
        frame_a = input_feed["frame_0"]
        frame_b = input_feed["frame_1"]
        return [(frame_a + frame_b) * np.float32(0.5)]


def test_noop_interpolator_lifecycle_and_identity_behavior() -> None:
    interpolator = NoOpInterpolator()
    assert isinstance(interpolator, FrameInterpolator)
    with pytest.raises(FrameInterpolatorError, match="not initialized"):
        interpolator.interpolate("a", "b")

    interpolator.initialize()
    interpolator.initialize()
    frame_b = object()
    assert interpolator.interpolate(object(), frame_b) is frame_b

    interpolator.shutdown()
    interpolator.shutdown()
    assert interpolator.initialized is False


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_rife_initialization_validates_layout_and_records_metadata(
    monkeypatch,
    tmp_path: Path,
    layout: str,
) -> None:
    model_path = tmp_path / f"rife_{layout}.onnx"
    model_path.touch()
    sessions = []
    shape = [1, 3, "height", "width"]
    if layout == "nhwc":
        shape = [1, "height", "width", 3]

    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    def create_session(path, sess_options, providers):
        session = _FakeSession(
            path,
            sess_options,
            providers,
            input_shapes=[shape, shape],
            output_shape=shape,
        )
        sessions.append(session)
        return session

    interpolator = RIFEInterpolator(
        model_path,
        input_layout=layout,
        output_layout=layout,
        session_factory=create_session,
    )
    interpolator.initialize()
    interpolator.initialize()

    assert len(sessions) == 1
    assert sessions[0].providers == ["CPUExecutionProvider"]
    assert interpolator.active_providers == ("CPUExecutionProvider",)
    assert len(interpolator.input_metadata) == 2
    assert interpolator.input_metadata[0].layout == layout
    assert interpolator.output_metadata is not None
    assert interpolator.output_metadata.layout == layout


def test_rife_interpolates_matching_frames_after_initialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rife.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    interpolator = RIFEInterpolator(model_path, session_factory=_FakeSession)
    interpolator.initialize()
    frame_a = np.zeros((4, 6, 3), dtype=np.uint8)
    frame_b = np.full((4, 6, 3), 255, dtype=np.uint8)

    output = interpolator.interpolate(frame_a, frame_b)

    assert output.shape == frame_a.shape
    assert output.dtype == np.uint8
    assert np.all(output == 128)
    assert interpolator.input_dimensions == (6, 4)
    assert interpolator.padded_input_dimensions == (32, 32)
    assert interpolator.output_dimensions == (6, 4)
    assert interpolator.last_inference_ms is not None


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_reusable_input_pair_matches_reference_conversion(layout):
    interpolator = RIFEInterpolator(input_layout=layout)
    rng = np.random.default_rng(42)
    frame_a = rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)
    frame_b = rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)

    tensor_a, tensor_b = interpolator._prepare_input_pair(
        frame_a, frame_b, padded_height=16, padded_width=32
    )

    np.testing.assert_allclose(
        tensor_a,
        interpolator._prepare_input(frame_a, 16, 32),
        rtol=0.0,
        atol=np.finfo(np.float32).eps,
    )
    np.testing.assert_allclose(
        tensor_b,
        interpolator._prepare_input(frame_b, 16, 32),
        rtol=0.0,
        atol=np.finfo(np.float32).eps,
    )
    assert tensor_a.flags.c_contiguous
    assert tensor_b.flags.c_contiguous


def test_frame_stage_scratch_buffers_are_reused_and_cleared_on_shutdown():
    interpolator = RIFEInterpolator(inference_width=16, inference_height=9)
    frame_a = np.zeros((72, 128, 3), dtype=np.uint8)
    frame_b = np.full_like(frame_a, 255)

    resized_a, resized_b = interpolator._resize_pair_for_inference(frame_a, frame_b)
    tensor_a, tensor_b = interpolator._prepare_input_pair(
        resized_a, resized_b, padded_height=32, padded_width=32
    )
    first_ids = tuple(map(id, (resized_a, resized_b, tensor_a, tensor_b)))

    resized_a, resized_b = interpolator._resize_pair_for_inference(frame_b, frame_a)
    tensor_a, tensor_b = interpolator._prepare_input_pair(
        resized_a, resized_b, padded_height=32, padded_width=32
    )
    assert tuple(map(id, (resized_a, resized_b, tensor_a, tensor_b))) == first_ids
    assert np.all(tensor_a[:, :, 9:, :] == 0.0)
    assert np.all(tensor_b[:, :, 9:, :] == 0.0)

    interpolator.shutdown()
    assert interpolator._inference_frame_buffers is None
    assert interpolator._input_tensor_buffers is None
    assert interpolator._full_midpoint_buffer is None
    assert interpolator._full_fallback_buffer is None


def test_rife_warmup_compiles_configured_internal_resolution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rife.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    sessions = []

    class CountingSession(_FakeSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.run_calls = 0

        def run(self, output_names, input_feed):
            self.run_calls += 1
            return super().run(output_names, input_feed)

    def create_session(*args, **kwargs):
        session = CountingSession(*args, **kwargs)
        sessions.append(session)
        return session

    interpolator = RIFEInterpolator(
        model_path,
        session_factory=create_session,
        inference_width=16,
        inference_height=16,
    )
    interpolator.initialize()
    try:
        assert interpolator.warmup(iterations=3) >= 0.0
        assert sessions[0].run_calls == 3
        assert interpolator.last_inference_ms is None
        with pytest.raises(ValueError, match="positive"):
            interpolator.warmup(iterations=0)
    finally:
        interpolator.shutdown()


def test_rife_optional_internal_resolution_restores_presented_dimensions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rife.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    interpolator = RIFEInterpolator(
        model_path,
        inference_width=320,
        inference_height=180,
        session_factory=_FakeSession,
    )
    interpolator.initialize()
    frame_a = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_b = np.full_like(frame_a, 255)

    output = interpolator.interpolate(frame_a, frame_b)

    assert output.shape == frame_a.shape
    assert output.dtype == np.uint8
    assert interpolator.input_dimensions == (320, 180)
    assert interpolator.padded_input_dimensions == (320, 192)
    assert interpolator.output_dimensions == (1280, 720)


def test_rife_rejects_mismatched_runtime_input_sizes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rife.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    interpolator = RIFEInterpolator(model_path, session_factory=_FakeSession)
    interpolator.initialize()

    with pytest.raises(RIFEInterpolatorError, match="dimensions must match"):
        interpolator.interpolate(
            np.zeros((4, 6, 3), dtype=np.uint8),
            np.zeros((4, 7, 3), dtype=np.uint8),
        )


def test_rife_missing_and_invalid_model_paths_are_clear(tmp_path: Path) -> None:
    with pytest.raises(RIFEInterpolatorError, match="requires an ONNX model path"):
        RIFEInterpolator().initialize()
    with pytest.raises(RIFEInterpolatorError, match="does not exist"):
        RIFEInterpolator(tmp_path / "missing.onnx").initialize()


def test_rife_invalid_onnx_load_error_is_clear(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "invalid.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    def reject_invalid_model(path, sess_options, providers):
        del path, sess_options, providers
        raise RuntimeError("invalid ONNX graph")

    with pytest.raises(RIFEInterpolatorError, match="Unable to load.*invalid ONNX"):
        RIFEInterpolator(
            model_path,
            session_factory=reject_invalid_model,
        ).initialize()


@pytest.mark.parametrize(
    ("session_kwargs", "error_pattern"),
    [
        ({"input_shapes": [[1, 3, "h", "w"]]}, "exactly two image inputs"),
        (
            {
                "input_shapes": [[1, 4, "h", "w"], [1, 3, "h", "w"]],
            },
            "exactly 3 image channels",
        ),
        (
            {"input_shapes": [[1, 3, "h"], [1, 3, "h", "w"]]},
            "rank-4",
        ),
        ({"tensor_type": "tensor(uint8)"}, "float32"),
        (
            {
                "input_shapes": [[1, 3, 8, 12], [1, 3, 8, 10]],
                "output_shape": [1, 3, 8, 12],
            },
            "input spatial dimensions must match",
        ),
        (
            {
                "input_shapes": [[1, 3, 8, 12], [1, 3, 8, 12]],
                "output_shape": [1, 3, 16, 24],
            },
            "input/output spatial dimensions must match",
        ),
    ],
)
def test_rife_rejects_unsupported_model_metadata(
    monkeypatch,
    tmp_path: Path,
    session_kwargs,
    error_pattern: str,
) -> None:
    model_path = tmp_path / "invalid.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    def create_session(path, sess_options, providers):
        return _FakeSession(path, sess_options, providers, **session_kwargs)

    with pytest.raises(RIFEInterpolatorError, match=error_pattern):
        RIFEInterpolator(
            model_path,
            session_factory=create_session,
        ).initialize()


def test_rife_directml_provider_selection_and_session_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rife.onnx"
    model_path.touch()
    sessions = []
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    def create_session(path, sess_options, providers):
        session = _FakeSession(
            path,
            sess_options,
            providers,
            active_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        sessions.append(session)
        return session

    interpolator = RIFEInterpolator(
        model_path,
        provider="directml",
        device_id=1,
        session_factory=create_session,
    )
    interpolator.initialize()

    assert sessions[0].providers == [("DmlExecutionProvider", {"device_id": 1})]
    assert sessions[0].sess_options.intra_op_num_threads == 1
    assert sessions[0].sess_options.enable_mem_pattern is False
    assert (
        sessions[0].sess_options.execution_mode
        == frame_interpolator.ort.ExecutionMode.ORT_SEQUENTIAL
    )
    assert interpolator.active_providers[0] == "DmlExecutionProvider"


def test_rife_directml_unavailable_and_silent_fallback_are_rejected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rife.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    with pytest.raises(RIFEInterpolatorError, match="is unavailable"):
        RIFEInterpolator(model_path, provider="directml").initialize()

    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    def create_fallback_session(path, sess_options, providers):
        return _FakeSession(
            path,
            sess_options,
            providers,
            active_providers=["CPUExecutionProvider"],
        )

    with pytest.raises(RIFEInterpolatorError, match="is not active"):
        RIFEInterpolator(
            model_path,
            provider="directml",
            session_factory=create_fallback_session,
        ).initialize()


def test_rife_shutdown_is_idempotent_and_clears_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "rife.onnx"
    model_path.touch()
    monkeypatch.setattr(
        frame_interpolator.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    interpolator = RIFEInterpolator(model_path, session_factory=_FakeSession)
    interpolator.initialize()

    interpolator.shutdown()
    interpolator.shutdown()

    assert interpolator.initialized is False
    assert interpolator.active_providers == ()
    assert interpolator.input_metadata == ()
    assert interpolator.output_metadata is None
    assert interpolator.last_inference_ms is None
    assert interpolator.output_dimensions is None


def test_rife_rejects_invalid_layout_provider_and_device() -> None:
    with pytest.raises(ValueError, match="input layout"):
        RIFEInterpolator("model.onnx", input_layout="chw")
    with pytest.raises(ValueError, match="execution provider"):
        RIFEInterpolator("model.onnx", provider="cuda")
    with pytest.raises(ValueError, match="zero or greater"):
        RIFEInterpolator("model.onnx", provider="directml", device_id=-1)
    with pytest.raises(ValueError, match="both be set"):
        RIFEInterpolator("model.onnx", inference_width=320)
    with pytest.raises(ValueError, match="positive integer"):
        RIFEInterpolator("model.onnx", inference_width=0, inference_height=180)
    with pytest.raises(TypeError, match="boolean"):
        RIFEInterpolator("model.onnx", temporal_stabilization="yes")
    with pytest.raises(TypeError, match="boolean"):
        RIFEInterpolator("model.onnx", regional_transform="yes")
    with pytest.raises(ValueError, match="positive"):
        RIFEInterpolator("model.onnx", regional_budget_ms=0.0)


def test_static_detail_restoration_preserves_texture_without_touching_motion():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    a[::2, ::2] = 255
    b = a.copy()
    b[:, 20:] = 127
    generated = np.full_like(a, 80)
    restored = RIFEInterpolator._restore_static_detail(generated, a, b)
    np.testing.assert_array_equal(restored[:, :18], b[:, :18])
    assert np.all(restored[:, 21:] == 80)


def test_temporal_stabilization_bypasses_effectively_duplicate_frames(tmp_path):
    model = tmp_path / "rife.onnx"
    model.touch()
    r = RIFEInterpolator(
        model, session_factory=_FakeSession, temporal_stabilization=True
    )
    r.initialize()
    frame = np.full((64, 64, 3), 80, dtype=np.uint8)
    try:
        np.testing.assert_array_equal(r.interpolate(frame, frame), frame)
        assert r.last_duplicate_bypass is True
        assert r.last_inference_ms == 0.0
        assert r.last_stabilized_fraction == 1.0
    finally:
        r.shutdown()


def test_motion_summary_preserves_five_percent_duplicate_boundary():
    a = np.zeros((20, 20, 3), dtype=np.uint8)
    b = a.copy()
    b.reshape(-1, 3)[:20] = 3
    mean_motion, p95_class = RIFEInterpolator._motion_summary(a, b)
    assert mean_motion > 0.0
    assert p95_class == 2.0

    b.reshape(-1, 3)[20] = 3
    _mean_motion, p95_class = RIFEInterpolator._motion_summary(a, b)
    assert p95_class == 3.0


def test_active_motion_fraction_counts_visible_endpoint_changes():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[:2, :5] = 13

    assert RIFEInterpolator._active_motion_fraction(a, b) == pytest.approx(0.1)


def test_temporal_stabilization_bypasses_sparse_ambient_motion(tmp_path):
    model = tmp_path / "rife.onnx"
    model.write_bytes(b"model")
    r = RIFEInterpolator(
        model,
        session_factory=_FakeSession,
        temporal_stabilization=True,
    )
    r.initialize()
    try:
        a = np.full((64, 64, 3), 80, dtype=np.uint8)
        b = a.copy()
        b[:10, :20] = 100  # 4.88% localized animation, not a strict duplicate.

        result = r.interpolate(a, b)

        np.testing.assert_array_equal(result, cv2.addWeighted(a, 0.5, b, 0.5, 0.0))
        assert r.last_duplicate_bypass is True
        assert r.last_inference_ms == 0.0
        assert r.last_interpolation_confidence == 0.0
    finally:
        r.shutdown()


def test_temporal_stabilization_keeps_inference_for_camera_motion(tmp_path):
    model = tmp_path / "rife.onnx"
    model.write_bytes(b"model")
    r = RIFEInterpolator(
        model,
        session_factory=_FakeSession,
        temporal_stabilization=True,
    )
    r.initialize()
    try:
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.full_like(a, 20)

        r.interpolate(a, b)

        assert r.last_duplicate_bypass is False
        assert r.last_inference_ms is not None
    finally:
        r.shutdown()


def test_temporal_stabilization_preserves_motion_and_clamps_low_motion():
    a = np.full((32, 32, 3), 100, dtype=np.uint8)
    b = np.full_like(a, 102)
    b[:, 20:] = 220
    generated = np.full_like(a, 180)
    stabilized, fraction = RIFEInterpolator._stabilize_low_motion(generated, a, b)
    np.testing.assert_array_equal(stabilized[:, :18], 101)
    assert np.all(stabilized[:, 21:] == 176)
    assert 0.4 < fraction < 0.8


def test_confidence_compositor_falls_back_from_inconsistent_generation():
    a = np.zeros((24, 24, 3), dtype=np.uint8)
    b = np.full_like(a, 50)
    generated = np.full_like(a, 255)

    composited, fallback_fraction, fallback = (
        RIFEInterpolator._confidence_composite(generated, a, b)
    )

    np.testing.assert_array_equal(composited, 25)
    assert fallback_fraction == 1.0
    assert np.count_nonzero(fallback) == fallback.size


def test_confidence_compositor_matches_small_generated_colour_shift():
    a = np.zeros((24, 24, 3), dtype=np.uint8)
    b = np.zeros_like(a)
    b[:, 12:] = 100
    generated = np.full_like(a, 55)

    composited, fallback_fraction, _fallback = (
        RIFEInterpolator._confidence_composite(generated, a, b)
    )

    np.testing.assert_array_equal(composited[:, :11], 0)
    np.testing.assert_array_equal(composited[:, 13:], 51)
    assert 0.4 < fallback_fraction < 0.6


def test_temporal_stabilization_runs_before_full_resolution_resize(
    monkeypatch, tmp_path
):
    model = tmp_path / "rife.onnx"
    model.touch()
    r = RIFEInterpolator(
        model,
        session_factory=_FakeSession,
        inference_width=16,
        inference_height=16,
        temporal_stabilization=True,
    )
    r.initialize()
    observed_shapes = []
    motion_shapes = []
    composite = RIFEInterpolator._confidence_composite

    def record_shape(output, frame_a, frame_b):
        observed_shapes.append((output.shape, frame_a.shape, frame_b.shape))
        return composite(output, frame_a, frame_b)

    monkeypatch.setattr(
        RIFEInterpolator, "_confidence_composite", staticmethod(record_shape)
    )

    def record_motion_shape(frame_a, frame_b):
        motion_shapes.append((frame_a.shape, frame_b.shape))
        return 10.0, 50.0

    monkeypatch.setattr(
        RIFEInterpolator, "_motion_summary", staticmethod(record_motion_shape)
    )
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = a.copy()
    b[:, 32:] = 200
    try:
        output = r.interpolate(a, b)
        assert output.shape == a.shape
        assert motion_shapes == [((16, 16, 3),) * 2]
        assert observed_shapes == [((16, 16, 3),) * 3]
    finally:
        r.shutdown()


def test_regional_transform_runs_at_inference_resolution(tmp_path):
    model = tmp_path / "rife.onnx"
    model.touch()
    interpolator = RIFEInterpolator(
        model,
        session_factory=_FakeSession,
        inference_width=32,
        inference_height=32,
        regional_transform=True,
    )
    interpolator.initialize()
    first = np.zeros((96, 128, 3), dtype=np.uint8)
    second = first.copy()
    cv2.rectangle(second, (48, 24), (88, 72), (180, 120, 80), -1)
    try:
        output = interpolator.interpolate(first, second)
        assert output.shape == first.shape
        assert interpolator.last_regional_result is not None
        assert interpolator.last_regional_result.analysis_ms >= 0.0
        assert interpolator.last_regional_result.provenance_counts
        assert 0.0 <= interpolator.last_interpolation_confidence <= 1.0
    finally:
        interpolator.shutdown()


def test_regional_full_resolution_fallback_restores_exact_static_detail(tmp_path):
    model = tmp_path / "rife.onnx"
    model.touch()
    interpolator = RIFEInterpolator(
        model,
        session_factory=_FakeSession,
        inference_width=16,
        inference_height=16,
        regional_transform=True,
    )
    interpolator.initialize()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[::2, ::2] = 255
    try:
        output = interpolator.interpolate(frame, frame.copy())
        np.testing.assert_array_equal(output, frame)
        assert interpolator.last_inference_ms == 0.0
        assert interpolator._full_midpoint_buffer is not None
    finally:
        interpolator.shutdown()


def test_regional_transform_skips_model_when_no_tile_is_temporally_eligible(
    tmp_path,
):
    model = tmp_path / "rife.onnx"
    model.touch()
    sessions = []

    class CountingSession(_FakeSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.run_calls = 0

        def run(self, output_names, input_feed):
            self.run_calls += 1
            return super().run(output_names, input_feed)

    def factory(*args, **kwargs):
        session = CountingSession(*args, **kwargs)
        sessions.append(session)
        return session

    interpolator = RIFEInterpolator(
        model,
        session_factory=factory,
        inference_width=32,
        inference_height=32,
        regional_transform=True,
    )
    interpolator.initialize()
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    cv2.putText(frame, "HUD", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,) * 3, 2)
    try:
        output = interpolator.interpolate(frame, frame.copy())
        np.testing.assert_array_equal(output, frame)
        assert sessions[0].run_calls == 0
        assert interpolator.last_inference_ms == 0.0
        assert interpolator.last_duplicate_bypass
    finally:
        interpolator.shutdown()


def test_regional_transform_learns_model_cost_then_enforces_deadline(tmp_path):
    model = tmp_path / "rife.onnx"
    model.touch()
    sessions = []

    class SlowCountingSession(_FakeSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.run_calls = 0

        def run(self, output_names, input_feed):
            self.run_calls += 1
            time.sleep(0.150)
            return super().run(output_names, input_feed)

    def factory(*args, **kwargs):
        session = SlowCountingSession(*args, **kwargs)
        sessions.append(session)
        return session

    interpolator = RIFEInterpolator(
        model,
        session_factory=factory,
        inference_width=32,
        inference_height=32,
        regional_transform=True,
        regional_budget_ms=100.0,
    )
    interpolator.initialize()
    first = np.full((64, 64, 3), 20, dtype=np.uint8)
    second = np.full_like(first, 25)
    try:
        interpolator.interpolate(first, second)
        assert sessions[0].run_calls == 1
        assert interpolator.regional_controller is not None
        assert interpolator.regional_controller.predicted_inference_cost_ms >= 150.0

        interpolator.interpolate(first, second)
        assert sessions[0].run_calls == 1
        assert interpolator.last_inference_ms == 0.0
        assert interpolator.last_duplicate_bypass
    finally:
        interpolator.shutdown()


def test_confidence_fallback_restores_full_resolution_static_texture(tmp_path):
    model = tmp_path / "rife.onnx"
    model.touch()
    r = RIFEInterpolator(
        model,
        session_factory=_FakeSession,
        inference_width=16,
        inference_height=16,
        temporal_stabilization=True,
    )
    r.initialize()
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[::2, ::2] = 255
    b = a.copy()
    b[:, 40:] = 180
    try:
        output = r.interpolate(a, b)
        np.testing.assert_array_equal(output[:, :28], a[:, :28])
        assert 0.0 < r.last_stabilized_fraction < 1.0
        assert r.last_interpolation_confidence == pytest.approx(
            1.0 - r.last_stabilized_fraction
        )
    finally:
        r.shutdown()


def test_downscaled_rife_preserves_static_full_resolution_pixels(tmp_path):
    model = tmp_path / "rife.onnx"
    model.touch()
    r = RIFEInterpolator(model, session_factory=_FakeSession,
                         inference_width=16, inference_height=16)
    r.initialize()
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[::2, ::2] = 255
    try:
        np.testing.assert_array_equal(r.interpolate(a, a), a)
    finally:
        r.shutdown()


def test_rife_total_timing_covers_all_reported_stages(tmp_path):
    model=tmp_path/"rife.onnx"; model.touch()
    r=RIFEInterpolator(model,session_factory=_FakeSession,
                       inference_width=16,inference_height=16)
    r.initialize()
    try:
        image=np.zeros((32,32,3),dtype=np.uint8)
        r.interpolate(image,image)
        assert r.last_total_ms >= r.last_preprocessing_ms + r.last_inference_ms + r.last_postprocessing_ms
    finally:r.shutdown()


def test_model_metadata_controls_padding_alignment(tmp_path):
    model=tmp_path/"rife.onnx";model.touch()
    class AlignedSession(_FakeSession):
        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={"urfts.input_alignment":"128"})
    r=RIFEInterpolator(model,session_factory=AlignedSession)
    r.initialize()
    try:
        image=np.zeros((180,320,3),dtype=np.uint8)
        assert r.interpolate(image,image).shape==image.shape
        assert r.padded_input_dimensions==(384,256)
    finally:r.shutdown()


def test_ifrnet_alignment_metadata_accepts_sixteen_pixel_padding(tmp_path):
    model=tmp_path/"ifrnet.onnx";model.touch()
    class IFRNetSession(_FakeSession):
        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={"urfts.input_alignment":"16"})
    r=RIFEInterpolator(model,session_factory=IFRNetSession)
    r.initialize()
    try:
        image=np.zeros((90,160,3),dtype=np.uint8)
        assert r.interpolate(image,image).shape==image.shape
        assert r.padded_input_dimensions==(160,96)
    finally:r.shutdown()

"""Unit tests for explainable regional frame routing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from regional_transform import (
    RegionProvenance,
    RegionRoute,
    RegionalConfig,
    RegionalTransformController,
)


def _textured_frame(height: int = 128, width: int = 192) -> np.ndarray:
    y, x = np.indices((height, width))
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (x * 3 + y) % 256
    frame[..., 1] = (x + y * 2) % 256
    frame[..., 2] = (x * 2 + y * 3) % 256
    return frame


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RegionalConfig(tile_size=2)
    with pytest.raises(ValueError):
        RegionalConfig(analysis_scale=0)


def test_analysis_routes_static_detail_without_scene_cut() -> None:
    first = _textured_frame()
    second = first.copy()
    controller = RegionalTransformController(
        RegionalConfig(tile_size=32, analysis_scale=4, text_threshold=1.0)
    )

    analysis = controller.analyze(first, second)

    assert not analysis.scene_cut
    assert np.count_nonzero(analysis.routes == RegionRoute.STATIC_DETAIL) > 0
    assert analysis.analysis_shape == (32, 48)
    assert analysis.analysis_ms >= 0.0


def test_analysis_detects_global_scene_cut() -> None:
    first = np.zeros((96, 128, 3), dtype=np.uint8)
    second = np.full_like(first, 255)
    analysis = RegionalTransformController().analyze(first, second)
    assert analysis.scene_cut
    assert np.all(analysis.routes == RegionRoute.SCENE_CUT)


def test_hud_like_region_is_kept_out_of_temporal_route() -> None:
    first = np.full((128, 192, 3), 32, dtype=np.uint8)
    cv2.putText(first, "FPS 60", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,) * 3, 2)
    second = first.copy()
    controller = RegionalTransformController(
        RegionalConfig(tile_size=32, analysis_scale=2, text_threshold=0.035)
    )
    analysis = controller.analyze(first, second)
    assert np.count_nonzero(analysis.routes == RegionRoute.HUD_TEXT) > 0


def test_coherent_camera_pan_is_admitted_to_temporal_model() -> None:
    first = _textured_frame(128, 192)
    transform = np.float32([[1, 0, 6], [0, 1, 2]])
    second = cv2.warpAffine(
        first,
        transform,
        (first.shape[1], first.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    controller = RegionalTransformController(
        RegionalConfig(tile_size=16, analysis_scale=2)
    )

    analysis = controller.analyze(first, second)
    selected = controller.allocate(
        analysis, budget_ms=34.0, elapsed_ms=analysis.analysis_ms
    )

    assert analysis.coherent_camera_motion
    assert analysis.global_motion_pixels >= 0.5
    assert analysis.global_motion_response >= 0.15
    assert np.count_nonzero(analysis.routes == RegionRoute.SMOOTH_MOTION) > 0
    assert controller.should_run_model(
        analysis,
        selected,
        34.0,
        elapsed_ms=analysis.analysis_ms,
    )
    assert controller.last_admission_reason == "model_camera_motion"


def test_moving_high_frequency_scene_is_not_mislabeled_as_hud() -> None:
    first = _textured_frame(128, 192)
    second = np.roll(first, 8, axis=1)
    controller = RegionalTransformController(
        RegionalConfig(tile_size=16, analysis_scale=2)
    )

    analysis = controller.analyze(first, second)

    assert np.count_nonzero(analysis.routes == RegionRoute.HUD_TEXT) == 0


def test_deadline_allocator_uses_highest_utility_tiles_first() -> None:
    frame = _textured_frame(64, 128)
    controller = RegionalTransformController(
        RegionalConfig(tile_size=32, analysis_scale=1)
    )
    analysis = controller.analyze(frame, frame.copy())
    analysis.routes.fill(RegionRoute.SMOOTH_MOTION)
    analysis.utility[:] = np.array([[1.0, 4.0, 3.0, 2.0], [8.0, 7.0, 6.0, 5.0]])
    selected = controller.allocate(analysis, budget_ms=2.1)
    assert np.count_nonzero(selected) == 2
    assert selected[1, 0]
    assert selected[1, 1]


def test_whole_model_cost_prediction_can_reject_a_missed_deadline() -> None:
    controller = RegionalTransformController(RegionalConfig(cost_alpha=0.5))
    assert controller.can_afford_inference(16.0, elapsed_ms=2.0)
    controller.record_inference_cost(20.0)
    assert controller.predicted_inference_cost_ms == pytest.approx(20.0)
    assert not controller.can_afford_inference(16.0, elapsed_ms=2.0)
    assert controller.can_afford_inference(25.0, elapsed_ms=2.0)
    controller.record_inference_cost(10.0)
    assert controller.predicted_inference_cost_ms == pytest.approx(15.0)
    with pytest.raises(ValueError):
        controller.record_inference_cost(float("nan"))


def test_whole_model_is_not_run_for_sparse_eligible_tiles() -> None:
    frame = _textured_frame(64, 160)
    controller = RegionalTransformController(
        RegionalConfig(
            tile_size=16,
            analysis_scale=1,
            minimum_model_tile_fraction=0.10,
            minimum_model_confidence_coverage=0.04,
        )
    )
    analysis = controller.analyze(frame, np.roll(frame, 1, axis=1))
    analysis.routes.fill(RegionRoute.LOW_DETAIL)
    analysis.confidence.fill(0.0)
    selected = np.zeros(analysis.grid_shape, dtype=bool)
    selected.flat[:3] = True
    analysis.routes[selected] = RegionRoute.SMOOTH_MOTION
    analysis.confidence[selected] = 1.0

    assert not controller.should_run_model(analysis, selected, 16.7)

    selected.flat[:8] = True
    analysis.routes[selected] = RegionRoute.SMOOTH_MOTION
    analysis.confidence[selected] = 1.0
    assert controller.should_run_model(analysis, selected, 16.7)


def test_model_admission_uses_full_transform_cost() -> None:
    frame = _textured_frame(64, 160)
    controller = RegionalTransformController()
    analysis = controller.analyze(frame, np.roll(frame, 2, axis=1))
    analysis.routes.fill(RegionRoute.SMOOTH_MOTION)
    analysis.confidence.fill(1.0)
    selected = np.ones(analysis.grid_shape, dtype=bool)

    controller.record_inference_cost(28.0)

    assert controller.predicted_transform_cost_ms == pytest.approx(28.0)
    assert not controller.should_run_model(analysis, selected, 16.7)


def test_deadline_admission_reprobes_and_recovers_from_one_slow_call() -> None:
    frame = _textured_frame(64, 160)
    controller = RegionalTransformController(
        RegionalConfig(deadline_probe_interval=3, cost_alpha=0.2)
    )
    analysis = controller.analyze(frame, np.roll(frame, 2, axis=1))
    analysis.routes.fill(RegionRoute.SMOOTH_MOTION)
    analysis.confidence.fill(1.0)
    selected = np.ones(analysis.grid_shape, dtype=bool)
    controller.record_inference_cost(44.0)

    assert not controller.should_run_model(analysis, selected, 34.0)
    assert not controller.should_run_model(analysis, selected, 34.0)
    assert controller.should_run_model(analysis, selected, 34.0)
    assert controller.last_admission_reason == "model_deadline_probe"

    controller.record_inference_cost(20.0)
    assert controller.predicted_transform_cost_ms == pytest.approx(32.0)
    assert controller.should_run_model(analysis, selected, 34.0)


def test_confidence_composition_rejects_bad_generated_region() -> None:
    first = _textured_frame()
    second = np.roll(first, 2, axis=1)
    generated = cv2.addWeighted(first, 0.5, second, 0.5, 0.0)
    generated[:, :64] = 255
    controller = RegionalTransformController(
        RegionalConfig(tile_size=32, analysis_scale=2, text_threshold=1.0)
    )
    analysis = controller.analyze(first, second)
    analysis.routes.fill(RegionRoute.SMOOTH_MOTION)
    analysis.confidence.fill(1.0)
    result = controller.compose_interpolation(
        generated, first, second, analysis=analysis
    )
    midpoint = cv2.addWeighted(first, 0.5, second, 0.5, 0.0)
    bad_error = np.mean(cv2.absdiff(result.frame[:, :64], midpoint[:, :64]))
    candidate_error = np.mean(cv2.absdiff(generated[:, :64], midpoint[:, :64]))
    assert bad_error < candidate_error * 0.25
    assert result.provenance_counts.get("midpoint_blend", 0) > 0
    assert result.pixel_confidence.shape == first.shape[:2]
    assert np.all((result.pixel_confidence >= 0.0) & (result.pixel_confidence <= 1.0))
    assert result.endpoint_fallback.shape == first.shape[:2]


def test_fractional_fallback_blends_at_requested_temporal_position():
    first = np.zeros((64, 96, 3), dtype=np.uint8)
    second = np.full_like(first, 30)
    controller = RegionalTransformController(
        RegionalConfig(tile_size=16, analysis_scale=1, text_threshold=1.0)
    )
    analysis = controller.analyze(first, second)
    analysis.routes.fill(RegionRoute.STATIC_DETAIL)
    analysis.confidence.fill(0.0)
    result = controller.compose_interpolation(
        np.full_like(first, 255),
        first,
        second,
        analysis=analysis,
        time_position=1.0 / 3.0,
    )
    assert float(result.frame.mean()) == pytest.approx(10.0, abs=1.0)
    assert result.provenance_counts.get("temporal_blend", 0) > 0


def test_scene_cut_returns_real_endpoint() -> None:
    first = np.zeros((96, 128, 3), dtype=np.uint8)
    second = np.full_like(first, 255)
    generated = np.full_like(first, 128)
    controller = RegionalTransformController()
    result = controller.compose_interpolation(generated, first, second)
    np.testing.assert_array_equal(result.frame, second)
    assert np.all(result.provenance == RegionProvenance.REAL)


def test_static_detail_never_accepts_temporal_candidate() -> None:
    first = _textured_frame()
    second = first.copy()
    generated = np.full_like(first, 240)
    controller = RegionalTransformController(
        RegionalConfig(tile_size=32, analysis_scale=2, text_threshold=1.0)
    )
    analysis = controller.analyze(first, second)
    analysis.routes.fill(RegionRoute.STATIC_DETAIL)
    analysis.confidence.fill(1.0)
    result = controller.compose_interpolation(
        generated, first, second, analysis=analysis
    )
    np.testing.assert_array_equal(result.frame, first)
    assert not np.any(result.selected)
    assert result.provenance_counts.get("midpoint_blend", 0) > 0


def test_unreliable_motion_uses_clean_endpoint_instead_of_ghosted_midpoint() -> None:
    first = np.zeros((64, 96, 3), dtype=np.uint8)
    second = first.copy()
    second[:, 32:64] = 255
    generated = cv2.addWeighted(first, 0.5, second, 0.5, 0.0)
    controller = RegionalTransformController(
        RegionalConfig(tile_size=32, analysis_scale=1, text_threshold=1.0)
    )
    analysis = controller.analyze(first, second)
    analysis.routes.fill(RegionRoute.UNRELIABLE_MOTION)
    result = controller.compose_interpolation(
        generated, first, second, analysis=analysis
    )
    assert float(np.mean(result.frame[:, 40:56])) > 245.0
    assert float(np.mean(result.endpoint_fallback[:, 40:56])) > 0.95


def test_feedback_reduces_reliability_after_large_error() -> None:
    first = _textured_frame()
    second = np.roll(first, 3, axis=1)
    generated = cv2.addWeighted(first, 0.5, second, 0.5, 0.0)
    controller = RegionalTransformController(
        RegionalConfig(tile_size=32, analysis_scale=2, text_threshold=1.0)
    )
    analysis_before = controller.analyze(first, second)
    analysis_before.routes.fill(RegionRoute.SMOOTH_MOTION)
    analysis_before.confidence.fill(1.0)
    controller.compose_interpolation(generated, first, second, analysis=analysis_before)
    error = controller.observe_real_frame(np.zeros_like(first))
    analysis_after = controller.analyze(first, second)
    assert error is not None and error > 20.0
    assert float(np.mean(analysis_after.confidence)) < float(
        np.mean(analysis_before.confidence)
    )


def test_provenance_overlay_has_same_shape_and_does_not_mutate() -> None:
    frame = _textured_frame(64, 96)
    original = frame.copy()
    provenance = np.array(
        [[RegionProvenance.REAL, RegionProvenance.INTERPOLATED, RegionProvenance.BLENDED]],
        dtype=np.uint8,
    )
    overlay = RegionalTransformController.provenance_overlay(frame, provenance)
    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, frame)
    np.testing.assert_array_equal(frame, original)

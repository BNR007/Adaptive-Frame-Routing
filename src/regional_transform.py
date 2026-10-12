"""Explainable regional routing for frame transformation.

The controller deliberately contains no neural model.  It analyses inexpensive
low-resolution tile features, decides which parts of an interpolation candidate
are reliable, allocates optional work against a deadline, and records the origin
of every output tile.  Keeping it independent from RIFE and the upscalers makes
each mechanism directly ablatable in experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import time
from typing import Iterable

import cv2
import numpy as np


class RegionRoute(IntEnum):
    """Selected transformation route for one analysis tile."""

    SCENE_CUT = 0
    STATIC_DETAIL = 1
    SMOOTH_MOTION = 2
    UNRELIABLE_MOTION = 3
    HUD_TEXT = 4
    LOW_DETAIL = 5


class RegionProvenance(IntEnum):
    """Origin of the pixels emitted for one output tile."""

    REAL = 0
    SPATIAL = 1
    INTERPOLATED = 2
    BLENDED = 3
    REUSED = 4
    DEADLINE_FALLBACK = 5
    MIDPOINT_BLEND = 6
    ADMISSION_FALLBACK = 7
    TEMPORAL_BLEND = 8


@dataclass(frozen=True, slots=True)
class RegionalConfig:
    tile_size: int = 16
    analysis_scale: int = 4
    scene_cut_threshold: float = 42.0
    static_motion_threshold: float = 2.5
    smooth_motion_threshold: float = 18.0
    detail_threshold: float = 0.075
    text_threshold: float = 0.22
    residual_floor: float = 7.0
    residual_motion_scale: float = 0.85
    confidence_floor: float = 0.10
    feedback_alpha: float = 0.18
    cost_alpha: float = 0.20
    hysteresis: float = 0.08
    minimum_model_tile_fraction: float = 0.10
    minimum_model_confidence_coverage: float = 0.04

    def __post_init__(self) -> None:
        if self.tile_size < 4:
            raise ValueError("Regional tile size must be at least four pixels.")
        if self.analysis_scale < 1:
            raise ValueError("Regional analysis scale must be positive.")
        for name in ("feedback_alpha", "cost_alpha"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        for name in (
            "minimum_model_tile_fraction",
            "minimum_model_confidence_coverage",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(slots=True)
class RegionalAnalysis:
    routes: np.ndarray
    confidence: np.ndarray
    utility: np.ndarray
    motion: np.ndarray
    edge_density: np.ndarray
    motion_consistency: np.ndarray
    text_likelihood: np.ndarray
    scene_cut: bool
    analysis_ms: float
    source_shape: tuple[int, int]
    analysis_shape: tuple[int, int]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.routes.shape


@dataclass(slots=True)
class RegionalResult:
    frame: np.ndarray
    provenance: np.ndarray
    confidence: np.ndarray
    pixel_confidence: np.ndarray
    endpoint_fallback: np.ndarray
    selected: np.ndarray
    route_counts: dict[str, int]
    provenance_counts: dict[str, int]
    analysis_ms: float
    composition_ms: float
    deadline_misses: int
    admission_reason: str


class RegionalTransformController:
    """Low-cost stateful regional analysis, routing, and composition."""

    def __init__(self, config: RegionalConfig | None = None) -> None:
        self.config = config or RegionalConfig()
        self._reliability: np.ndarray | None = None
        self._cost_ms: np.ndarray | None = None
        self._previous_routes: np.ndarray | None = None
        self._pending_prediction: np.ndarray | None = None
        self._pending_confidence: np.ndarray | None = None
        self._pending_shape: tuple[int, int] | None = None
        self._transform_cost_ms: float | None = None
        self.last_feedback_error = 0.0
        self.last_admission_reason = "not_evaluated"

    def reset(self) -> None:
        self._reliability = None
        self._cost_ms = None
        self._previous_routes = None
        self._pending_prediction = None
        self._pending_confidence = None
        self._pending_shape = None
        self._transform_cost_ms = None
        self.last_feedback_error = 0.0
        self.last_admission_reason = "not_evaluated"

    @property
    def predicted_inference_cost_ms(self) -> float | None:
        """Compatibility alias for the learned end-to-end model cost."""

        return self._transform_cost_ms

    @property
    def predicted_transform_cost_ms(self) -> float | None:
        return self._transform_cost_ms

    def can_afford_inference(
        self, budget_ms: float | None, *, elapsed_ms: float = 0.0
    ) -> bool:
        """Return whether the learned whole-model cost fits this deadline."""

        if budget_ms is None or self._transform_cost_ms is None:
            return True
        remaining = max(0.0, float(budget_ms) - max(0.0, float(elapsed_ms)))
        return self._transform_cost_ms <= remaining

    def record_inference_cost(self, measured_ms: float) -> None:
        """Update the whole-transform cost prediction.

        The historical method name remains for compatibility. Callers pass the
        measured preprocessing + inference + composition + restoration cost,
        because admitting work from inference time alone missed the expensive
        CPU postprocessing stage.
        """

        if not np.isfinite(measured_ms) or measured_ms < 0.0:
            raise ValueError("Measured inference cost must be finite and non-negative.")
        if self._transform_cost_ms is None:
            self._transform_cost_ms = float(measured_ms)
            return
        alpha = self.config.cost_alpha
        self._transform_cost_ms = (
            self._transform_cost_ms * (1.0 - alpha) + float(measured_ms) * alpha
        )

    def should_run_model(
        self,
        analysis: RegionalAnalysis,
        selected: np.ndarray,
        budget_ms: float | None,
        *,
        elapsed_ms: float = 0.0,
    ) -> bool:
        """Admit a whole-frame neural pass only when it has enough useful work."""

        if selected.shape != analysis.grid_shape:
            raise ValueError("Selected tile mask must match the analysis grid.")
        eligible = selected & (analysis.routes == RegionRoute.SMOOTH_MOTION)
        tile_count = int(eligible.size)
        selected_count = int(np.count_nonzero(eligible))
        if tile_count == 0 or selected_count == 0:
            self.last_admission_reason = "no_eligible_tiles"
            return False
        tile_fraction = selected_count / tile_count
        confidence_coverage = float(np.sum(analysis.confidence[eligible])) / tile_count
        if tile_fraction < self.config.minimum_model_tile_fraction:
            self.last_admission_reason = "sparse_coverage"
            return False
        if confidence_coverage < self.config.minimum_model_confidence_coverage:
            self.last_admission_reason = "low_confidence_coverage"
            return False
        if not self.can_afford_inference(budget_ms, elapsed_ms=elapsed_ms):
            self.last_admission_reason = "deadline"
            return False
        self.last_admission_reason = "model"
        return True

    @staticmethod
    def _validate_pair(frame_a: np.ndarray, frame_b: np.ndarray) -> None:
        for value, name in ((frame_a, "frame_a"), (frame_b, "frame_b")):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array.")
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
                raise ValueError(f"{name} must be an HxWx3 uint8 image.")
        if frame_a.shape != frame_b.shape:
            raise ValueError("Regional endpoint frames must have matching shapes.")

    def _analysis_pair(
        self, frame_a: np.ndarray, frame_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = frame_a.shape[:2]
        scale = self.config.analysis_scale
        target = (max(1, width // scale), max(1, height // scale))
        if target == (width, height):
            return frame_a, frame_b
        return (
            cv2.resize(frame_a, target, interpolation=cv2.INTER_AREA),
            cv2.resize(frame_b, target, interpolation=cv2.INTER_AREA),
        )

    @staticmethod
    def _tile_means(image: np.ndarray, tile: int) -> np.ndarray:
        height, width = image.shape[:2]
        rows = (height + tile - 1) // tile
        columns = (width + tile - 1) // tile
        padded = cv2.copyMakeBorder(
            image,
            0,
            rows * tile - height,
            0,
            columns * tile - width,
            cv2.BORDER_REPLICATE,
        )
        return padded.reshape(rows, tile, columns, tile).mean(axis=(1, 3))

    @staticmethod
    def _tile_standard_deviation(image: np.ndarray, tile: int) -> np.ndarray:
        means = RegionalTransformController._tile_means(image, tile)
        squared = RegionalTransformController._tile_means(
            image.astype(np.float32) ** 2, tile
        )
        return np.sqrt(np.maximum(0.0, squared - means**2))

    def _ensure_state(self, shape: tuple[int, int]) -> None:
        if self._reliability is None or self._reliability.shape != shape:
            self._reliability = np.ones(shape, dtype=np.float32)
            self._cost_ms = np.ones(shape, dtype=np.float32)
            self._previous_routes = None

    def analyze(self, frame_a: np.ndarray, frame_b: np.ndarray) -> RegionalAnalysis:
        """Calculate inexpensive features and choose an initial route per tile."""

        self._validate_pair(frame_a, frame_b)
        started = time.perf_counter()
        first, second = self._analysis_pair(frame_a, frame_b)
        gray_a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(gray_a, gray_b)
        tile = max(4, self.config.tile_size // self.config.analysis_scale)

        motion = self._tile_means(difference, tile).astype(np.float32)
        motion_spread = self._tile_standard_deviation(difference, tile).astype(
            np.float32
        )
        motion_consistency = np.clip(
            1.0 - motion_spread / np.maximum(motion + 8.0, 8.0), 0.0, 1.0
        )

        edges = cv2.Canny(gray_b, 48, 128)
        edge_density = (
            self._tile_means((edges > 0).astype(np.float32), tile).astype(np.float32)
        )
        # Dense, fine alternating gradients are a cheap HUD/text proxy.  It is
        # intentionally conservative: false positives only choose a stable
        # spatial route rather than corrupting text temporally.
        laplacian = cv2.Laplacian(gray_b, cv2.CV_32F)
        high_frequency = (np.abs(laplacian) > 34.0).astype(np.float32)
        text_likelihood = self._tile_means(high_frequency, tile).astype(np.float32)

        global_change = float(np.mean(difference))
        changed_fraction = float(np.mean(difference > 24))
        scene_cut = (
            global_change >= self.config.scene_cut_threshold
            and changed_fraction >= 0.55
        )

        routes = np.full(motion.shape, RegionRoute.LOW_DETAIL, dtype=np.uint8)
        if scene_cut:
            routes.fill(RegionRoute.SCENE_CUT)
        else:
            detailed = edge_density >= self.config.detail_threshold
            # HUD/text usually has strong high-frequency structure while
            # remaining nearly fixed between endpoints. Requiring low motion
            # prevents detailed rocks, vegetation, and particles from being
            # misclassified as interface content.
            text = (
                (text_likelihood >= self.config.text_threshold)
                & (edge_density >= self.config.detail_threshold)
                & (
                    (motion <= self.config.static_motion_threshold * 1.6)
                    | (
                        text_likelihood
                        >= max(0.45, self.config.text_threshold * 3.0)
                    )
                )
            )
            static = motion <= self.config.static_motion_threshold
            smooth = (
                (motion > self.config.static_motion_threshold)
                & (motion <= self.config.smooth_motion_threshold)
                & (motion_consistency >= 0.46)
            )
            unreliable = (
                (motion > self.config.smooth_motion_threshold)
                | (motion_consistency < 0.30)
            )
            routes[static & detailed] = RegionRoute.STATIC_DETAIL
            routes[smooth] = RegionRoute.SMOOTH_MOTION
            routes[unreliable] = RegionRoute.UNRELIABLE_MOTION
            routes[text] = RegionRoute.HUD_TEXT

        self._ensure_state(routes.shape)
        assert self._reliability is not None
        assert self._cost_ms is not None
        confidence = np.clip(
            (0.58 * motion_consistency)
            + (0.42 * self._reliability)
            - np.clip(motion / 80.0, 0.0, 0.45),
            0.0,
            1.0,
        ).astype(np.float32)
        confidence[np.isin(routes, (RegionRoute.SCENE_CUT, RegionRoute.HUD_TEXT))] = 0.0
        confidence[routes == RegionRoute.UNRELIABLE_MOTION] *= 0.25
        confidence[routes == RegionRoute.SMOOTH_MOTION] = np.maximum(
            confidence[routes == RegionRoute.SMOOTH_MOTION], 0.55
        )

        # Hysteresis prevents tiles near a boundary from changing route every
        # frame.  Only keep the previous route when confidence is ambiguous.
        if self._previous_routes is not None:
            ambiguous = np.abs(confidence - 0.5) < self.config.hysteresis
            routes[ambiguous] = self._previous_routes[ambiguous]
        self._previous_routes = routes.copy()

        expected_gain = (
            0.30
            + edge_density * 1.8
            + np.clip(motion / 32.0, 0.0, 1.0) * 0.4
        )
        utility = (expected_gain * confidence) / np.maximum(self._cost_ms, 0.05)
        analysis_ms = (time.perf_counter() - started) * 1000.0
        return RegionalAnalysis(
            routes=routes,
            confidence=confidence,
            utility=utility.astype(np.float32),
            motion=motion,
            edge_density=edge_density,
            motion_consistency=motion_consistency,
            text_likelihood=text_likelihood,
            scene_cut=scene_cut,
            analysis_ms=analysis_ms,
            source_shape=frame_a.shape[:2],
            analysis_shape=first.shape[:2],
        )

    def allocate(
        self,
        analysis: RegionalAnalysis,
        budget_ms: float | None,
        *,
        elapsed_ms: float = 0.0,
    ) -> np.ndarray:
        """Select tiles in utility order without exceeding the remaining budget."""

        # The current interpolator is temporal-only. Static-detail tiles belong
        # to the future spatial-upscaling allocator and must not consume this
        # deadline.
        eligible = analysis.routes == RegionRoute.SMOOTH_MOTION
        if budget_ms is None:
            return eligible
        remaining = max(0.0, float(budget_ms) - max(0.0, float(elapsed_ms)))
        selected = np.zeros(analysis.grid_shape, dtype=bool)
        assert self._cost_ms is not None
        flat_indices = np.argsort(analysis.utility.ravel())[::-1]
        for flat_index in flat_indices:
            row, column = np.unravel_index(flat_index, analysis.grid_shape)
            if not eligible[row, column]:
                continue
            cost = float(self._cost_ms[row, column])
            if cost <= remaining:
                selected[row, column] = True
                remaining -= cost
        return selected

    @staticmethod
    def _expand_tiles(
        values: np.ndarray,
        height: int,
        width: int,
        *,
        interpolation: int = cv2.INTER_NEAREST,
    ) -> np.ndarray:
        return cv2.resize(values, (width, height), interpolation=interpolation)

    @staticmethod
    def _counts(values: np.ndarray, enum_type: type[IntEnum]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for member in enum_type:
            count = int(np.count_nonzero(values == int(member)))
            if count:
                counts[member.name.lower()] = count
        return counts

    def compose_interpolation(
        self,
        generated: np.ndarray,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        *,
        analysis: RegionalAnalysis | None = None,
        budget_ms: float | None = None,
        time_position: float = 0.5,
    ) -> RegionalResult:
        """Accept, blend, or reject an interpolation candidate per region."""

        self._validate_pair(frame_a, frame_b)
        if not np.isfinite(time_position) or not 0.0 < time_position < 1.0:
            raise ValueError("Interpolation time position must be between zero and one.")
        if generated.shape != frame_a.shape or generated.dtype != np.uint8:
            raise ValueError("Generated candidate must match the endpoint images.")
        analysis = analysis or self.analyze(frame_a, frame_b)
        started = time.perf_counter()
        selected = self.allocate(
            analysis, budget_ms, elapsed_ms=analysis.analysis_ms
        )
        # Static-detail tiles are candidates for spatial enhancement, never
        # temporal synthesis.  Only predictable motion may accept model pixels.
        selected &= analysis.routes == RegionRoute.SMOOTH_MOTION
        height, width = frame_a.shape[:2]
        temporal_blend = cv2.addWeighted(
            frame_a, 1.0 - time_position, frame_b, time_position, 0.0
        )
        nearest_endpoint = frame_a if time_position < 0.5 else frame_b
        endpoint_tiles = np.isin(
            analysis.routes,
            (
                RegionRoute.SCENE_CUT,
                RegionRoute.UNRELIABLE_MOTION,
                RegionRoute.HUD_TEXT,
            ),
        ) | ((analysis.routes == RegionRoute.SMOOTH_MOTION) & ~selected)
        endpoint_fallback = self._expand_tiles(
            endpoint_tiles.astype(np.float32),
            height,
            width,
            interpolation=cv2.INTER_LINEAR,
        )
        endpoint_fallback = cv2.GaussianBlur(
            endpoint_fallback, (0, 0), sigmaX=1.1
        )
        endpoint_fallback = np.clip(endpoint_fallback, 0.0, 1.0).astype(
            np.float32
        )
        stable_fallback = np.rint(
            nearest_endpoint.astype(np.float32) * endpoint_fallback[..., None]
            + temporal_blend.astype(np.float32)
            * (1.0 - endpoint_fallback[..., None])
        ).astype(np.uint8)

        residual = cv2.cvtColor(
            cv2.absdiff(generated, temporal_blend), cv2.COLOR_BGR2GRAY
        )
        residual_tiles = self._tile_means(
            residual,
            max(4, self.config.tile_size),
        ).astype(np.float32)
        residual_tiles = cv2.resize(
            residual_tiles,
            (analysis.grid_shape[1], analysis.grid_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        allowed = (
            self.config.residual_floor
            + analysis.motion * self.config.residual_motion_scale
        )
        residual_confidence = np.clip(
            1.0 - residual_tiles / np.maximum(allowed, 1.0), 0.0, 1.0
        )
        confidence_tiles = np.clip(
            analysis.confidence * residual_confidence, 0.0, 1.0
        )
        confidence_tiles[~selected] = 0.0
        confidence_tiles[
            np.isin(
                analysis.routes,
                (
                    RegionRoute.SCENE_CUT,
                    RegionRoute.HUD_TEXT,
                    RegionRoute.UNRELIABLE_MOTION,
                    RegionRoute.LOW_DETAIL,
                ),
            )
        ] = 0.0

        confidence = self._expand_tiles(
            confidence_tiles, height, width, interpolation=cv2.INTER_LINEAR
        )
        # A light blur feathers tile boundaries without shifting image content.
        confidence = cv2.GaussianBlur(confidence, (0, 0), sigmaX=1.1)
        confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
        output = np.rint(
            generated.astype(np.float32) * confidence[..., None]
            + stable_fallback.astype(np.float32) * (1.0 - confidence[..., None])
        ).astype(np.uint8)

        provenance = np.full(
            analysis.grid_shape, RegionProvenance.BLENDED, dtype=np.uint8
        )
        provenance[confidence_tiles >= 0.92] = RegionProvenance.INTERPOLATED
        low_confidence = confidence_tiles <= self.config.confidence_floor
        provenance[low_confidence & ~endpoint_tiles] = (
            RegionProvenance.MIDPOINT_BLEND
            if abs(time_position - 0.5) < 1e-9
            else RegionProvenance.TEMPORAL_BLEND
        )
        provenance[low_confidence & endpoint_tiles] = RegionProvenance.REAL
        eligible = analysis.routes == RegionRoute.SMOOTH_MOTION
        provenance[eligible & ~selected] = RegionProvenance.DEADLINE_FALLBACK
        if analysis.scene_cut:
            provenance.fill(RegionProvenance.REAL)
            output = frame_b.copy()

        self._pending_prediction = output.copy()
        self._pending_confidence = confidence_tiles.copy()
        self._pending_shape = analysis.grid_shape
        composition_ms = (time.perf_counter() - started) * 1000.0
        return RegionalResult(
            frame=output,
            provenance=provenance,
            confidence=confidence_tiles,
            pixel_confidence=confidence,
            endpoint_fallback=endpoint_fallback,
            selected=selected,
            route_counts=self._counts(analysis.routes, RegionRoute),
            provenance_counts=self._counts(provenance, RegionProvenance),
            analysis_ms=analysis.analysis_ms,
            composition_ms=composition_ms,
            deadline_misses=int(np.count_nonzero(eligible & ~selected)),
            admission_reason=self.last_admission_reason,
        )

    def compose_fallback(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        *,
        analysis: RegionalAnalysis,
        time_position: float = 0.5,
    ) -> RegionalResult:
        """Build provenance for a rejected neural pass without candidate scoring."""

        self._validate_pair(frame_a, frame_b)
        started = time.perf_counter()
        height, width = frame_a.shape[:2]
        endpoint_tiles = np.isin(
            analysis.routes,
            (RegionRoute.SCENE_CUT,),
        )
        endpoint_fallback = self._expand_tiles(
            endpoint_tiles.astype(np.float32),
            height,
            width,
            interpolation=cv2.INTER_NEAREST,
        )
        if not np.isfinite(time_position) or not 0.0 < time_position < 1.0:
            raise ValueError("Interpolation time position must be between zero and one.")
        midpoint = cv2.addWeighted(
            frame_a, 1.0 - time_position, frame_b, time_position, 0.0
        )
        endpoint_mask = cv2.compare(endpoint_fallback, 0.5, cv2.CMP_GE)
        cv2.copyTo(frame_b, endpoint_mask, midpoint)
        confidence_tiles = np.zeros(analysis.grid_shape, dtype=np.float32)
        pixel_confidence = np.zeros((height, width), dtype=np.float32)
        selected = np.zeros(analysis.grid_shape, dtype=bool)
        provenance = np.full(
            analysis.grid_shape,
            (
                RegionProvenance.MIDPOINT_BLEND
                if abs(time_position - 0.5) < 1e-9
                else RegionProvenance.TEMPORAL_BLEND
            ),
            dtype=np.uint8,
        )
        eligible = analysis.routes == RegionRoute.SMOOTH_MOTION
        provenance[eligible] = (
            RegionProvenance.DEADLINE_FALLBACK
            if self.last_admission_reason == "deadline"
            else RegionProvenance.ADMISSION_FALLBACK
        )
        if analysis.scene_cut:
            provenance.fill(RegionProvenance.REAL)
            midpoint = frame_b.copy()
        self._pending_prediction = midpoint.copy()
        self._pending_confidence = confidence_tiles
        self._pending_shape = analysis.grid_shape
        return RegionalResult(
            frame=midpoint,
            provenance=provenance,
            confidence=confidence_tiles,
            pixel_confidence=pixel_confidence,
            endpoint_fallback=endpoint_fallback,
            selected=selected,
            route_counts=self._counts(analysis.routes, RegionRoute),
            provenance_counts=self._counts(provenance, RegionProvenance),
            analysis_ms=analysis.analysis_ms,
            composition_ms=(time.perf_counter() - started) * 1000.0,
            deadline_misses=int(np.count_nonzero(eligible)),
            admission_reason=self.last_admission_reason,
        )

    def describe_precomposed_fallback(
        self,
        fallback_frame: np.ndarray,
        *,
        analysis: RegionalAnalysis,
    ) -> RegionalResult:
        """Record fallback provenance without recomputing an already-built frame.

        The frame interpolator has already formed the midpoint when admission
        rejects neural work.  The older path formed it again, expanded masks,
        copied it twice, and then discarded that image after source-resolution
        restoration.  This path keeps the scientific routing counters while
        avoiding that duplicate pixel work.
        """

        started = time.perf_counter()
        if fallback_frame.ndim != 3 or fallback_frame.dtype != np.uint8:
            raise ValueError("Fallback frame must be a uint8 colour image.")
        confidence_tiles = np.zeros(analysis.grid_shape, dtype=np.float32)
        selected = np.zeros(analysis.grid_shape, dtype=bool)
        provenance = np.full(
            analysis.grid_shape,
            RegionProvenance.MIDPOINT_BLEND,
            dtype=np.uint8,
        )
        eligible = analysis.routes == RegionRoute.SMOOTH_MOTION
        provenance[eligible] = (
            RegionProvenance.DEADLINE_FALLBACK
            if self.last_admission_reason == "deadline"
            else RegionProvenance.ADMISSION_FALLBACK
        )
        output = fallback_frame
        if analysis.scene_cut:
            provenance.fill(RegionProvenance.REAL)
        height, width = output.shape[:2]
        pixel_confidence = np.zeros((height, width), dtype=np.float32)
        endpoint_fallback = np.zeros((height, width), dtype=np.float32)
        self._pending_prediction = output.copy()
        self._pending_confidence = confidence_tiles
        self._pending_shape = analysis.grid_shape
        return RegionalResult(
            frame=output,
            provenance=provenance,
            confidence=confidence_tiles,
            pixel_confidence=pixel_confidence,
            endpoint_fallback=endpoint_fallback,
            selected=selected,
            route_counts=self._counts(analysis.routes, RegionRoute),
            provenance_counts=self._counts(provenance, RegionProvenance),
            analysis_ms=analysis.analysis_ms,
            composition_ms=(time.perf_counter() - started) * 1000.0,
            deadline_misses=int(np.count_nonzero(eligible)),
            admission_reason=self.last_admission_reason,
        )

    def observe_real_frame(self, actual: np.ndarray) -> float | None:
        """Update per-tile reliability after the next real frame becomes known."""

        if self._pending_prediction is None or self._pending_shape is None:
            return None
        if actual.shape != self._pending_prediction.shape or actual.dtype != np.uint8:
            self._pending_prediction = None
            self._pending_confidence = None
            self._pending_shape = None
            return None
        error = cv2.cvtColor(
            cv2.absdiff(self._pending_prediction, actual), cv2.COLOR_BGR2GRAY
        )
        tile = max(4, self.config.tile_size)
        tile_error = self._tile_means(error, tile).astype(np.float32)
        tile_error = cv2.resize(
            tile_error,
            (self._pending_shape[1], self._pending_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        observed_reliability = np.exp(-tile_error / 24.0).astype(np.float32)
        self._ensure_state(self._pending_shape)
        assert self._reliability is not None
        alpha = self.config.feedback_alpha
        self._reliability *= 1.0 - alpha
        self._reliability += observed_reliability * alpha
        self.last_feedback_error = float(np.mean(tile_error))
        self._pending_prediction = None
        self._pending_confidence = None
        self._pending_shape = None
        return self.last_feedback_error

    def record_costs(
        self,
        measured_ms: float,
        selected: np.ndarray,
    ) -> None:
        """Update the cost predictor from a measured regional operation."""

        if measured_ms < 0.0 or selected.dtype != bool:
            raise ValueError("Measured cost must be non-negative and mask boolean.")
        self._ensure_state(selected.shape)
        assert self._cost_ms is not None
        count = int(np.count_nonzero(selected))
        if count == 0:
            return
        per_tile = float(measured_ms) / count
        alpha = self.config.cost_alpha
        self._cost_ms[selected] = (
            self._cost_ms[selected] * (1.0 - alpha) + per_tile * alpha
        )

    @staticmethod
    def provenance_overlay(
        frame: np.ndarray,
        provenance: np.ndarray,
        *,
        opacity: float = 0.32,
    ) -> np.ndarray:
        """Return a colour-coded provenance view without mutating ``frame``."""

        if not 0.0 <= opacity <= 1.0:
            raise ValueError("Overlay opacity must be between zero and one.")
        palette = np.array(
            [
                (96, 220, 96),    # real: green
                (255, 180, 48),   # spatial: cyan/blue in BGR
                (255, 96, 48),    # interpolated: blue
                (32, 220, 240),   # blended: yellow
                (220, 64, 220),   # reused: purple
                (48, 48, 255),    # deadline fallback: red
                (128, 180, 255),  # midpoint blend: pale orange
                (180, 96, 255),   # admission fallback: pink
                (96, 220, 255),   # non-midpoint temporal blend: amber
            ],
            dtype=np.uint8,
        )
        labels = RegionalTransformController._expand_tiles(
            provenance.astype(np.uint8), frame.shape[0], frame.shape[1]
        )
        colors = palette[np.clip(labels, 0, len(palette) - 1)]
        return cv2.addWeighted(frame, 1.0 - opacity, colors, opacity, 0.0)


def route_sequence(values: np.ndarray) -> Iterable[RegionRoute]:
    """Yield typed route labels for reporting code."""

    for value in values.ravel():
        yield RegionRoute(int(value))

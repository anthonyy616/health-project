"""
Iris-Calibrated Pupil Segmentation - Pure Functions
Stateless landmark math and pixel-level pupil segmentation for pupil
dilation measurement. No persistent state, no buffering - every function
is independently unit-testable with synthetic inputs.

DESIGN DEVIATION FROM THE LITERAL IMPLEMENTATION PLAN WORDING:
The implementation plan says "Implement MediaPipe Iris for eye tracking...
Calculate pupil diameter", but that phrasing is anatomically imprecise: the
iris ring's physical diameter is nearly constant across humans (~11.7mm)
and does NOT change when the pupil dilates/constricts. Only the pupil does.

So this module uses the iris ring for two things only:
  (a) locating a small eye crop around the iris center, and
  (b) establishing a pixels-per-mm scale factor from the known anatomical
      iris diameter (AVERAGE_IRIS_DIAMETER_MM).
The actual pupil size is then found with basic image processing (CLAHE +
Otsu threshold + circular-contour detection) inside that crop, and
converted to mm with the scale factor from (b). Tracking iris landmarks
alone would report a nearly-constant number every frame, which defeats the
goal of tracking dilation changes.
"""

import numpy as np
import cv2
from typing import Optional, Sequence
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Iris landmark indices (478-landmark MediaPipe Face Landmarker model already
# used by FaceDetector - no separate MediaPipe Iris solution needed).
#
# EMPIRICALLY VERIFIED 2026-08-18 against real FaceDetector output (UTKFace
# images, 200x200 crops): landmarks 468-477 ARE produced, each iris as a
# center landmark + 4 ring points at roughly 90-degree intervals.
#
# IMPORTANT - naming/pairing gotcha: the constant names below follow the
# stage plan / MediaPipe's SUBJECT-perspective naming (right iris = 468-472,
# left iris = 473-477). MediaPipe's "right" is the subject's right eye, which
# appears on the IMAGE-LEFT side of the camera frame. FaceDetector names its
# contour lists from the CAMERA-VIEW: FaceDetector.LEFT_EYE_LANDMARKS
# (indices 33/133...) surrounds the image-left eye, and
# FaceDetector.RIGHT_EYE_LANDMARKS (indices 362/263...) surrounds the
# image-right eye. Verified: iris 468-472 lies inside the LEFT_EYE_LANDMARKS
# contour, iris 473-477 inside RIGHT_EYE_LANDMARKS.
#
# The LEFT/RIGHT pairing in detect.py accounts for this swap so the
# displayed L/R labels match what Anthony sees in the camera feed.
# ---------------------------------------------------------------------------
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]   # center + 4 ring points (subject's right = image-left eye)
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]    # center + 4 ring points (subject's left = image-right eye)

# Anatomical constants
AVERAGE_IRIS_DIAMETER_MM = 11.7   # ~11.7mm +/- 0.5mm, low inter-person variance
MIN_PLAUSIBLE_PUPIL_RATIO = 0.2   # pupil:iris diameter ratio physiological floor
MAX_PLAUSIBLE_PUPIL_RATIO = 0.9   # physiological ceiling

# Segmentation constants
CIRCULARITY_THRESHOLD = 0.6       # 4*pi*area/perimeter^2 minimum for a circular pupil candidate
PUPIL_RADIUS_TOLERANCE = 1.25     # slack on the expected-radius upper bound (digitization etc.)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# A "pupil" smaller than this in pixels is not a pupil - used to discard
# specks/lashes that happen to be circular.
MIN_PUPIL_RADIUS_PX = 2.0


def compute_iris_diameter_px(
    landmarks: np.ndarray,
    ring_indices: Sequence[int]
) -> float:
    """
    Compute the iris ring diameter in pixels.

    The first entry of ring_indices is treated as the iris CENTER and the
    remaining four as ring points. The diameter is estimated as the mean of
    the two largest pairwise distances among the 4 ring points. This is
    deliberately ORDERING-AGNOSTIC: whatever the exact top/right/bottom/left
    order of the ring points is (it varies with head tilt/perspective), the
    two largest pairwise distances are the two ring diameters, so an
    empirically different ring ordering never changes the result.

    Args:
        landmarks: (N, 3) array from FaceResult.landmarks (x, y, z)
        ring_indices: one of RIGHT_IRIS_INDICES / LEFT_IRIS_INDICES

    Returns:
        Iris ring diameter in pixels (mean of the two largest pairwise ring
        distances), or 0.0 if the ring is degenerate (missing landmarks,
        all points coincident/zero).
    """
    if landmarks is None or len(landmarks) == 0:
        return 0.0

    ring = [i for i in ring_indices[1:] if i < len(landmarks)]
    if len(ring) < 3:
        logger.warning("Fewer than 3 iris ring landmarks available - iris diameter unavailable")
        return 0.0

    pts = landmarks[ring, :2].astype(np.float64)

    # All pairwise distances among the ring points
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(float(np.linalg.norm(pts[i] - pts[j])))

    largest = sorted(dists)[-2:]
    diameter = float(np.mean(largest))

    if diameter < 1e-6:
        logger.warning("Iris ring landmarks are coincident/zero - iris diameter unavailable")
        return 0.0

    return diameter


def compute_px_per_mm(iris_diameter_px: float) -> Optional[float]:
    """
    Establish the pixels-per-mm scale factor from the known anatomical iris
    diameter (~11.7mm, near-constant across humans).

    Args:
        iris_diameter_px: iris ring diameter in pixels

    Returns:
        Scale factor in px/mm, or None if the input is non-positive
        (missing/degenerate iris landmarks).
    """
    if iris_diameter_px is None or iris_diameter_px <= 0:
        logger.warning(
            f"Invalid iris diameter {iris_diameter_px}px - cannot establish px/mm scale"
        )
        return None
    return iris_diameter_px / AVERAGE_IRIS_DIAMETER_MM


def compute_eye_aspect_ratio(
    landmarks: np.ndarray,
    eye_contour_indices: Sequence[int]
) -> float:
    """
    Compute the eye aspect ratio (EAR) for blink detection.

    Geometric adaptation of the classic 6-point EAR formula that works with
    whichever contour points are available: the eye corners are taken as the
    two contour points farthest apart, and the eye height is the sum of the
    two maximum perpendicular distances from the corner-to-corner line (one
    per side of the line). EAR = height / width.

    This is equivalent to the classic formula (v1 + v2) / (2h): for a
    symmetric open eye the two vertical distances v1, v2 are each ~half the
    full eye height, giving the same ratio. EAR collapses toward ~0 when the
    eye closes (both vertical distances shrink), which is the blink signal.

    Args:
        landmarks: (N, 3) array from FaceResult.landmarks
        eye_contour_indices: FaceDetector.LEFT_EYE_LANDMARKS or
            FaceDetector.RIGHT_EYE_LANDMARKS

    Returns:
        EAR float >= 0 (0 for degenerate inputs)
    """
    if landmarks is None or len(landmarks) == 0:
        return 0.0

    valid = [i for i in eye_contour_indices if i < len(landmarks)]
    if len(valid) < 3:
        logger.warning("Fewer than 3 eye contour landmarks available - EAR unavailable")
        return 0.0

    pts = landmarks[valid, :2].astype(np.float64)
    n = len(pts)

    # Eye corners = farthest pair of contour points
    max_dist = -1.0
    corner_i = corner_j = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d > max_dist:
                max_dist = d
                corner_i, corner_j = i, j

    width = max_dist
    if width < 1e-6:
        return 0.0

    p1, p2 = pts[corner_i], pts[corner_j]
    line = p2 - p1

    # Signed perpendicular distance of every contour point from the
    # corner-to-corner line. Sign splits the points into the two eyelids.
    # (2D scalar cross product written out - np.cross with 2D vectors is
    # deprecated in NumPy 2.0.)
    signed = (line[0] * (pts[:, 1] - p1[1]) - line[1] * (pts[:, 0] - p1[0])) / width
    d_top = float(np.max(signed))
    d_bottom = float(-np.min(signed))

    return (d_top + d_bottom) / width


def is_blinking(ear: float, threshold: float = 0.2) -> bool:
    """
    Threshold check for blink detection.

    Args:
        ear: eye aspect ratio
        threshold: EAR below which the eye counts as closed/blinking

    Returns:
        True if ear < threshold (eye closed/blinking)
    """
    return ear < threshold


def segment_pupil(
    eye_crop_gray: np.ndarray,
    expected_radius_px: Optional[float]
) -> Optional[float]:
    """
    Detect the pupil radius in pixels inside a grayscale eye crop.

    Pipeline: CLAHE (lighting compensation) -> Otsu threshold (pupil is the
    darkest region) -> contour detection -> circularity filter -> pick the
    candidate whose size is most plausible for a pupil.

    Args:
        eye_crop_gray: grayscale crop roughly centered on the iris
        expected_radius_px: expected pupil radius upper bound in pixels
            (derived by the caller from the iris diameter and
            MAX_PLAUSIBLE_PUPIL_RATIO). Used to reject blobs that are too
            big to be a pupil (e.g. the iris itself or a shadow). May be
            None/<=0 when no bound is known - then the largest sufficiently
            circular candidate is returned instead.

    Returns:
        Detected pupil radius in pixels, or None if no plausible circular
        dark region was found.
    """
    if eye_crop_gray is None or eye_crop_gray.size == 0:
        logger.warning("Empty eye crop - cannot segment pupil")
        return None

    # 1. CLAHE lighting compensation
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    equalized = clahe.apply(eye_crop_gray)

    # 2. Otsu threshold - isolate the darkest region (pupil)
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Find contours, keep only roughly circular ones
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area <= 0 or perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < CIRCULARITY_THRESHOLD:
            continue
        _, radius = cv2.minEnclosingCircle(cnt)
        if radius >= MIN_PUPIL_RADIUS_PX:
            candidates.append(float(radius))

    if not candidates:
        return None

    # 4. Constrain the search to the plausible pupil size range
    if expected_radius_px is not None and expected_radius_px > 0:
        valid = [r for r in candidates if r <= expected_radius_px * PUPIL_RADIUS_TOLERANCE]
        if not valid:
            # Everything found exceeds the plausible pupil size - the
            # segmentation likely grabbed the iris or a shadow. Report no
            # pupil rather than a wrong one.
            return None
        if len(valid) == 1:
            return valid[0]
        # Multiple plausible blobs - pick the closest to the expected size
        return min(valid, key=lambda r: abs(r - expected_radius_px))

    # No expected bound - return the largest sufficiently circular candidate
    return max(candidates)


def compute_dilation_mm(
    pupil_radius_px: float,
    px_per_mm: Optional[float]
) -> Optional[float]:
    """
    Convert a pupil radius in pixels to pupil DIAMETER in mm.

    Args:
        pupil_radius_px: detected pupil radius in pixels
        px_per_mm: scale factor from compute_px_per_mm

    Returns:
        Pupil diameter in mm, or None if the scale factor is invalid.
    """
    if px_per_mm is None or px_per_mm <= 0 or pupil_radius_px is None:
        return None
    return (pupil_radius_px * 2.0) / px_per_mm

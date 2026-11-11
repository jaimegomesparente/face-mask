"""
Face detection and video anonymization engine.
Uses YOLOv8-Face (ONNX) for deep-learning face detection + Persistent Multi-Target ID Tracking.
"""

import math
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Restrict OpenCV's FFmpeg backend to local files: a crafted "video" (e.g. an HLS
# playlist disguised as .mp4) must not make the demuxer open other files or URLs.
# Read by OpenCV each time a capture is opened.
_cv_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
if "protocol_whitelist" not in _cv_opts:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(filter(None, [_cv_opts, "protocol_whitelist;file"]))

import cv2  # noqa: E402
import numpy as np

BBox = Tuple[int, int, int, int]
FBox = Tuple[float, float, float, float]

# Container formats accepted as input, as reported by ffprobe (format_name).
ALLOWED_CONTAINERS = {
    "mov,mp4,m4a,3gp,3g2,mj2",
    "matroska,webm",
    "avi",
}


def check_input_container(path: str) -> str:
    """
    Probe the real container format of a file (by content, not extension) and reject
    anything other than MP4/MOV, MKV/WebM or AVI, e.g. playlists or concat scripts
    renamed to .mp4.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-protocol_whitelist", "file",
        "-show_entries", "format=format_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        os.path.abspath(path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fmt = res.stdout.strip()
    if res.returncode != 0 or fmt not in ALLOWED_CONTAINERS:
        raise ValueError(f"Unsupported or invalid video container: {fmt or res.stderr.strip() or 'unknown'}")
    return fmt


@dataclass
class Detection:
    bbox: FBox                    # x, y, w, h
    score: float
    reject: Optional[str] = None  # None if accepted


class YOLOv8FaceDetector:
    """
    High-performance face detector using YOLOv8-Face ONNX.
    Robust to profile angles, motion blur, and variable lighting.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        conf_thresh: float = 0.30,
        nms_thresh: float = 0.45
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Detection]:
        h, w = frame.shape[:2]
        conf_limit = conf_threshold if conf_threshold is not None else self.conf_thresh
        tw = th = self.input_size

        # Letterbox to maintain aspect ratio without distortion
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        top = (th - nh) // 2
        bottom = th - nh - top
        left = (tw - nw) // 2
        right = tw - nw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (tw, th), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        if preds is None or len(preds) == 0:
            return []

        out = preds[0]
        if out.shape[0] < out.shape[1]:
            out = out.transpose()

        boxes = []
        scores = []
        for row in out:
            score = float(row[4])
            if score >= conf_limit:
                cx, cy, bw, bh = row[:4]
                x1 = int(round((cx - 0.5 * bw - left) / scale))
                y1 = int(round((cy - 0.5 * bh - top) / scale))
                bw_orig = int(round(bw / scale))
                bh_orig = int(round(bh / scale))

                # Boundary clipping
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                bw_orig = max(1, min(w - x1, bw_orig))
                bh_orig = max(1, min(h - y1, bh_orig))

                boxes.append([x1, y1, bw_orig, bh_orig])
                scores.append(score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_limit, self.nms_thresh)
        if len(indices) == 0:
            return []

        results: List[Detection] = []
        for idx in indices:
            i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
            bx, by, bw, bh = boxes[i]
            if min(bw, bh) < 10.0:
                continue
            results.append(Detection(bbox=(float(bx), float(by), float(bw), float(bh)), score=scores[i]))

        return results


class PersistentTracker:
    """
    Multi-target temporal tracker assigning persistent IDs to faces.
    Interpolates brief losses (profile turns / occlusions) and smooths trajectories.
    Prevents background sticking and phantom decor anchoring.
    """

    def __init__(
        self,
        max_gap: int = 15,
        max_dist: float = 0.65,
        max_size_ratio: float = 1.65,
        min_hits: int = 3,
        high_conf: float = 0.40,
        profile_conf: float = 0.28,
        hold_frames: int = 3,
        lead_frames: int = 3,
        max_interp_gap: int = 15,
        min_ref_size: float = 40.0,
    ):
        self.max_gap = max_gap
        self.max_dist = max_dist
        self.max_size_ratio = max_size_ratio
        self.min_hits = min_hits
        self.high_conf = high_conf
        self.profile_conf = profile_conf
        self.hold_frames = hold_frames
        self.lead_frames = lead_frames
        self.max_interp_gap = max_interp_gap
        self.min_ref_size = min_ref_size

    def _dist(self, a: FBox, b: FBox, limit: float) -> Optional[float]:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ref = (max(aw, ah) + max(bw, bh)) / 2.0
        if ref <= 0:
            return None
        sa = math.sqrt(max(aw * ah, 1e-6))
        sb = math.sqrt(max(bw * bh, 1e-6))
        # Adaptive size ratio: smaller boxes naturally have higher relative discretization variance
        ratio_limit = self.max_size_ratio if ref >= 40.0 else (self.max_size_ratio + 0.25)
        if max(sa, sb) / min(sa, sb) > ratio_limit:
            return None
        # Effective reference size incorporates camera/body motion floor
        eff_ref = max(ref, self.min_ref_size)
        d = math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2)) / eff_ref
        return d if d <= limit else None

    def _link(self, per_frame: List[List[Detection]]) -> List[dict]:
        tracklets: List[dict] = []
        active: List[dict] = []
        next_track_id = 1

        for f, dets in enumerate(per_frame):
            cands = [d for d in dets if d.reject is None]
            active = [t for t in active if f - t['last'] <= self.max_gap]

            pairs = []
            for ti, t in enumerate(active):
                dt = f - t['last']
                limit = min(0.90, 0.75 + 0.08 * (dt - 1)) if dt <= 3 else min(0.80, 0.65 + 0.03 * (dt - 1))
                for di, d in enumerate(cands):
                    dist = self._dist(t['bbox'], d.bbox, limit)
                    if dist is not None:
                        pairs.append((dist, ti, di))
            pairs.sort()

            used_t, used_d = set(), set()
            for _, ti, di in pairs:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                t, d = active[ti], cands[di]
                t['obs'][f] = (d.bbox, d.score)
                t['bbox'] = d.bbox
                t['last'] = f

            for di, d in enumerate(cands):
                if di in used_d:
                    continue
                # Do not spawn a new track if it already overlaps an active person
                if any(self._dist(t['bbox'], d.bbox, 0.45) is not None for t in active):
                    continue
                t = {
                    'id': next_track_id,
                    'obs': {f: (d.bbox, d.score)},
                    'bbox': d.bbox,
                    'last': f
                }
                next_track_id += 1
                tracklets.append(t)
                active.append(t)

        return tracklets

    def _keep(self, t: dict, conf_threshold: Optional[float] = None) -> bool:
        scores = [s for _, s in t['obs'].values()]
        n = len(scores)
        mx = max(scores)

        # Anonymization must never mask less than the per-frame preview: any track
        # containing a detection above the user's threshold is always kept.
        if conf_threshold is not None and mx >= conf_threshold:
            return True

        if n < self.min_hits:
            return False
        frames = sorted(t['obs'].keys())
        span = frames[-1] - frames[0] + 1
        hit_rate = n / span
        # Reject tracks with very low temporal density (e.g. sporadic background false alarms)
        if span > 10 and hit_rate < 0.25:
            return False

        mean = sum(scores) / n
        return mx >= self.high_conf or (n >= 4 and mean >= self.profile_conf)

    def _expand_track(self, t: dict, n_frames: int, frame_size: Tuple[int, int] = (0, 0)) -> List[Tuple[int, FBox, int, bool]]:
        frames = sorted(t['obs'])
        obs_dict = {f: t['obs'][f][0] for f in frames}

        # Segment track if temporal gap is too large to interpolate smoothly, or speed jump
        segments: List[List[int]] = []
        cur_seg = [frames[0]]
        for i in range(1, len(frames)):
            f_prev = frames[i - 1]
            f_cur = frames[i]
            dt = f_cur - f_prev
            b_prev = obs_dict[f_prev]
            b_cur = obs_dict[f_cur]

            dist = math.hypot((b_cur[0] + b_cur[2]/2) - (b_prev[0] + b_prev[2]/2),
                              (b_cur[1] + b_cur[3]/2) - (b_prev[1] + b_prev[3]/2))
            ref_size = max(b_prev[2], b_prev[3], self.min_ref_size)
            speed = dist / dt if dt > 0 else 0.0

            if dt > self.max_interp_gap or (dt > 1 and speed > 1.5 * ref_size):
                segments.append(cur_seg)
                cur_seg = [f_cur]
            else:
                cur_seg.append(f_cur)
        segments.append(cur_seg)

        out_tracks: List[Tuple[int, FBox, int, bool]] = []
        for seg_frames in segments:
            first, last = seg_frames[0], seg_frames[-1]
            xs = np.array(seg_frames, dtype=np.float64)
            boxes = np.array([obs_dict[f] for f in seg_frames], dtype=np.float64)

            full = np.arange(first, last + 1)
            interp = np.stack([np.interp(full, xs, boxes[:, i]) for i in range(4)], axis=1)

            # Extend each segment with a few frames on both sides, using the nearest box.
            # The detector typically fires a few frames after a face appears and stops a few
            # frames before it leaves (including through a frame border): over-masking a
            # stale area is harmless, a visible face is not.
            start = max(0, first - self.lead_frames)
            end = min(n_frames - 1, last + self.hold_frames)
            parts = []
            if start < first:
                parts.append(np.repeat(interp[:1], first - start, axis=0))
            parts.append(interp)
            if end > last:
                parts.append(np.repeat(interp[-1:], end - last, axis=0))
            seq = np.vstack(parts)

            # Apply 3-point centered smoothing [0.20, 0.60, 0.20] to suppress pixel-quantization jitter
            if len(seq) >= 3:
                kernel = np.array([0.20, 0.60, 0.20])
                smoothed_seq = np.empty_like(seq)
                for coord_i in range(4):
                    padded = np.pad(seq[:, coord_i], (1, 1), mode='edge')
                    smoothed_seq[:, coord_i] = np.convolve(padded, kernel, mode='valid')
                seq = smoothed_seq

            idx = np.arange(start, end + 1)
            detected = set(seg_frames)
            for k_idx, fr in enumerate(idx):
                fr_int = int(fr)
                box = (float(seq[k_idx, 0]), float(seq[k_idx, 1]), float(seq[k_idx, 2]), float(seq[k_idx, 3]))
                out_tracks.append((fr_int, box, t['id'], fr_int in detected))

        return out_tracks

    def build(
        self,
        per_frame: List[List[Detection]],
        frame_size: Tuple[int, int] = (0, 0),
        conf_threshold: Optional[float] = None
    ) -> Dict[int, List[Tuple[FBox, Optional[int], bool]]]:
        n = len(per_frame)
        out: Dict[int, List[Tuple[FBox, Optional[int], bool]]] = {}
        for t in self._link(per_frame):
            if not self._keep(t, conf_threshold):
                continue
            for fr, bbox, track_id, det in self._expand_track(t, n, frame_size):
                out.setdefault(fr, []).append((bbox, track_id, det))

        # Safety net: every raw detection above the user's threshold is masked on its
        # own frame, even if the linker dropped it (e.g. suppressed next to another track).
        if conf_threshold is not None:
            for fr, dets in enumerate(per_frame):
                for d in dets:
                    if d.reject is None and d.score >= conf_threshold:
                        out.setdefault(fr, []).append((d.bbox, None, True))
        return out


class FaceMasker:
    """Core face detection, masking, and persistent ID visualization engine."""

    DEFAULT_PIXEL_SIZE = 16
    # Maximum number of mosaic blocks along the longest side of a masked region.
    # Keeps large (close-up) faces from staying recognizable behind a fine mosaic.
    MAX_MOSAIC_BLOCKS = 10
    DEFAULT_PADDING_RATIO = 0.22
    DEFAULT_BOX_COLOR = (60, 255, 0)
    DEFAULT_BOX_THICKNESS = 1
    DEFAULT_CONF_THRESHOLD = 0.30

    def __init__(self, model_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            yolo_path = os.path.join(base_dir, "yolov8n-face.onnx")
            yunet_path = os.path.join(base_dir, "face_detection_yunet_2023mar.onnx")
            model_path = yolo_path if os.path.exists(yolo_path) else yunet_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = model_path
        self.is_yolo = "yolo" in os.path.basename(model_path).lower()
        # cv2.dnn networks / FaceDetectorYN are not thread-safe (setInput + forward).
        # Streamlit shares this instance across sessions, each running in its own thread.
        self._lock = threading.Lock()
        if self.is_yolo:
            self._detector = YOLOv8FaceDetector(model_path, conf_thresh=self.DEFAULT_CONF_THRESHOLD)
        else:
            self._detector = None
            self._current_size = (0, 0)

    def _get_yunet_detector(self, width: int, height: int, conf_threshold: float) -> cv2.FaceDetectorYN:
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(width, height),
                score_threshold=conf_threshold,
                nms_threshold=0.35,
                top_k=5000
            )
            self._current_size = (width, height)
        else:
            self._detector.setScoreThreshold(conf_threshold)
            if self._current_size != (width, height):
                self._detector.setInputSize((width, height))
                self._current_size = (width, height)
        return self._detector

    def detect_faces(self, frame: np.ndarray, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> List[Detection]:
        with self._lock:
            return self._detect_faces_unlocked(frame, conf_threshold)

    def _detect_faces_unlocked(self, frame: np.ndarray, conf_threshold: float) -> List[Detection]:
        if self.is_yolo:
            return self._detector.detect(frame, conf_threshold=conf_threshold)

        h, w = frame.shape[:2]
        detector = self._get_yunet_detector(w, h, conf_threshold=conf_threshold)
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        results: List[Detection] = []
        for f in faces:
            x, y, fw, fh = (float(v) for v in f[:4])
            score = float(f[14])
            results.append(Detection((x, y, fw, fh), score, None))
        return results

    @staticmethod
    def expand_bbox_for_head(
        bbox: FBox,
        frame_shape: Tuple[int, ...],
        padding_ratio: float = DEFAULT_PADDING_RATIO
    ) -> BBox:
        frame_h, frame_w = frame_shape[:2]
        x, y, w, h = (int(round(v)) for v in bbox)

        pad_x = int(w * padding_ratio * 1.35)
        pad_top = int(h * padding_ratio * 1.30)
        pad_bottom = int(h * padding_ratio * 0.75)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_top)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_bottom)

        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def apply_pixelation(image: np.ndarray, x: int, y: int, w: int, h: int, pixel_size: int = DEFAULT_PIXEL_SIZE) -> None:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            return

        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Block size grows with the region so a face never gets more than
        # MAX_MOSAIC_BLOCKS blocks across, whatever its size on screen.
        block = max(pixel_size, math.ceil(max(roi_w, roi_h) / FaceMasker.MAX_MOSAIC_BLOCKS))
        small_w = max(1, roi_w // block)
        small_h = max(1, roi_h // block)

        # INTER_AREA averages every pixel of a block (INTER_LINEAR only samples a few)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        image[y1:y2, x1:x2] = pixelated

    @staticmethod
    def draw_tracking_box(
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color_bgr: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
        thickness: int = DEFAULT_BOX_THICKNESS
    ) -> None:
        cv2.rectangle(image, (x, y), (x + w, y + h), color_bgr, thickness)

    @staticmethod
    def draw_id_badge(
        image: np.ndarray,
        x: int,
        y: int,
        track_id: int,
        color_bgr: Tuple[int, int, int] = DEFAULT_BOX_COLOR
    ) -> None:
        label = f"ID {track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

        pad_x = 4
        pad_y = 3
        badge_y2 = max(th + 2 * pad_y, y - 2)
        badge_y1 = badge_y2 - th - 2 * pad_y
        badge_x1 = max(0, x)
        badge_x2 = min(image.shape[1], x + tw + 2 * pad_x)

        cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 18, 18), -1)
        cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), color_bgr, 1)
        cv2.putText(
            image, label,
            (badge_x1 + pad_x, badge_y2 - pad_y),
            font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
        )

    def _render(
        self,
        image: np.ndarray,
        boxes: List[Tuple[Tuple[BBox, BBox], Optional[int], bool]],
        pixel_size: int,
        box_color_bgr: Tuple[int, int, int],
        box_thickness: int
    ) -> int:
        # 1. Pixelate every box, overlapping ones included: pixelating twice is harmless,
        #    dropping a box could leave part of a face visible.
        #    The whole head box (face + forehead, hairline, ears, chin) is pixelated.
        for (_, (hx, hy, hw, hh)), _, _ in boxes:
            self.apply_pixelation(image, hx, hy, hw, hh, pixel_size=pixel_size)

        # 2. Draw thin green head tracking box + ID badge, deduplicated (overlay only)
        drawn = self._dedupe_frame_boxes(boxes)
        for (_, (hx, hy, hw, hh)), track_id, _ in drawn:
            self.draw_tracking_box(image, hx, hy, hw, hh, color_bgr=box_color_bgr, thickness=box_thickness)
            if track_id is not None:
                self.draw_id_badge(image, hx, hy, track_id, color_bgr=box_color_bgr)
        return len(drawn)

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        iw = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        ih = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter = iw * ih
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _dedupe_frame_boxes(
        cls,
        boxes: List[Tuple[Tuple[BBox, BBox], Optional[int], bool]]
    ) -> List[Tuple[Tuple[BBox, BBox], Optional[int], bool]]:
        """
        Guarantees that no two overlay boxes are drawn on the same person on the same frame.
        Real YOLO detections always take priority over interpolated ones.
        Only used for drawing: pixelation is always applied to every box.
        """
        if len(boxes) <= 1:
            return boxes

        # Sort: actual detections first (det == True), then smaller track_id
        sorted_boxes = sorted(boxes, key=lambda b: (b[2] is True, -(b[1] or 9999)), reverse=True)
        kept = []
        for candidate in sorted_boxes:
            ((cfx, cfy, cfw, cfh), _), c_id, c_det = candidate
            overlap = False
            for ((kfx, kfy, kfw, kfh), _), k_id, k_det in kept:
                face_iou = cls._iou((cfx, cfy, cfw, cfh), (kfx, kfy, kfw, kfh))
                cdist = math.hypot((cfx + cfw / 2) - (kfx + kfw / 2), (cfy + cfh / 2) - (kfy + kfh / 2))
                ref = min(max(cfw, cfh), max(kfw, kfh))
                if face_iou > 0.60 or (ref > 0 and cdist < 0.35 * ref):
                    overlap = True
                    break
            if not overlap:
                kept.append(candidate)

        return kept

    def process_frame(
        self,
        frame: np.ndarray,
        pixel_size: int = DEFAULT_PIXEL_SIZE,
        padding_ratio: float = DEFAULT_PADDING_RATIO,
        box_color_bgr: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
        box_thickness: int = DEFAULT_BOX_THICKNESS,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        debug: bool = False
    ) -> Tuple[np.ndarray, int]:
        out_frame = frame.copy()
        dets = self.detect_faces(out_frame, conf_threshold=conf_threshold)

        boxes = []
        for i, d in enumerate(dets):
            if d.reject is not None or d.score < conf_threshold:
                continue
            fx, fy, fw, fh = (int(round(v)) for v in d.bbox)
            head_box = self.expand_bbox_for_head(d.bbox, out_frame.shape, padding_ratio)
            boxes.append((((fx, fy, fw, fh), head_box), i + 1, True))

        face_count = self._render(out_frame, boxes, pixel_size, box_color_bgr, box_thickness)
        return out_frame, face_count

    def _scan_video(
        self,
        input_path: str,
        conf_threshold: float,
        progress_callback: Optional[Callable[[int, int], None]],
        total_frames: int
    ) -> List[List[Detection]]:
        cap = cv2.VideoCapture(input_path)
        per_frame: List[List[Detection]] = []
        scan_conf = min(0.20, conf_threshold)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                per_frame.append(self.detect_faces(frame, conf_threshold=scan_conf))
                if progress_callback and total_frames > 0:
                    progress_callback(min(len(per_frame), total_frames), total_frames * 2)
        finally:
            cap.release()
        return per_frame

    def process_video(
        self,
        input_path: str,
        output_path: str,
        pixel_size: int = DEFAULT_PIXEL_SIZE,
        padding_ratio: float = DEFAULT_PADDING_RATIO,
        box_color_bgr: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
        box_thickness: int = DEFAULT_BOX_THICKNESS,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        use_tracking: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        debug: bool = False
    ) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        check_input_container(input_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # ---- Pass 1: Detection & Multi-Object Tracking ----
        masks: Dict[int, List[Tuple[FBox, int, bool]]] = {}
        if use_tracking:
            per_frame = self._scan_video(input_path, conf_threshold, progress_callback, total_frames)
            masks = PersistentTracker().build(per_frame, frame_size=(width, height), conf_threshold=conf_threshold)

        # ---- Pass 2: Rendering & Export ----
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, f"temp_video_{os.getpid()}_{np.random.randint(10000, 99999)}.mp4")

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        offset = total_frames if use_tracking else 0
        denom = total_frames * 2 if use_tracking else total_frames

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if use_tracking:
                    raw_tracks = masks.get(frame_idx, [])
                    boxes = [
                        (
                            (
                                (int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))),
                                self.expand_bbox_for_head(b, frame.shape, padding_ratio)
                            ),
                            track_id,
                            det
                        )
                        for b, track_id, det in raw_tracks
                    ]
                else:
                    dets = self.detect_faces(frame, conf_threshold=conf_threshold)
                    boxes = [
                        (
                            (
                                (int(round(d.bbox[0])), int(round(d.bbox[1])), int(round(d.bbox[2])), int(round(d.bbox[3]))),
                                self.expand_bbox_for_head(d.bbox, frame.shape, padding_ratio)
                            ),
                            i + 1,
                            True
                        )
                        for i, d in enumerate(dets)
                    ]

                self._render(frame, boxes, pixel_size, box_color_bgr, box_thickness)
                writer.write(frame)
                frame_idx += 1

                if progress_callback and denom > 0:
                    progress_callback(min(offset + frame_idx, denom), denom)

        finally:
            cap.release()
            writer.release()

        try:
            self._mux_audio_with_ffmpeg(temp_video_path, input_path, output_path)
        finally:
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass

        return output_path

    @staticmethod
    def _mux_audio_with_ffmpeg(processed_video_path: str, original_video_path: str, output_path: str) -> None:
        # Absolute paths: never start with "-" (option injection) and never parsed as
        # a protocol; inputs restricted to local files.
        processed_video_path = os.path.abspath(processed_video_path)
        original_video_path = os.path.abspath(original_video_path)
        output_path = os.path.abspath(output_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-protocol_whitelist", "file",
            "-i", processed_video_path,
            "-protocol_whitelist", "file",
            "-i", original_video_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            cmd_fallback = [
                "ffmpeg",
                "-y",
                "-protocol_whitelist", "file",
                "-i", processed_video_path,
                "-protocol_whitelist", "file",
                "-i", original_video_path,
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                output_path
            ]
            subprocess.run(cmd_fallback, check=True)
"""Plate detection (YOLOv9 ONNX) + OCR (fast-plate-ocr) + multi-frame tracking.

Each physical plate gets a track. Every reading of that plate (even blurry or
partial ones) is accumulated and the most reliable text is chosen by
confidence-weighted voting, so the result improves as the plate gets closer.
"""
import warnings
from collections import defaultdict

import cv2
import numpy as np

warnings.filterwarnings("ignore")

DETECTOR_MODEL = "yolo-v9-s-608-license-plate-end2end"
OCR_MODEL = "cct-s-v2-global-model"


class Track:
    def __init__(self, tid, box, frame_no):
        self.id = tid
        self.box = box
        self.first_frame = frame_no
        self.last_seen = frame_no
        self.hits = 0
        self.votes = defaultdict(float)   # text -> summed confidence
        self.count = defaultdict(int)     # text -> occurrences
        self.best_text = ""
        self.best_conf = 0.0

    def add(self, text, conf):
        if not text:
            return False
        self.votes[text] += conf
        self.count[text] += 1
        # weight: summed confidence favours texts seen often AND confidently
        best = max(self.votes, key=lambda t: (self.votes[t], len(t)))
        changed = best != self.best_text
        self.best_text = best
        self.best_conf = self.votes[best] / self.count[best]
        return changed


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PlateReader:
    def __init__(self, det_conf=0.25, min_text_len=4, max_missed=20):
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector

        self.detector = create_detector(DETECTOR_MODEL, conf_thresh=det_conf)
        self.ocr = LicensePlateRecognizer(OCR_MODEL, device="cpu")
        self.rgb = self.ocr.config.image_color_mode == "rgb"
        self.min_len = min_text_len
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks, self.next_id = [], 1

    def _read(self, crop):
        if self.rgb:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pred = self.ocr.run(img, return_confidence=True)[0]
        text = pred.plate.strip().upper()
        conf = float(np.mean(pred.char_probs)) if pred.char_probs is not None and len(text) else 0.0
        return text, conf

    def process(self, frame, frame_no):
        """Return (list of (track, box, is_new_track, text_changed), )."""
        h, w = frame.shape[:2]
        dets = self.detector.predict(frame)
        events = []
        used = set()
        for d in sorted(dets, key=lambda d: -d.confidence):
            bb = d.bounding_box
            box = (max(0, bb.x1), max(0, bb.y1), min(w, bb.x2), min(h, bb.y2))
            if box[2] - box[0] < 8 or box[3] - box[1] < 4:
                continue
            # match to existing track (IoU, then centre distance)
            best, best_s = None, 0.0
            for t in self.tracks:
                if t.id in used:
                    continue
                s = _iou(box, t.box)
                if s < 0.15:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    tx, ty = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
                    bw = max(box[2] - box[0], t.box[2] - t.box[0])
                    if abs(cx - tx) < bw * 1.2 and abs(cy - ty) < bw * 0.6:
                        s = 0.15
                if s > best_s:
                    best, best_s = t, s
            new = False
            if best is None:
                best = Track(self.next_id, box, frame_no)
                self.next_id += 1
                self.tracks.append(best)
                new = True
            best.box, best.last_seen = box, frame_no
            best.hits += 1
            used.add(best.id)

            # pad the crop slightly: helps partially cut/blurry plates
            pw, ph = int((box[2] - box[0]) * 0.06), int((box[3] - box[1]) * 0.12)
            crop = frame[max(0, box[1] - ph):min(h, box[3] + ph), max(0, box[0] - pw):min(w, box[2] + pw)]
            if crop.size == 0:
                continue
            text, conf = self._read(crop)
            changed = False
            if len(text) >= self.min_len:
                changed = best.add(text, conf)
            events.append((best, box, new, changed))
        self.tracks = [t for t in self.tracks if frame_no - t.last_seen <= self.max_missed]
        return events

"""Baby-face filter: geometric face warps plus skin smoothing and blush.

Works on a BGR frame and a (478, 2) array of MediaPipe Face Landmarker
points in pixel coordinates. All warps are expressed as a single backward
displacement field (destination pixel -> source pixel) that is applied with
one cv2.remap call, limited to the region around the face.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# MediaPipe face mesh landmark indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
             160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
            385, 384, 398]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
        267, 0, 37, 39, 40, 185]
FOREHEAD_TOP, CHIN, BETWEEN_EYES, NOSE_TIP = 10, 152, 168, 1
FACE_RIGHT_EDGE, FACE_LEFT_EDGE = 234, 454
NOSE_WINGS = (98, 327)
MOUTH_CORNERS = (61, 291)
MOUTH_INNER = (13, 14)
CHIN_SIDES = (150, 379)
CHEEK_EDGES = (93, 323)
CHEEKS = (50, 280)


@dataclass
class Settings:
    strength: float = 1.0   # global multiplier for all geometric warps
    eyes: float = 0.40      # eye enlargement
    nose: float = 0.35      # nose shrink
    mouth: float = 0.25     # mouth shrink
    chin: float = 0.10      # shorten the lower face
    round: float = 0.06     # rounder face: narrower chin, fuller cheeks
    head: float = 0.12      # bigger forehead / cranium
    smooth: float = 0.65    # skin smoothing
    blush: float = 0.30     # pink cheeks


def _norm(v):
    n = np.hypot(*v)
    return v / n if n > 1e-6 else v


class BabyFaceFilter:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self._grid_shape = None
        self._gx = self._gy = None

    def _grids(self, h, w):
        if self._grid_shape != (h, w):
            self._gx, self._gy = np.meshgrid(np.arange(w, dtype=np.float32),
                                             np.arange(h, dtype=np.float32))
            self._grid_shape = (h, w)
        return self._gx, self._gy

    def apply(self, frame, faces, amount=1.0):
        """Apply the filter in place to every face; `amount` fades it in/out.

        Returns the landmarks moved along with the warp, so later stages can
        work on the warped face.
        """
        if amount <= 0:
            return list(faces)
        return [self._apply_one(frame, lm.astype(np.float32), amount) for lm in faces]

    def _warps(self, lm, fw, k):
        """Local warps as (kind, centre, radius, param) for one face."""
        s = self.settings
        up = _norm(lm[FOREHEAD_TOP] - lm[CHIN])
        center = lm[NOSE_TIP]
        warps = []
        for eye in (RIGHT_EYE, LEFT_EYE):
            pts = lm[eye]
            ew = np.linalg.norm(pts[0] - pts[8])  # corner to corner
            warps.append(("bulge", pts.mean(0), 1.15 * ew, s.eyes * k))
        nose_w = np.linalg.norm(lm[NOSE_WINGS[0]] - lm[NOSE_WINGS[1]])
        nose_c = (lm[NOSE_WINGS[0]] + lm[NOSE_WINGS[1]] + lm[NOSE_TIP]) / 3
        warps.append(("bulge", nose_c, 1.0 * nose_w, -s.nose * k))
        mouth_w = np.linalg.norm(lm[MOUTH_CORNERS[0]] - lm[MOUTH_CORNERS[1]])
        mouth_c = (lm[MOUTH_INNER[0]] + lm[MOUTH_INNER[1]]) / 2
        warps.append(("bulge", mouth_c, 0.8 * mouth_w, -s.mouth * k))
        forehead = lm[BETWEEN_EYES] + 0.6 * (lm[FOREHEAD_TOP] - lm[BETWEEN_EYES])
        warps.append(("bulge", forehead, 0.8 * fw, s.head * k))
        warps.append(("push", lm[CHIN], 0.6 * fw, up * (s.chin * k * fw)))
        for i in CHIN_SIDES:
            warps.append(("push", lm[i], 0.35 * fw,
                          _norm(center - lm[i]) * (s.round * k * fw)))
        for i in CHEEK_EDGES:
            warps.append(("push", lm[i], 0.35 * fw,
                          _norm(lm[i] - center) * (s.round * k * fw)))
        out = []
        for kind, c, r, p in warps:
            if r < 2 or not np.any(p):
                continue
            if kind == "push":
                # clamp to avoid fold-over
                p = np.asarray(p, np.float32)
                pn = np.hypot(*p)
                if pn > 0.6 * r:
                    p = p * (0.6 * r / pn)
            out.append((kind, c, r, p))
        return out

    @staticmethod
    def _offset(dx, dy, kind, r, p):
        """Backward offset (source - destination) of one warp at offsets dx, dy from its centre."""
        f = np.clip(1.0 - (dx * dx + dy * dy) / (r * r), 0.0, None)
        f *= f
        if kind == "bulge":
            # sample closer to the centre -> magnify (p > 0) or shrink (p < 0)
            return -dx * f * p, -dy * f * p
        # move content at the centre by vector p
        return -p[0] * f, -p[1] * f

    def _forward(self, pts, warps):
        """Where the warp moves source points: solve dst + offset(dst) = src."""
        dst = pts.copy()
        for _ in range(4):
            off = np.zeros_like(pts)
            for kind, c, r, p in warps:
                ox, oy = self._offset(dst[:, 0] - c[0], dst[:, 1] - c[1], kind, r, p)
                off[:, 0] += ox
                off[:, 1] += oy
            dst = pts - off
        return dst

    def _apply_one(self, frame, lm, amount):
        s = self.settings
        h, w = frame.shape[:2]
        fw = float(np.linalg.norm(lm[FACE_LEFT_EDGE] - lm[FACE_RIGHT_EDGE]))
        if fw < 20:
            return lm

        # Region of interest: face plus a margin wide enough for every warp
        oval = lm[FACE_OVAL]
        margin = 0.45 * fw
        x0 = int(max(0, oval[:, 0].min() - margin))
        y0 = int(max(0, oval[:, 1].min() - margin))
        x1 = int(min(w, oval[:, 0].max() + margin))
        y1 = int(min(h, oval[:, 1].max() + margin))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return lm
        roi = frame[y0:y1, x0:x1]

        # Skin look first (in source space), so the warp carries it along
        if s.smooth > 0 or s.blush > 0:
            self._skin(roi, lm - (x0, y0), fw, amount)

        warps = self._warps(lm, fw, s.strength * amount)
        if not warps:
            return lm
        gx, gy = self._grids(h, w)
        px, py = gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
        ox = np.zeros_like(px)
        oy = np.zeros_like(py)
        for kind, c, r, p in warps:
            # Only touch the warp's own bounding box inside the ROI
            bx0 = int(max(x0, c[0] - r)) - x0
            by0 = int(max(y0, c[1] - r)) - y0
            bx1 = int(min(x1, c[0] + r + 1)) - x0
            by1 = int(min(y1, c[1] + r + 1)) - y0
            if bx1 <= bx0 or by1 <= by0:
                continue
            sl = (slice(by0, by1), slice(bx0, bx1))
            dx, dy = self._offset(px[sl] - c[0], py[sl] - c[1], kind, r, p)
            ox[sl] += dx
            oy[sl] += dy

        frame[y0:y1, x0:x1] = cv2.remap(frame, px + ox, py + oy,
                                        cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
        return self._forward(lm, warps)

    def _skin(self, roi, lm, fw, amount):
        s = self.settings
        rh, rw = roi.shape[:2]
        # Work on a downscaled copy for speed
        scale = min(1.0, 160.0 / fw)
        sw, sh = max(1, int(rw * scale)), max(1, int(rh * scale))
        small_lm = lm * scale

        mask = np.zeros((sh, sw), np.float32)
        cv2.fillPoly(mask, [small_lm[FACE_OVAL].astype(np.int32)], 1.0)
        for part in (RIGHT_EYE, LEFT_EYE, RIGHT_BROW, LEFT_BROW, LIPS):
            hull = cv2.convexHull(small_lm[part].astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 0.0)
        sfw = fw * scale
        mask = cv2.erode(mask, None, iterations=max(1, int(sfw * 0.02)))
        mask = cv2.GaussianBlur(mask, (0, 0), sfw * 0.04)

        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
        size = (rw, rh)
        out = roi.astype(np.float32)

        if s.smooth > 0:
            # guided filter: edge-preserving like bilateral, but O(N)
            smooth = cv2.ximgproc.guidedFilter(small, small, max(2, int(sfw * 0.035)), 180.0)
            # slight baby glow: lift brightness a touch
            smooth = cv2.convertScaleAbs(smooth, alpha=1.04, beta=6)
            smooth = cv2.resize(smooth, size, interpolation=cv2.INTER_LINEAR)
            a = cv2.resize(mask * min(1.0, s.smooth * amount), size)[..., None]
            out = out * (1 - a) + smooth.astype(np.float32) * a

        if s.blush > 0:
            blob = np.zeros((sh, sw), np.float32)
            for i in CHEEKS:
                c = small_lm[i]
                # blush slightly below and outside the cheekbone point
                cv2.circle(blob, (int(c[0]), int(c[1] + 0.03 * sfw)),
                           max(1, int(0.11 * sfw)), 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), sfw * 0.07) * mask
            pink = np.array([150, 130, 255], np.float32)  # BGR
            a = cv2.resize(blob * (0.45 * min(1.0, s.blush * amount)), size)
            out = out * (1 - a[..., None]) + pink * a[..., None]

        np.clip(out, 0, 255, out=out)
        roi[:] = out.astype(np.uint8)

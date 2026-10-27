# Copyright (c) 2026 Azamat M
# Licensed under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).
# Non-commercial use only. Unlike the rest of this project, this file is not MIT.
"""Neural baby-face stage.

Uses the generator from "Lifespan Age Transformation Synthesis" (Or-El et al.,
ECCV 2020, https://github.com/royorel/Lifespan_Age_Transformation_Synthesis),
loaded from third_party/lifespan. That code and its weights are licensed
CC BY-NC-SA 4.0: non-commercial use only.

Per face and frame: crop and align the face the way the model was trained
(FFHQ-style crop, 256x256, background and clothes painted grey), translate it to
the 0-2 years age class on the GPU, and blend the result back into the frame.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from baby_filter import (FACE_LEFT_EDGE, FACE_OVAL, FACE_RIGHT_EDGE, LEFT_EYE,
                         MOUTH_CORNERS, NOSE_TIP, RIGHT_EYE)

HERE = Path(__file__).resolve().parent
LIFESPAN = HERE / "third_party" / "lifespan"
SEGMENTER_MODEL = HERE / "models" / "selfie_segmenter.tflite"

SIZE = 256
NUM_CLASSES = 6          # 0-2, 3-6, 7-9, 15-19, 30-39, 50-69
STYLE_PER_CLASS = 50


def _load_networks():
    spec = importlib.util.spec_from_file_location(
        "lifespan_networks", LIFESPAN / "models" / "networks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def align_matrix(lm):
    """2x3 affine from frame to the model's 256x256 crop (FFHQ rule, larger crop)."""
    eye_l = lm[RIGHT_EYE].mean(0)   # image-left eye
    eye_r = lm[LEFT_EYE].mean(0)
    eye_avg = (eye_l + eye_r) / 2
    eye_to_eye = eye_r - eye_l
    mouth_avg = (lm[MOUTH_CORNERS[0]] + lm[MOUTH_CORNERS[1]]) / 2
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 2.2)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    src = np.float32([c - x - y, c - x + y, c + x + y])
    dst = np.float32([(0, 0), (0, SIZE), (SIZE, SIZE)])
    return cv2.getAffineTransform(src, dst)


class BabyGAN:
    def __init__(self, gender="male", device="cuda", age=0.0):
        if not torch.cuda.is_available() and device == "cuda":
            raise RuntimeError("CUDA GPU not available")
        self.device = torch.device(device)
        self.half = self.device.type == "cuda"
        self.nets = _load_networks()
        self._generators = {}
        self.gender = gender
        self.age = age
        self._latent_key = None
        self._latent = None
        self._prev = []          # per-face previous output, for temporal smoothing
        self.temporal = 0.5      # 0 = no smoothing, 0.9 = very smooth
        self.blend = 1.0         # how much of the neural output to show

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self.segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
                output_confidence_masks=True))

    def warmup(self):
        """Load the model and let cuDNN pick kernels, so the first frame isn't slow."""
        for _ in range(3):
            self._run([np.full((SIZE, SIZE, 3), 128, np.uint8)])

    def reset(self):
        """Forget temporal state (after switching model or losing the face)."""
        self._prev = []

    # --- model -----------------------------------------------------------

    def _generator(self):
        g = self._generators.get(self.gender)
        if g is None:
            n = self.nets
            g = n.Generator(3, 3, ngf=64, style_dim=STYLE_PER_CLASS * NUM_CLASSES,
                            n_downsampling=2, id_enc_norm=n.PixelNorm,
                            padding_type="reflect", conv_weight_norm=True,
                            decoder_norm="pixel", actvn="lrelu",
                            normalize_mlp=True, modulated_conv=True)
            path = LIFESPAN / "checkpoints" / f"{self.gender}s_model" / "latest_net_g_running.pth"
            g.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            g.eval().to(self.device)
            if self.half:
                g.half()
            self._generators[self.gender] = g
        return g

    def _age_latent(self, g):
        """Style latent for a (possibly fractional) age class, computed once."""
        key = (self.gender, round(self.age, 2))
        if self._latent_key != key:
            lo = int(np.floor(self.age))
            hi = min(lo + 1, NUM_CLASSES - 1)
            t = self.age - lo
            cond = torch.zeros(2, STYLE_PER_CLASS * NUM_CLASSES, device=self.device)
            cond[0, lo * STYLE_PER_CLASS:(lo + 1) * STYLE_PER_CLASS] = 1
            cond[1, hi * STYLE_PER_CLASS:(hi + 1) * STYLE_PER_CLASS] = 1
            with torch.no_grad():
                lat = g.decoder.mlp(cond.to(next(g.parameters()).dtype))
            self._latent = (1 - t) * lat[:1] + t * lat[1:]
            self._latent_key = key
        return self._latent

    @torch.no_grad()
    def _run(self, crops):
        g = self._generator()
        latent = self._age_latent(g)
        x = torch.from_numpy(np.stack(crops)).to(self.device)
        x = x.permute(0, 3, 1, 2).flip(1)                   # BGR->RGB, NCHW
        x = x.to(latent.dtype) / 127.5 - 1.0
        d = g.decoder
        out = g.id_encoder(x)
        lat = latent.expand(x.shape[0], -1)
        for blk in (d.StyledConvBlock_0, d.StyledConvBlock_1, d.StyledConvBlock_2,
                    d.StyledConvBlock_3, d.StyledConvBlock_up0, d.StyledConvBlock_up1):
            out = blk(out, lat)
        out = d.conv_img(out)
        out = ((out.flip(1).permute(0, 2, 3, 1) + 1.0) * 127.5).clamp(0, 255)
        return out.float().cpu().numpy()

    # --- per frame -------------------------------------------------------

    def _mask_background(self, crop, lm, M):
        """Grey out what the model was trained without: background and clothes."""
        res = self.segmenter.segment(self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        keep = res.confidence_masks[0].numpy_view().reshape(SIZE, SIZE) > 0.5
        # Below the jawline keep only a neck-wide strip (the rest is clothes)
        oval = cv2.transform(lm[FACE_OVAL][None].astype(np.float32), M)[0]
        jaw_y = np.sort(oval[:, 1])[-8]            # lower jaw corners level
        lower = oval[oval[:, 1] >= jaw_y]
        cx = lower[:, 0].mean()
        half = 0.45 * (lower[:, 0].max() - lower[:, 0].min())
        face = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(face, [oval.astype(np.int32)], 1)
        ys, xs = np.ogrid[:SIZE, :SIZE]
        clothes = (ys > jaw_y) & (face == 0) & (np.abs(xs - cx) > half)
        crop[~keep | clothes] = 128

    def apply(self, frame, faces, amount=1.0):
        """Replace every face in `frame` (in place) with its baby version."""
        if not faces or amount <= 0:
            self._prev = []
            return frame
        crops, mats = [], []
        for lm in faces:
            M = align_matrix(lm)
            src = frame
            if np.sqrt(abs(np.linalg.det(M[:, :2]))) < 0.5:  # avoid aliasing on big faces
                src = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                M2 = M.copy()
                M2[:, :2] *= 2.0
            else:
                M2 = M
            crop = cv2.warpAffine(src, M2, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
            self._mask_background(crop, lm, M)
            crops.append(crop)
            mats.append(M)

        outs = self._run(crops)

        if len(self._prev) != len(outs):
            self._prev = [None] * len(outs)
        for i, (lm, M, out, crop) in enumerate(zip(faces, mats, outs, crops)):
            if self._prev[i] is not None and self.temporal > 0:
                out = self.temporal * self._prev[i] + (1 - self.temporal) * out
            self._prev[i] = out
            self._paste(frame, lm, M, out, crop, amount * self.blend)
        return frame

    def _paste(self, frame, lm, M, out, crop, alpha):
        h, w = frame.shape[:2]
        # The model only knows near-frontal faces: fade it out on strong head turns
        span = lm[FACE_LEFT_EDGE, 0] - lm[FACE_RIGHT_EDGE, 0]
        if abs(span) > 1:
            turn = abs((lm[NOSE_TIP, 0] - lm[FACE_RIGHT_EDGE, 0]) / span - 0.5) * 2
            alpha *= float(np.clip((0.75 - turn) / 0.35, 0.0, 1.0))
        if alpha <= 0.01:
            return
        # Blend region: face oval (incl. forehead), feathered, in crop space
        oval = cv2.transform(lm[FACE_OVAL][None].astype(np.float32), M)[0]
        mask = np.zeros((SIZE, SIZE), np.float32)
        cv2.fillPoly(mask, [oval.astype(np.int32)], 1.0)
        mask = cv2.GaussianBlur(cv2.erode(mask, None, iterations=4), (0, 0), 6)

        # Match overall colour/brightness to the live frame so the seam disappears
        m = mask > 0.5
        if m.sum() > 100:
            src = crop.astype(np.float32)[m]
            dst = out[m]
            out = out + (src.mean(0) - dst.mean(0)) * 0.7

        # Back to frame space, only over the face's bounding box
        Minv = cv2.invertAffineTransform(M)
        corners = cv2.transform(np.float32([[[0, 0], [SIZE, 0], [0, SIZE], [SIZE, SIZE]]]), Minv)[0]
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        Mroi = Minv.copy()
        Mroi[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        out_f = cv2.warpAffine(np.clip(out, 0, 255).astype(np.float32), Mroi, size,
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        mask_f = cv2.warpAffine(mask, Mroi, size, flags=cv2.INTER_LINEAR)[..., None] * alpha
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = np.clip(roi * (1 - mask_f) + out_f * mask_f, 0, 255).astype(np.uint8)

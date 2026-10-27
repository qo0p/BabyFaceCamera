#!/usr/bin/env python3
"""Baby Face Camera: webcam -> baby-face filter -> virtual webcam.

Reads a real webcam, applies the baby-face filter to every detected face and
streams the result to a v4l2loopback virtual camera that Zoom, Telegram,
browsers etc. can select as "BabyFace Camera".
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Quieter Qt (OpenCV's preview window) and MediaPipe logging; must precede imports
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

# The pip OpenCV build points Qt at an empty font folder on import, which leaves
# slider labels and tooltips blank. Point it at system fonts instead; Qt reads
# this when the first window opens, so it must come after `import cv2`.
for _fonts in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype"):
    if Path(_fonts).is_dir():
        os.environ["QT_QPA_FONTDIR"] = _fonts
        break

from baby_filter import BabyFaceFilter, Settings

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
CONFIG_FILE = (Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
               / "babyface-camera" / "settings.json")
WINDOW = "BabyFace Camera"
CONTROLS = "BabyFace Controls"

# Preview sliders: (label, target, field, max slider value); slider value = field * 100
WARP_SLIDERS = [
    ("Strength %", "warp", "strength", 200),
    ("Eyes", "warp", "eyes", 100),
    ("Nose", "warp", "nose", 100),
    ("Mouth", "warp", "mouth", 100),
    ("Chin", "warp", "chin", 30),
    ("Round", "warp", "round", 20),
    ("Head", "warp", "head", 30),
    ("Smooth", "warp", "smooth", 100),
    ("Blush", "warp", "blush", 100),
]
NEURAL_SLIDERS = [
    ("Baby AI %", "gan", "blend", 100),
    ("Age 0-2..3-6", "gan", "age", 100),
    ("Stability", "gan", "temporal", 90),
]

# In neural mode the network does most of the work; the warp only reshapes the
# head (shorter chin, rounder face) so the generated baby face fits it.
NEURAL_WARP = dict(eyes=0.15, nose=0.15, mouth=0.05, chin=0.12, round=0.06,
                   head=0.10, smooth=0.0, blush=0.15)
NEURAL_DEFAULTS = dict(blend=1.0, age=0.0, temporal=0.5)


class OneEuro:
    """One Euro filter for landmark arrays: steady when still, responsive in motion."""

    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(dt, cutoff):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))

    def __call__(self, x, t):
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(dt, cutoff)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="0",
                   help="real camera index, /dev/videoN path or video file (default 0)")
    p.add_argument("-o", "--output", default="/dev/video10",
                   help="v4l2loopback device to stream to (default /dev/video10)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--faces", type=int, default=1, help="max faces to filter")
    p.add_argument("--strength", type=float,
                   help="global warp strength, 0..2 (default: last used, else 1.0)")
    p.add_argument("--mode", choices=["auto", "neural", "classic"], default="auto",
                   help="neural: AI baby face on the GPU (realistic); classic: warp and "
                        "smoothing only (any CPU); auto: neural when a CUDA GPU is available")
    p.add_argument("--gender", choices=["male", "female"],
                   help="which neural model to use (key g switches it live; "
                        "default: last used, else male)")
    p.add_argument("--no-preview", action="store_true", help="run without a window")
    p.add_argument("--no-virtual", action="store_true",
                   help="preview only, don't open the virtual camera")
    p.add_argument("--mirror-output", action="store_true",
                   help="mirror the stream sent to apps (most apps mirror self-view already)")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--reset-settings", action="store_true",
                   help=f"start from defaults instead of the settings saved in {CONFIG_FILE}")
    return p.parse_args()


def open_capture(src, width, height, fps):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
    elif src.startswith("/dev/video"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"error: cannot open camera '{src}'")
    if src.isdigit() or src.startswith("/dev/video"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_virtual_cam(device, width, height, fps):
    import pyvirtualcam
    if not os.path.exists(device):
        sys.exit(f"error: {device} does not exist. Load the virtual camera driver first:\n"
                 f"  sudo ./setup_virtualcam.sh\n"
                 f"or run with --no-virtual for a preview only.")
    try:
        cam = pyvirtualcam.Camera(width, height, fps, device=device,
                                  fmt=pyvirtualcam.PixelFormat.BGR,
                                  backend="v4l2loopback")
    except Exception as e:  # pyvirtualcam raises RuntimeError with a useful text
        sys.exit(f"error: cannot open virtual camera {device}: {e}")
    print(f"Streaming to virtual camera {cam.device} ({width}x{height} @ {fps} fps)")
    return cam


def create_landmarker(model_path, faces):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if not Path(model_path).exists():
        sys.exit(f"error: face model not found at {model_path}\n"
                 f"download it with:\n  curl -L -o {model_path} {MODEL_URL}")
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.FaceLandmarker.create_from_options(opts)


def create_gan(mode, gender):
    if mode == "classic":
        return None
    try:
        import torch
        from baby_gan import BabyGAN
        torch.backends.cudnn.benchmark = True
        gan = BabyGAN(gender=gender)
        gan.warmup()
        print(f"Neural baby face: on ({torch.cuda.get_device_name(0)}, {gender} model)")
        return gan
    except Exception as e:
        if mode == "neural":
            sys.exit(f"error: neural mode unavailable: {e}\nsee README (Neural mode setup)")
        print(f"Neural mode unavailable ({e}); using the classic filter")
        return None


def defaults(args, neural):
    """Slider targets with their default values for the given mode."""
    strength = 1.0 if args.strength is None else args.strength
    warp = Settings(strength=strength, **(NEURAL_WARP if neural else {}))
    return {"warp": warp, "gan": SimpleNamespace(**NEURAL_DEFAULTS)}


class Prefs:
    """Slider values, mode and gender, remembered between runs."""

    def __init__(self, path, fresh=False):
        self.path = path
        self.data = {}
        if not fresh:
            try:
                self.data = json.loads(path.read_text())
                if not isinstance(self.data, dict):
                    raise ValueError("not a JSON object")
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as e:
                print(f"warning: ignoring saved settings {path}: {e}")
                self.data = {}
        self._written = self._dump()

    def _dump(self):
        return json.dumps(self.data, indent=2, sort_keys=True)

    def save(self):
        text = self._dump()
        if text == self._written:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(text + "\n")
            os.replace(tmp, self.path)  # atomic: never leaves a half-written file
        except OSError as e:
            print(f"warning: cannot save settings to {self.path}: {e}")
        self._written = text


def slider_values(obj, target):
    return {field: round(getattr(obj, field), 2)
            for _, t, field, _ in WARP_SLIDERS + NEURAL_SLIDERS if t == target}


def load_values(obj, target, values):
    """Copy known, in-range fields from a saved dict onto a slider target."""
    if not isinstance(values, dict):
        return
    for _, t, field, maxv in WARP_SLIDERS + NEURAL_SLIDERS:
        v = values.get(field)
        if t == target and isinstance(v, (int, float)):
            setattr(obj, field, min(max(float(v), 0.0), maxv / 100))


def load_mode(targets, prefs, args, neural):
    """Set the warp sliders for a mode: defaults, then what was saved for it."""
    warp = targets["warp"]
    load_values(warp, "warp", slider_values(defaults(args, neural)["warp"], "warp"))
    load_values(warp, "warp", prefs.data.get("warp", {}).get("neural" if neural else "classic"))
    if args.strength is not None:
        warp.strength = args.strength


def remember(prefs, targets, gan, use_gan):
    d = prefs.data
    if not isinstance(d.get("warp"), dict):
        d["warp"] = {}
    d["warp"]["neural" if use_gan else "classic"] = slider_values(targets["warp"], "warp")
    if gan:  # only record mode and model when the neural option exists
        d["mode"] = "neural" if use_gan else "classic"
        d["gender"] = gan.gender
        d["gan"] = slider_values(gan, "gan")


def setup_window(sliders, targets, w, h):
    # Video and sliders in separate windows: 12 sliders under the video would
    # squeeze it to nothing at normal window sizes
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    pw = min(w, 960)
    cv2.resizeWindow(WINDOW, pw, pw * h // w)
    cv2.moveWindow(WINDOW, 40, 40)
    # Qt always reserves room for an image: use it for a short header, and let
    # the window size itself to the header plus the sliders
    cv2.namedWindow(CONTROLS, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    for label, target, field, maxv in sliders:
        cv2.createTrackbar(label, CONTROLS, int(round(getattr(targets[target], field) * 100)),
                           maxv, lambda _v: None)
    header = np.full((56, 520, 3), 40, np.uint8)
    cv2.putText(header, "Effect settings: changes apply live", (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "Press r in the video window to reset", (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.imshow(CONTROLS, header)
    cv2.moveWindow(CONTROLS, 40 + pw + 20, 40)


def read_sliders(sliders, targets):
    if cv2.getWindowProperty(CONTROLS, cv2.WND_PROP_VISIBLE) < 1:
        return  # controls window closed: keep the last values
    for label, target, field, _ in sliders:
        v = cv2.getTrackbarPos(label, CONTROLS)
        if v >= 0:
            setattr(targets[target], field, v / 100.0)


def set_sliders(sliders, values):
    if cv2.getWindowProperty(CONTROLS, cv2.WND_PROP_VISIBLE) < 1:
        return
    for label, target, field, _ in sliders:
        cv2.setTrackbarPos(label, CONTROLS, int(round(getattr(values[target], field) * 100)))


def draw_hud(img, fps, enabled, n_faces, virtual, gan, use_gan):
    mode = f"neural ({gan.gender})" if use_gan else "classic"
    lines = [f"{fps:4.1f} fps  faces: {n_faces}  filter: {'ON' if enabled else 'OFF'}"
             f"  mode: {mode}  virtual cam: {'on' if virtual else 'off'}",
             "space: filter on/off  " + ("n: neural/classic  g: male/female  " if gan else "")
             + "l: landmarks  r: reset sliders  q: quit"]
    # Translucent bar instead of a thick black outline: OpenCV 5 treats putText
    # thickness as font weight, so an outline pass comes out wider than the text
    bar = img[:12 + 22 * len(lines)]
    bar[:] = (bar * 0.45).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    if args.input == args.output or args.input == args.output.removeprefix("/dev/video"):
        sys.exit("error: input and output are the same device")

    cv2.setNumThreads(4)  # more threads only add spin overhead for these small ROIs
    prefs = Prefs(CONFIG_FILE, fresh=args.reset_settings)
    saved_gender = prefs.data.get("gender")
    gender = args.gender or (saved_gender if saved_gender in ("male", "female") else "male")
    gan = create_gan(args.mode, gender)
    use_gan = gan is not None and (args.mode == "neural" or prefs.data.get("mode") != "classic")
    settings = Settings()
    flt = BabyFaceFilter(settings)
    targets = {"warp": settings, "gan": gan}
    load_mode(targets, prefs, args, use_gan)
    if gan:
        load_values(gan, "gan", NEURAL_DEFAULTS)
        load_values(gan, "gan", prefs.data.get("gan"))
    if prefs.data and not args.reset_settings:
        print(f"Using saved settings from {CONFIG_FILE}")
    sliders = WARP_SLIDERS + (NEURAL_SLIDERS if gan else [])
    mp, landmarker = create_landmarker(args.model, args.faces)
    cap = open_capture(args.input, args.width, args.height, args.fps)

    ok, frame = cap.read()
    if not ok:
        sys.exit("error: camera returned no frames")
    h, w = frame.shape[:2]
    print(f"Camera {args.input}: {w}x{h}")

    cam = None if args.no_virtual else open_virtual_cam(args.output, w, h, args.fps)
    preview = not args.no_preview
    if preview:
        setup_window(sliders, targets, w, h)

    smoothers = []
    enabled, show_lm = True, False
    amount = 0.0           # fades the effect in/out when a face appears/disappears
    t0 = time.monotonic()
    last_ts = -1
    fps, fps_t, fps_n = 0.0, t0, 0
    saved_t = t0

    try:
        while True:
            if frame is None:
                ok, frame = cap.read()
                if not ok:
                    print("camera stream ended")
                    break
            now = time.monotonic()
            ts = max(last_ts + 1, int((now - t0) * 1000))
            last_ts = ts

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            faces = []
            if len(result.face_landmarks) != len(smoothers):
                smoothers = [OneEuro() for _ in result.face_landmarks]
            for lms, sm in zip(result.face_landmarks, smoothers):
                pts = np.array([(p.x * w, p.y * h) for p in lms], np.float32)
                faces.append(sm(pts, now))

            target = 1.0 if (enabled and faces) else 0.0
            amount += float(np.clip(target - amount, -0.15, 0.15))  # ~0.2 s fade
            if preview:
                read_sliders(sliders, targets)
            if faces and amount > 0:
                warped = flt.apply(frame, faces, amount)
                if use_gan:
                    gan.apply(frame, warped, amount)

            if cam is not None:
                cam.send(cv2.flip(frame, 1) if args.mirror_output else frame)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0
                if not preview:
                    print(f"\r{fps:4.1f} fps, faces: {len(faces)}  ", end="", flush=True)
            if now - saved_t >= 1.0:  # writes only when something changed
                remember(prefs, targets, gan, use_gan)
                prefs.save()
                saved_t = now

            if preview:
                view = cv2.flip(frame, 1)  # mirror like a real mirror
                if show_lm:
                    for f in faces:
                        for x, y in f:
                            cv2.circle(view, (int(w - 1 - x), int(y)), 1, (0, 255, 0), -1)
                draw_hud(view, fps, enabled, len(faces), cam is not None, gan, use_gan)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    enabled = not enabled
                elif key == ord("l"):
                    show_lm = not show_lm
                elif key == ord("r"):
                    d = defaults(args, use_gan)
                    load_values(settings, "warp", slider_values(d["warp"], "warp"))
                    if gan:
                        load_values(gan, "gan", NEURAL_DEFAULTS)
                    set_sliders(sliders, targets)
                elif key == ord("n") and gan:
                    remember(prefs, targets, gan, use_gan)  # keep this mode's sliders
                    use_gan = not use_gan
                    load_mode(targets, prefs, args, use_gan)
                    set_sliders(sliders, targets)
                elif key == ord("g") and gan:
                    gan.gender = "female" if gan.gender == "male" else "male"
                    gan.reset()
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            frame = None
    except KeyboardInterrupt:
        pass
    finally:
        remember(prefs, targets, gan, use_gan)
        prefs.save()
        cap.release()
        if cam is not None:
            cam.close()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

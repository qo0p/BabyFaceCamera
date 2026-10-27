# BabyFace Camera

Turns your webcam into a baby. It reads the real webcam, applies a live
baby-face filter, and streams the result to a virtual webcam called
**BabyFace Camera** that Zoom, Telegram, Google Meet, Discord, OBS and so on can pick
like any other camera.

There are two modes:

- **Neural (default with an NVIDIA GPU):** realistic, similar to popular "baby face" camera lenses.
  A neural network redraws your face as a 0–2 year old. It uses the generator from
  [Lifespan Age Transformation Synthesis](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis)
  (Or-El et al., ECCV 2020). A light warp first makes the head rounder and the chin
  shorter, so the baby face fits it. Runs at about 27 fps at 720p on an RTX 4070 Laptop GPU.
- **Classic (any computer):** geometry and skin only. It enlarges the eyes, shrinks the nose
  and mouth, shortens the chin, makes the face rounder, smooths the skin and adds pink cheeks.
  It runs on the CPU (about 14 ms per face).

Both modes track 478 face landmarks with MediaPipe.

> **Neural mode is for non-commercial use only.** It uses a third-party model licensed under
> CC BY-NC-SA 4.0. Classic mode doesn't use it. See [License and legal](#license-and-legal).

## Setup (once)

```bash
cd ~/Projects/Claude/BabyFaceCamera
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
./download_models.sh
```

`download_models.sh` fetches the MediaPipe models, the Lifespan code at a pinned commit, and both
neural models (2 × 213 MB, MD5-checked) into `third_party/lifespan`. On a machine without an
NVIDIA GPU, run `./download_models.sh --classic` and leave out torch.

Create the virtual camera device `/dev/video10` (needs sudo):

```bash
sudo ./setup_virtualcam.sh --persist
```

`--persist` makes it come back after a reboot. Leave it out to create the device only until the next reboot,
and use `--remove` to delete the device.

## Use

```bash
./run.sh
```

A preview window opens, and the filtered video goes to the virtual camera. Then in
**Zoom / Telegram / Meet** choose **BabyFace Camera** as your camera.

> Start BabyFace Camera **before** you open the camera settings in the other app.
> The virtual device only shows up as a camera while something is streaming to it
> (`exclusive_caps=1`, which Chrome-based apps and Zoom need).

Two windows open: **BabyFace Camera** shows the video, and **BabyFace Controls** has the sliders.
The keys below work while the video window is focused.


| Key / slider | Action |
|---|---|
| `space` | filter on/off (the stream continues, unfiltered) |
| `n` | switch between neural and classic mode |
| `g` | switch between the male and female neural model |
| `l` | show landmarks (preview only) |
| `r` | reset sliders to defaults |
| `q` / `Esc` | quit |
| sliders | strength of each part of the effect; changes are live |
| `Baby AI %` | how much of the neural face to show |
| `Age 0-2..3-6` | 0 = baby, 100 = toddler |
| `Stability` | smooths the neural face over time; higher means less flicker but more lag |

The buttons above the video only change the preview, not what apps receive. Hover over a button to see its tooltip:

| Button | Action |
|---|---|
| ← → ↑ ↓ | pan a zoomed-in preview |
| first picture icon | zoom back to 100% |
| second picture icon | zoom in to pixel level |
| magnifiers | zoom in / out (also mouse wheel) |
| floppy disk | save the current frame as an image |
| clipboard | copy the current frame |
| last icon (greyed out) | properties panel, not used |

The preview is mirrored like a real mirror. The stream to apps is not mirrored, because Zoom and
Telegram mirror your self-view themselves.

Useful options (`./run.sh --help`):

```
-i 2                  use another real camera (index or /dev/videoN)
-o /dev/video11       stream to another loopback device
--width 1920 --height 1080
--strength 1.5        stronger warp (0..2)
--mode classic        no neural network (neural, classic or auto)
--gender female       start with the female neural model
--faces 3             filter up to 3 faces
--no-preview          run without a window
--no-virtual          preview only, no virtual camera needed
--mirror-output       mirror what apps receive
```

## Troubleshooting

- **`/dev/video10 does not exist`**: run `sudo ./setup_virtualcam.sh`.
- **Camera not listed in an app**: make sure `./run.sh` is running, then restart
  the app or reopen its video settings. Snap and Flatpak apps (for example Telegram from the Snap Store) also
  need camera access: `snap connect telegram-desktop:camera`.
- **"cannot open camera 0"**: another app is using the real webcam, or it has another index.
  Run `ls /dev/video*` and pass `-i N`.
- **Neural mode says "unavailable"**: check `nvidia-smi` and that torch sees the GPU:
  `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`. Run `./download_models.sh` if models are missing.
- **Baby face looks wrong for you**: try the other model with `g`, or lower `Baby AI %`.
  The network works best looking roughly at the camera; strong head turns blend back toward your real face.
- **Low fps in a dark room**: the webcam lowers its own frame rate in low light. Turn on a light.

## Files

- `babyface_cam.py`: capture, landmark tracking (with One Euro smoothing), preview, virtual camera output
- `baby_filter.py`: the warp and skin filter (one displacement field + `cv2.remap`, guided-filter skin smoothing)
- `baby_gan.py`: the neural stage. It aligns the face to the model's 256×256 crop, greys out the background and
  clothes (MediaPipe selfie segmenter), runs the age-0–2 generator in fp16 on the GPU, matches the colour and
  feathers the result back into the frame
- `download_models.sh`: fetches all models
- `setup_virtualcam.sh`: creates or removes the v4l2loopback device

## License and legal

### This project

- All code in this repository is under the [MIT License](LICENSE), except `baby_gan.py`.
- `baby_gan.py` is under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
  because it is written to run the Lifespan model, which uses that license.

This repository contains **only original code**. It does not include or redistribute any third-party
code, model weights or datasets. `download_models.sh` downloads them from their official sources
onto your computer, and they keep their own licenses.

### Third-party components

| Component | Used for | License | Where it comes from |
|---|---|---|---|
| [Lifespan Age Transformation Synthesis](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis) code and pretrained models | neural mode | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (**non-commercial**) | GitHub, University of Washington server |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) and its Face Landmarker and Selfie Segmenter models | face tracking, segmentation | Apache 2.0 | PyPI, Google |
| [PyTorch](https://pytorch.org) | neural mode | BSD-3-Clause | PyPI |
| [OpenCV](https://opencv.org) | image processing, preview | Apache 2.0 | PyPI |
| [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) | virtual camera output | GPL-2.0 | PyPI |
| [v4l2loopback](https://github.com/umlaeute/v4l2loopback) | virtual camera device | GPL-2.0 | Ubuntu packages |

### What you may and may not do with neural mode

The Lifespan model is licensed for **non-commercial use only**. With neural mode you **may**:

- use it personally, for example in your own video calls
- share, modify and publish this project for free

You **may not**, without permission from the Lifespan authors (University of Washington):

- sell this software, or a product or service that includes neural mode
- offer it behind a paywall, subscription or paid service
- use it in any other commercial way

If you share a modified version of the neural part, you must credit the authors and use the
same CC BY-NC-SA 4.0 license.

Classic mode doesn't use the Lifespan model, so it isn't restricted this way. It still uses the
other components in the table above under their licenses.

### Distribution note

Distribute this project **as source code only**, with users downloading the models themselves (as
`download_models.sh` does). Don't publish a bundle, such as a Docker image or a packaged app, that
includes both pyvirtualcam (GPL-2.0) and the Lifespan model (CC BY-NC-SA 4.0). These two licenses
are not compatible with each other when they are distributed together.

### Credits

Neural mode uses the work of Roy Or-El, Soumyadip Sengupta, Ohad Fried, Eli Shechtman and
Ira Kemelmacher-Shlizerman. If you use it in research, please cite:

```bibtex
@inproceedings{orel2020lifespan,
  title={Lifespan Age Transformation Synthesis},
  author={Or-El, Roy and Sengupta, Soumyadip and Fried, Ohad and Shechtman, Eli and Kemelmacher-Shlizerman, Ira},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

### Responsible use

- This filter changes how a face looks in live video. Use it for fun and with your own face.
  Don't use it to deceive people, impersonate someone, or get around identity checks.
- Get consent before filtering or recording other people, and follow the rules of the apps and
  meetings you use it in.
- Only publish screenshots or demo videos of yourself or of people who have agreed to it.

### Trademarks and disclaimer

This is an independent project. It is not affiliated with, endorsed by or sponsored by Snap Inc.,
Zoom, Telegram, Google, the University of Washington or the Lifespan authors. Product names are
trademarks of their owners and are used only to describe compatibility.

The software is provided "as is", without warranty of any kind (see [LICENSE](LICENSE)). This
section summarizes the licenses for convenience. It is not legal advice, and the license texts
linked above are what apply.

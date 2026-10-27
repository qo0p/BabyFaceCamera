#!/usr/bin/env bash
# Download every model BabyFace Camera uses. Safe to re-run: skips what exists.
#   ./download_models.sh            MediaPipe models + neural baby-face model
#   ./download_models.sh --classic  MediaPipe models only (no GPU needed)
set -euo pipefail
cd "$(dirname "$0")"

MP=https://storage.googleapis.com/mediapipe-models
mkdir -p models
get() {  # url dest
    [ -s "$2" ] || { echo "Downloading $(basename "$2")"; curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"; }
}
get "$MP/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" models/face_landmarker.task
get "$MP/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite" models/selfie_segmenter.tflite

[ "${1:-}" = "--classic" ] && exit 0

# Lifespan Age Transformation Synthesis (CC BY-NC-SA 4.0, non-commercial use only)
LS=third_party/lifespan
COMMIT=29de7bd8e35c33000654af236bdb15269877790c
if [ ! -d "$LS/.git" ]; then
    git clone -q https://github.com/royorel/Lifespan_Age_Transformation_Synthesis.git "$LS"
fi
git -C "$LS" -c advice.detachedHead=false checkout -q "$COMMIT"

URL=https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/pretrained_models
declare -A MD5=([males]=0079186147ec816176b946a073d1f396 [females]=0675f809413c026170cf1f22b27f3c5d)
for m in males females; do
    [ -s "$LS/checkpoints/${m}_model/latest_net_g_running.pth" ] && continue
    zip="$LS/checkpoints/${m}_model.zip"
    mkdir -p "$LS/checkpoints"
    echo "Downloading ${m} model (213 MB)"
    curl -fL --progress-bar -o "$zip" "$URL/${m}_model.zip"
    echo "${MD5[$m]}  $zip" | md5sum -c --quiet
    unzip -q -o "$zip" -d "$LS/checkpoints" && rm "$zip"
done
echo "All models ready."

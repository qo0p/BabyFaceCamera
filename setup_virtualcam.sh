#!/usr/bin/env bash
# Create the "BabyFace Camera" virtual webcam (/dev/video10) with v4l2loopback.
#   sudo ./setup_virtualcam.sh            load now (until reboot)
#   sudo ./setup_virtualcam.sh --persist  also load automatically at every boot
#   sudo ./setup_virtualcam.sh --remove   unload and remove the boot config
set -euo pipefail

NR="${VIDEO_NR:-10}"
LABEL="BabyFace Camera"
OPTS=(devices=1 "video_nr=$NR" "card_label=$LABEL" exclusive_caps=1 max_buffers=2)

[ "$EUID" -eq 0 ] || exec sudo "$0" "$@"

if [ "${1:-}" = "--remove" ]; then
    rm -f /etc/modules-load.d/babyface-cam.conf /etc/modprobe.d/babyface-cam.conf
    modprobe -r v4l2loopback && echo "Virtual camera removed."
    exit 0
fi

if ! modinfo v4l2loopback >/dev/null 2>&1; then
    echo "Installing v4l2loopback..."
    apt-get install -y v4l2loopback-dkms
fi

if [ -e "/dev/video$NR" ]; then
    echo "/dev/video$NR already exists: $(cat /sys/class/video4linux/video$NR/name 2>/dev/null)"
elif lsmod | grep -q '^v4l2loopback'; then
    echo "v4l2loopback is already loaded with other devices (OBS?)."
    echo "Unload it first (sudo modprobe -r v4l2loopback) or start BabyFace with -o /dev/videoN."
    exit 1
else
    modprobe v4l2loopback "${OPTS[@]}"
    echo "Created /dev/video$NR \"$LABEL\"."
fi

if [ "${1:-}" = "--persist" ]; then
    echo v4l2loopback > /etc/modules-load.d/babyface-cam.conf
    echo "options v4l2loopback devices=1 video_nr=$NR card_label=\"$LABEL\" exclusive_caps=1 max_buffers=2" \
        > /etc/modprobe.d/babyface-cam.conf
    echo "It will be created automatically at boot."
fi

#!/usr/bin/env bash
# Start BabyFace Camera; all arguments are passed through (see --help).
cd "$(dirname "$0")"
exec .venv/bin/python babyface_cam.py "$@"

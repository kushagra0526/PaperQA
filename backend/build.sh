#!/usr/bin/env bash
set -e

echo "==> Installing requirements"
# --extra-index-url is declared inside requirements.txt as a pip options line
# (the line starting with "--extra-index-url https://download.pytorch.org/whl/cpu").
# pip reads and honours it automatically when processing the file, so it does
# not need to be repeated here on the command line.
pip install --no-cache-dir -r requirements.txt

echo "==> Pre-caching model weights"
python preload.py

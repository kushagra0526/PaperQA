#!/usr/bin/env bash
set -e

echo "==> Installing requirements"
pip install --no-cache-dir -r requirements.txt

echo "==> Pre-caching model weights"
python preload.py

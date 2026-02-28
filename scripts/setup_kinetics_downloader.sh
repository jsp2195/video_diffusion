#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH=${1:-/mnt/data/kinetics-dataset-main.zip}
DEST_DIR=${2:-vendor/kinetics_downloader}

mkdir -p "${DEST_DIR}"
if [ ! -f "${ZIP_PATH}" ]; then
  echo "Downloader zip not found at ${ZIP_PATH}. Skipping unzip."
  exit 0
fi

unzip -o "${ZIP_PATH}" -d "${DEST_DIR}"

echo "Unzipped to ${DEST_DIR}"

echo "Available downloader scripts:"
find "${DEST_DIR}" -maxdepth 3 -type f \( -name '*.sh' -o -name 'k400_*' \) -print

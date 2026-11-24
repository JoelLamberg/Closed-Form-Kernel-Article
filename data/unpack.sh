#!/usr/bin/env bash
# Unzip every raw-results archive into an unz/ folder next to it.
set -e
cd "$(dirname "$0")"
for z in map/results_301x301.zip corner/results_corner.zip backface/results_backface.zip \
         focus/results_focus.zip stronger_H/results_stronger_H.zip; do
  d="$(dirname "$z")/unz"; mkdir -p "$d"; unzip -q -o "$z" -d "$d"; echo "unpacked $z -> $d"
done

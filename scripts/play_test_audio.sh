#!/usr/bin/env bash
# Feeds an mp3 into the "narrator_mic_sink" PipeWire virtual mic so narrator
# picks it up as microphone input. Starts the virtual sink if it isn't
# already running.
#
# Usage: scripts/play_test_audio.sh [path/to/file.mp3]
#   (defaults to test.mp3 at the repo root)
#
# Run narrator against it with:
#   PIPEWIRE_NODE=narrator_mic_sink narrator
# and set `audio.device: pipewire` in the narrator config.
set -euo pipefail

SINK_NAME="narrator_mic_sink"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP3_PATH="${1:-$SCRIPT_DIR/../test.mp3}"

if [[ ! -f "$MP3_PATH" ]]; then
    echo "No such file: $MP3_PATH" >&2
    exit 1
fi

if ! pw-cli ls Node 2>/dev/null | grep -q "node.name = \"$SINK_NAME\""; then
    echo "Starting virtual mic sink ($SINK_NAME)..."
    pw-loopback -n "Narrator Mic Sink" \
        --capture-props="media.class=Audio/Sink node.name=$SINK_NAME node.description=\"Narrator Mic Sink\"" &
    sleep 1
fi

echo "Playing $MP3_PATH into $SINK_NAME..."
pw-play --target="$SINK_NAME" "$MP3_PATH"

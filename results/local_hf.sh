#!/usr/bin/env sh
set -eu
# Explicit existing local checkpoints: no network downloads or paid requests.
: "${TARGET_CHECKPOINT:?Set a local target checkpoint directory}"
: "${DRAFT_CHECKPOINT:?Set a local draft checkpoint with identical tokenizer}"
python -m speculative_decoding hf --target "$TARGET_CHECKPOINT" --draft "$DRAFT_CHECKPOINT" --prompt 'A small experiment:' --max-new-tokens 32 --output artifacts/local-hf.json

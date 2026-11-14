#!/usr/bin/env bash
# Runs render_onnx.py in the vendored booster_mjlab env (the one students
# trained with) plus onnxruntime. All args pass through.
#
# The course folder is found automatically; set COURSE_DIR to override.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_course() {
  if [ -n "${COURSE_DIR:-}" ]; then echo "$COURSE_DIR"; return; fi
  local d="$HERE"
  while [ "$d" != "/" ]; do
    for c in "$d" "$d/course" "$d/cs498-booster-track/course"; do
      if [ -f "$c/hw3/vendor/booster_mjlab/pyproject.toml" ]; then echo "$c"; return; fi
    done
    d="$(dirname "$d")"
  done
}
export COURSE_DIR="$(find_course)"
VENDOR="$COURSE_DIR/hw3/vendor/booster_mjlab"
if [ -z "$COURSE_DIR" ] || [ ! -f "$VENDOR/pyproject.toml" ]; then
  echo "Can't find $VENDOR. Set COURSE_DIR=/path/to/cs498-booster-track/course" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
exec uv run --project "$VENDOR" --locked --no-dev --with onnxruntime \
  python "$HERE/render_onnx.py" "$@"

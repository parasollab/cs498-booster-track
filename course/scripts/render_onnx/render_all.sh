#!/usr/bin/env bash
# Render every .onnx under a folder on Delta CPU nodes, one Slurm task per file.
#
#   ./render_all.sh ONNX_FOLDER OUT_DIR [SLURM_ACCOUNT]
#
# Video names: an .onnx inside a subfolder is named after its top-level
# subfolder (e.g. student1/checkpoints/x.onnx -> student1.mp4); a top-level .onnx keeps
# its filename minus "-1" / " (2)" suffixes. Byte-identical duplicates are
# rendered once. Each task writes NAME.mp4 and NAME.webm (plays in VS Code);
# when all tasks finish, OUT_DIR/summary.csv is assembled.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="$(cd "${1:?usage: render_all.sh ONNX_FOLDER OUT_DIR [ACCOUNT]}" && pwd)"
mkdir -p "${2:?usage: render_all.sh ONNX_FOLDER OUT_DIR [ACCOUNT]}"
OUT="$(cd "$2" && pwd)"
ACCOUNT="${3:-bign-delta-cpu}"
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

# Build the per-task list (onnx path <TAB> name), skipping macOS junk and duplicates.
WORK="$OUT/.render_jobs"; mkdir -p "$WORK/logs" "$WORK/summaries"
LIST="$WORK/list.tsv"; : > "$LIST"
declare -A seen_hash seen_name
while IFS= read -r -d '' f; do
  h=$(md5sum "$f" | cut -d' ' -f1)
  [ -n "${seen_hash[$h]:-}" ] && { echo "skip duplicate: ${f#$IN/} (same as ${seen_hash[$h]})"; continue; }
  rel="${f#$IN/}"
  if [[ "$rel" == */* ]]; then name="${rel%%/*}"; else
    name=$(basename "$f" .onnx | sed -E 's/ \([0-9]+\)$//; s/-[0-9]+$//; s/ /_/g'); fi
  if [ -n "${seen_name[$name]:-}" ]; then name="${name}_$(basename "$f" .onnx | tr ' ' _)"; fi
  seen_hash[$h]="$rel"; seen_name[$name]=1
  printf "%s\t%s\n" "$f" "$name" >> "$LIST"
done < <(find "$IN" -name "*.onnx" -not -path "*__MACOSX*" -not -name "._*" -print0 | sort -z)

N=$(wc -l < "$LIST")
[ "$N" -gt 0 ] || { echo "No .onnx files under $IN" >&2; exit 1; }
echo "Rendering $N policies:"; cut -f2 "$LIST" | sed 's/^/  /'

JOB=$(sbatch --parsable --account="$ACCOUNT" --partition=cpu --cpus-per-task=16 --mem=32G \
  --time=01:00:00 --job-name=onnx-render --array=0-$((N - 1)) \
  --output="$WORK/logs/%A_%a.out" --export=ALL,HERE="$HERE",LIST="$LIST",OUT="$OUT",WORK="$WORK" \
  --wrap='set -euo pipefail
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
ONNX=$(cut -f1 <<<"$LINE"); NAME=$(cut -f2 <<<"$LINE")
TMP=$(mktemp -d); ln -s "$ONNX" "$TMP/$NAME.onnx"
"$HERE/render_onnx.sh" "$TMP/$NAME.onnx" --out-dir "$OUT" --summary "$WORK/summaries/$NAME.csv"
FF=$(ls "$COURSE_DIR"/hw3/vendor/booster_mjlab/.venv/lib/python3.*/site-packages/imageio_ffmpeg/binaries/ffmpeg-* | head -1)
"$FF" -y -loglevel error -i "$OUT/$NAME.mp4" -c:v libvpx-vp9 -b:v 0 -crf 32 -row-mt 1 "$OUT/$NAME.webm"
echo "DONE $NAME"')

# Merge per-task summaries once every task has finished (success or not).
sbatch --parsable --account="$ACCOUNT" --partition=cpu --cpus-per-task=1 --time=00:05:00 \
  --job-name=onnx-summary --dependency=afterany:"$JOB" --output="$WORK/logs/summary.out" \
  --wrap="cd '$WORK/summaries' && f=\$(ls *.csv | head -1) && { head -1 \"\$f\"; for c in *.csv; do tail -n +2 \"\$c\"; done; } > '$OUT/summary.csv'" >/dev/null

echo
echo "Submitted job $JOB. Watch with: squeue -u \$USER"
echo "Videos and summary.csv will appear in $OUT (logs in $WORK/logs)."

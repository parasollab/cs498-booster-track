# HW3 ONNX Renderer

Renders each student's `.onnx` policy in simulation and saves a video.

Each video runs: stand 2 s, walk forward 5 s, turn 5 s, stand 3 s.

## Setup

- Run on Delta from a clone of `cs498-booster-track`.
- Have `uv` installed.

## Usage

Render a folder of `.onnx` files (submits Slurm jobs):

```bash
./render_all.sh <onnx_folder> <output_folder> [slurm_account]
```

Render one or two files directly:

```bash
./render_onnx.sh student1.onnx --out-dir videos
```

## Output

- `NAME.mp4` and `NAME.webm` for each student (`.webm` plays in VS Code)
- `summary.csv` with commanded vs. measured speed and number of falls

"""Test the actual course templates using fake commands, without submitting jobs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory(prefix="booster-batch-") as tmp:
        root = Path(tmp).resolve()
        (root / "scripts").mkdir()
        (root / "src/course_tasks").mkdir(parents=True)
        (root / "pyproject.toml").touch()
        (root / "bin").mkdir()
        capture = root / "capture.json"
        (root / "bin/uv").write_text("#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ['BATCH_CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd(),'wandb':os.environ.get('WANDB_DIR')}))\n")
        (root / "bin/nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n")
        for p in (root / "bin").iterdir():
            p.chmod(0o755)
        env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
            SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID="mock", BATCH_CAPTURE=str(capture))
        for template in ("train_delta.sbatch", "train_deltaai.sbatch"):
            script = PROJECT / "scripts" / template
            subprocess.run(["bash", "-n", str(script)], check=True)
            original = subprocess.run(["git", "-C", str(PROJECT.parent), "show",
                "HEAD:course/scripts/" + template], check=True, capture_output=True, text=True).stdout
            assert [line for line in script.read_text().splitlines() if line.startswith("#SBATCH")] == [
                line for line in original.splitlines() if line.startswith("#SBATCH")], "Cluster resources changed"
            # Slurm executes a spooled copy outside the checkout.
            spooled = root / "bin" / template
            spooled.write_text(script.read_text())
            script = spooled
            for task in ("Mjlab-Velocity-Flat-Booster-K1", "Course-Booster-K1-Rewards",
                         "Course-Booster-K1-Rewards-Candidate", "Course-Cartpole-Balance",
                         "Course-HW2-DoubleCartpole-Balance-PPO"):
                for override in ([], ["--log-root", "custom root"], ["--log-root=custom root"]):
                    extras = ["--env.scene.num-envs", "32", "--agent.run-name", "space name", *override]
                    subprocess.run(["bash", str(script), task, *extras], cwd=root, env=env,
                        check=True, capture_output=True, text=True)
                    got = json.loads(capture.read_text())
                    prefix = (["run", "train"] if task in {"Course-Cartpole-Balance", "Course-HW2-DoubleCartpole-Balance-PPO"} else
                        ["run", "--project", "hw3/vendor/booster_mjlab", "--locked", "--no-dev",
                         "python", "hw3/run.py", "train"])
                    logargs = [] if override else ["--log-root", str(root / "logs/rsl_rl")]
                    assert got["argv"] == [*prefix, task, *logargs, *extras], got
                    assert got["cwd"] == str(root) and got["wandb"] == str(root / "logs")
            invalid = dict(env, SLURM_SUBMIT_DIR=str(root / "bin"))
            failed = subprocess.run(["bash", str(script), "Course-Booster-K1-Rewards"],
                env=invalid, capture_output=True, text=True)
            assert failed.returncode != 0 and "submit from the repo root" in failed.stderr
            missing = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            assert missing.returncode != 0 and "Usage:" in missing.stderr
    print("PASS: both templates, all routes, quoting, log-root forms, cwd/W&B, invalid cwd/missing task")


if __name__ == "__main__":
    main()

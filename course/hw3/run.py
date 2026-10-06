"""Run from the course Python-project root with the isolated vendored project."""
import sys

SUPPORTED_TASKS = (
    "Mjlab-Velocity-Flat-Booster-K1",
    "Course-Booster-K1-Rewards",
    "Course-Booster-K1-Rewards-Candidate",
)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "play", "list", "check"}:
        raise SystemExit("Usage: hw3/run.py {train|play|list|check} [task and CLI args]")
    command = sys.argv.pop(1)
    if command in {"train", "play"} and (len(sys.argv) < 2 or sys.argv[1] not in SUPPORTED_TASKS):
        raise SystemExit("Choose a supported Booster task: " + ", ".join(SUPPORTED_TASKS))
    import course_booster  # noqa: F401
    if command == "train":
        from booster_mjlab.scripts.train import main as entry
    elif command == "play":
        from booster_mjlab.scripts.play import main as entry
    elif command == "list":
        from mjlab.tasks.registry import list_tasks
        for task in SUPPORTED_TASKS:
            assert task in list_tasks(), task
            print(task)
        return
    else:
        from course_booster.check import main as entry
    entry()


if __name__ == "__main__":
    main()

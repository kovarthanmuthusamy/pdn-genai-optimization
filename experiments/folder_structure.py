import sys
from pathlib import Path
import subprocess

def ask(prompt):
    return input(f"{prompt}\n> ").strip()

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_experiment.py <experiment_name>")
        sys.exit(1)

    exp_name = sys.argv[1]
    base = Path("experiments") / exp_name

    # Directory structure
    dirs = [
        base / "checkpoints",
        base / "metrics",
        base / "visuals"
    ]

    files = [
        base / "config.yaml",
        base / "git_commit.txt",
        base / "metrics" / "loss.csv",
        base / "metrics" / "summary.json"
    ]

    # Create directories
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Create files
    for f in files:
        f.touch(exist_ok=True)

    # Interactive notes
    notes_path = base / "notes.md"
    if not notes_path.exists() or notes_path.stat().st_size == 0:
        print("\n--- Experiment Notes (interactive) ---")

        goal = ask("Goal of this experiment?")
        changes = ask("What changed compared to previous experiment?")
        results = ask("Key results / observations?")
        decision = ask("Decision (keep / discard / modify)?")

        notes_text = f"""# Experiment: {exp_name}

## Goal
{goal}

## Changes
{changes}

## Results
{results}

## Decision
{decision}
"""
        notes_path.write_text(notes_text)

    print(f"\nExperiment structure created at: {base}")

if __name__ == "__main__":
    main()

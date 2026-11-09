"""Rerun Part 2 from a completed Part 1 database export."""
import argparse
import json
from week11.pipeline import OUT, run_agents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Completed Part 1 artifact directory name; defaults to latest")
    args = parser.parse_args()
    run_id = args.run_id or json.loads((OUT / "latest.json").read_text())["run_id"]
    if not run_id or any(c not in "0123456789T" for c in run_id):
        parser.error("Expected a timestamp run ID such as 20261109T184725")
    analysis = json.loads((OUT / run_id / "analysis.json").read_text())
    print(json.dumps(run_agents(analysis), indent=2))


if __name__ == "__main__":
    main()

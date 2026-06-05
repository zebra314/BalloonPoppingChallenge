"""Launch evaluation for the RLAgent.

Run from the repository root:

    python doc/examples/evaluate_rl_agent.py
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from BalloonPoppingGymEnv.agents.rl_agent import RLAgent
from BalloonPoppingGymEnv.evaluation.evaluate import evaluate_scenario


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RLAgent on a scenario.")
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
        help="Scenario number to evaluate.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("matplotlib", "vpython", "none"),
        default="matplotlib",
        help="Renderer to use. Use 'none' for headless evaluation.",
    )
    parser.add_argument(
        "--launch-time",
        type=float,
        default=1.0,
        help="Simulation time, in seconds, when the agent should launch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode

    evaluate_scenario(
        RLAgent,
        agent_kwargs={"launch_time": args.launch_time},
        agent_name="RL Agent",
        scenario_number=args.scenario,
        render_mode=render_mode,
    )


if __name__ == "__main__":
    main()

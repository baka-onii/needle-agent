"""Public streaming API example: uv run python examples/basic.py --demo."""

import argparse
import os
from pathlib import Path

from agent_runtime import Agent, AgentConfig
from agent_runtime.models.demo import DemoActionModel, DemoReasoningModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--workspace", default=str(Path(__file__).parent / "workspace"))
    parser.add_argument("request", nargs="?", default="Find the authentication implementation.")
    args = parser.parse_args()
    config = AgentConfig(
        workspace_root=args.workspace,
        read_only=True,
        llm_base_url=os.getenv("NEEDLE_LLM_BASE_URL", "http://127.0.0.1:8080"),
        llm_model=os.getenv("NEEDLE_LLM_MODEL", "ornith"),
        llm_api_key=os.getenv("NEEDLE_LLM_API_KEY"),
    )
    models = {"reasoning": DemoReasoningModel(), "action": DemoActionModel()} if args.demo else {}
    with Agent(config, **models) as agent:
        for event in agent.stream(args.request):
            if event["type"] == "action":
                print(f"→ {event['action']}")
            elif event["type"] == "complete":
                state = event["state"]
                print(f"\n[{state['status']}; {state['step_count']} tool steps]")
                print(state["final_answer"])


if __name__ == "__main__":
    main()

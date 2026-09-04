"""Minimal live example: reasoning via llama.cpp server, actions via Needle.

Prerequisites:
  1. Serve a model (see README for the recommended llama-server flags).
  2. Run from the repo root:  .venv\\Scripts\\python.exe examples\\basic.py
"""

from agent_runtime.agent import Agent
from agent_runtime.config import AgentConfig


def main() -> None:
    config = AgentConfig(workspace_root=".")
    agent = Agent(config=config)
    state = agent.run("List the top-level directory, then summarize what this project is.")
    print(f"[{state['status']}] after {state['step_count']} tool steps:")
    print(state["final_answer"])


if __name__ == "__main__":
    main()

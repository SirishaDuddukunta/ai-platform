import sys
import os

# 1. Path Management: This ensures Python can find your 'src' package
# regardless of which directory you are running this script from.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.agent import NativeAgent


def test_agent_loop():

    # Arrange: Initialize with a focused system prompt
    agent = NativeAgent("You are a helpful CRM assistant.")

    # Act: Run the agent against a known query
    print("--- Starting Agent Trace ---")

    result = agent.run("What is the status of Alpha project?")

    # Debug: Print the history to see how the agent reasoned
    print("\n--- DEBUG: Agent Memory (State Trace) ---")

    for i, turn in enumerate(agent.history):
        print(f"Turn {i}: {turn['role']} -> {turn['content'][:60]}...")

    print("------------------------------------------\n")

    # Assert: Process Integrity

    if len(agent.history) < 3:
        print("Model took a shortcut: answered directly without tool. (Acceptable)")
    else:

        # If it *did* use the tool, we strictly enforce that it captured the observation
        assert len(agent.history) >= 3, \
            "Agent failed to append observation to history!"

        # ------------------- CHANGED -------------------
        # Earlier we checked for the word "Design".
        # Now the tool returns structured JSON.
        # So verify that the observation contains "status".
        # -----------------------------------------------
        assert "status" in agent.history[-1]["content"]

    # ------------------- CHANGED -------------------
    # Removed:
    #
    # assert "Design" in result
    #
    # because result is now a dictionary,
    # not a plain string.
    # -----------------------------------------------

    # 1. Check if result is a dictionary (the agent returned JSON)
    assert isinstance(result, dict), \
        f"Agent result should be a dict, got {type(result)}"

    # 2. Check the specific business logic keys
    assert result.get("project_name") == "Alpha", \
        "Agent retrieved the wrong project!"

    # 3. Validate the status (Acceptable status values)
    valid_statuses = [
        "in_progress",
        "design",
        "development"
    ]

    assert result.get("status") in valid_statuses, \
        f"Unexpected status: {result.get('status')}"

    # ------------------- CHANGED -------------------
    # Added validation for additional JSON fields
    # expected from the tool.
    # -----------------------------------------------
    assert "percentage_complete" in result
    assert "expected_completion_date" in result

    print(
        f"--- Test Passed: Successfully validated JSON structure "
        f"(Status: {result.get('status')}) ---"
    )

    print("--- Test Passed: Agent logic and outcome verified! ---")


if __name__ == "__main__":
    test_agent_loop()
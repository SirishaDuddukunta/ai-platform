import sys
import os

# Path Management: Ensures Python finds the 'src' package
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

    # Assert: Retrieval Integrity
    # If the history is >= 3, it means it used the tool. We validate the content.
    if len(agent.history) >= 3:
        last_obs = agent.history[-1]["content"]
        assert "Alpha" in last_obs, f"Failed to retrieve Alpha project data. Got: {last_obs}"
        assert "Design" in last_obs, f"Failed to retrieve correct phase. Got: {last_obs}"
        print("Retrieval Validation: Passed (Data retrieved correctly)")
    else:
        print("Model took a shortcut: answered directly without tool. (Acceptable)")

    # Assert: Structural JSON Validation
    assert isinstance(result, dict), f"Agent result should be a dict, got {type(result)}"

    # Check the specific business logic keys
    assert result.get("project_name") == "Alpha", "Agent retrieved the wrong project!"

    # Validate the status
    valid_statuses = ["in_progress", "design", "development"]
    assert result.get("status") in valid_statuses, f"Unexpected status: {result.get('status')}"

    # Validate additional schema fields
    assert "percentage_complete" in result, "Missing key: percentage_complete"
    assert "expected_completion_date" in result, "Missing key: expected_completion_date"

    print(
        f"--- Test Passed: Successfully validated JSON structure "
        f"(Status: {result.get('status')}) ---"
    )


if __name__ == "__main__":
    test_agent_loop()
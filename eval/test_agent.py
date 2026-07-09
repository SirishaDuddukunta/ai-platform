import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.agent import NativeAgent # Now this will work!
from src.core.agent import NativeAgent

def test_agent_loop():
    # Arrange: Initialize with a simple prompt
    agent = NativeAgent("You are a helpful CRM assistant.")
    
    # Act: Run the agent
    print("--- Starting Agent Trace ---")
    result = agent.run("What is the status of Alpha project?")
    
    # Assert: Verify state transitions
    # 1. Did the history actually grow? (It should have at least 3 turns: System, User, Observation)
    assert len(agent.history) >= 3, "Agent failed to append observation to history!"
    
    # 2. Did the agent actually store the result?
    last_observation = agent.history[-1]['content']
    assert "Design" in last_observation, "Agent did not correctly process tool result!"
    
    print("--- Test Passed: Traceability and State Consistency Verified ---")

if __name__ == "__main__":
    test_agent_loop()
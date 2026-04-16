import json
import random

# ==========================================
# DAY 8: STABLE DEVOPS TOOLS
# ==========================================
# What: Functions to check status and restart servers.
# Why: To give the AI "hands" to interact with infrastructure.

def get_server_status(hostname: str):
    """Checks the current health and latency of a specific production server."""
    statuses = {
        "prod-db-01": "🟢 Online - Latency 12ms",
        "web-lb-02": "🔴 Offline - Connection Timeout",
        "ci-runner-linux": "🟡 Maintenance Mode"
    }
    return statuses.get(hostname, "❓ Server unknown or not in inventory.")

def restart_server(hostname: str):
    """Simulates a server restart command."""
    return f"🚀 Restart command sent to {hostname}. Estimated downtime: 2 mins."


# ==========================================
# DAY 9: ERROR HANDLING & FLAKY TOOLS
# ==========================================
# What: A tool designed to fail periodically.
# Why: To test the 'Self-Correction' logic in our main loop.

def query_database(query_string: str):
    """Queries the internal database for specific records. High chance of timeout."""
    # Simulate a 50% failure rate for testing
    if random.random() < 0.5:
        raise Exception("Timeout: Database is under heavy load. Try again in 5 seconds.")
    
    return f"Success: Found 3 records matching '{query_string}'."


# ==========================================
# CUMULATIVE TOOL SCHEMA (DAY 8 & 9)
# ==========================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_server_status",
            "description": "Checks health of a server.",
            "parameters": {
                "type": "object",
                "properties": {"hostname": {"type": "string"}},
                "required": ["hostname"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restart_server",
            "description": "Restarts a production server safely.",
            "parameters": {
                "type": "object",
                "properties": {"hostname": {"type": "string"}},
                "required": ["hostname"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Queries the internal database for specific records.",
            "parameters": {
                "type": "object",
                "properties": {"query_string": {"type": "string"}},
                "required": ["query_string"]
            }
        }
    }
]
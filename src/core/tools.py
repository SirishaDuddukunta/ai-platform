# ==========================================
# DAY 8: TOOL DEFINITIONS
# ==========================================
# DEFINITION: 
# What: This file contains standard Python functions and their JSON "descriptions."
# Why: LLMs don't see your code; they only see the JSON description. If the 
# description is clear, the LLM knows when to "call" this function.

import json

def get_server_status(hostname: str):
    """
    A mock DevOps tool to check server health.
    In production, this would be an API call or SSH command.
    """
    statuses = {
        "prod-db-01": "🟢 Online - Latency 12ms",
        "web-lb-02": "🔴 Offline - Connection Timeout",
        "ci-runner-linux": "🟡 Maintenance Mode"
    }
    return statuses.get(hostname, "❓ Server unknown or not in inventory.")

def restart_server(hostname: str):
    """
    A mock DevOps tool to restart a server.
    In production, this would trigger automation tooling.
    """
    known_servers = {"prod-db-01", "web-lb-02", "ci-runner-linux"}
    if hostname not in known_servers:
        return "❓ Restart failed: server unknown or not in inventory."
    return f"🔁 Restart command sent to {hostname}. Monitoring health checks now."

# This is the "Schema" that tells the LLM what the function does
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_server_status",
            "description": "Checks the current health and latency of a specific production server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string", 
                        "description": "The name of the server to check (e.g., prod-db-01)"
                    }
                },
                "required": ["hostname"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restart_server",
            "description": "Restarts a specific server and reports restart initiation status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string",
                        "description": "The name of the server to restart (e.g., web-lb-02)"
                    }
                },
                "required": ["hostname"]
            }
        }
    }
]
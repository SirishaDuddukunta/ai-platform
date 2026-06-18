import json
import random
import time
from datetime import datetime

# ==========================================
# DAY 11: SECURITY GUARDRAILS (UTILITY)
# ==========================================
# DEFINITION: A pre-processing gate to identify malicious intent.
def security_scanner(user_input: str):
    """Scans input for dangerous patterns like system command injection."""
    prohibited_keywords = ["rm -rf", "DROP TABLE", "format C:", "sudo", "shutdown"]
    
    for word in prohibited_keywords:
        if word.lower() in user_input.lower():
            return False, f"⚠️ Security Alert: Prohibited command pattern '{word}' detected."
    return True, "Passed"

# ==========================================
# DAY 10: OBSERVABILITY WRAPPER
# ==========================================
# DEFINITION: A simple logger to track tool performance.
# Why: To monitor model drift and latency in production-grade systems.
def log_tool_usage(f_name, status, duration, error=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": f_name,
        "status": status,
        "duration_ms": round(duration * 1000, 2),
        "error": error
    }
    print(f"📊 [OBSERVABILITY] {json.dumps(log_entry)}")

# ==========================================
# DAY 8: STABLE DEVOPS TOOLS
# ==========================================
# What: Functions to check status and restart servers.
# Why: To give the AI "hands" to interact with infrastructure.

def get_server_status(hostname: str):
    """Checks the current health and latency of a specific production server."""
    # Day 10 Observability Integration
    start_time = time.time()
    statuses = {
        "prod-db-01": "🟢 Online - Latency 12ms",
        "web-lb-02": "🔴 Offline - Connection Timeout",
        "ci-runner-linux": "🟡 Maintenance Mode"
    }
    result = statuses.get(hostname, "❓ Server unknown or not in inventory.")
    
    # Day 10 Observability Integration
    log_tool_usage("get_server_status", "success", time.time() - start_time)
    return result

def restart_server(hostname: str):
    """Simulates a server restart command."""
    # Day 10 Observability Integration
    return f"🚀 Restart command sent to {hostname}. Estimated downtime: 2 mins."

# ==========================================
# DAY 9: ERROR HANDLING & FLAKY TOOLS
# ==========================================
# What: A tool designed to fail periodically.
# Why: To test the 'Self-Correction' logic in our main loop.

def query_database(query_string: str):
    """Queries the internal database for specific records. High chance of timeout."""
    # Day 10 Observability Integration
    start_time = time.time()
    try:
        # Simulate a 50% failure rate for testing Day 9 Resilience
        if random.random() < 0.5:
            raise Exception("Timeout: Database is under heavy load. Try again in 5 seconds.")
        
        result = f"Success: Found 3 records matching '{query_string}'."
        
        # Day 10 Observability Integration
        log_tool_usage("query_database", "success", time.time() - start_time)
        return result
    except Exception as e:
        # Day 10 Observability Integration
        log_tool_usage("query_database", "failed", time.time() - start_time, error=str(e))
        raise e

# ==========================================
# CUMULATIVE TOOL SCHEMA (DAY 8, 9, 10, 11)
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
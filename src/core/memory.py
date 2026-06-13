import json
import os

MEMORY_FILE = "data/history.json"

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

def load_history():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
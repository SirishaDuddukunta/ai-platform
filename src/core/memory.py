import json
import os

# Create a folder specifically for user histories
USER_DATA_DIR = "data/users"
os.makedirs(USER_DATA_DIR, exist_ok=True)

def get_user_file(user_id: str):
    return os.path.join(USER_DATA_DIR, f"{user_id}.json")

def load_history(user_id: str):
    file_path = get_user_file(user_id)
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def save_history(user_id: str, history: list):
    file_path = get_user_file(user_id)
    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)
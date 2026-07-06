import json
import requests 
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# 1. Load the Golden Dataset
def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# 2. Real Agent Call (Integration with your FastAPI app)
def get_agent_response(question):
    """
    Sends the question to your live FastAPI agent (http://127.0.0.1:8000/chat)
    """
    url = "http://127.0.0.1:8000/chat"  # MUST match the @app.post decorator in your main.py
    payload = {"question": question}
    
    try:
        # Sending the request to your live server
        response = requests.post(url, json=payload)
        
        # Check if the server responded correctly
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error ({response.status_code}): {response.text}")
            return {"answer": "API Error", "contexts": []}
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is your server running?")
        return {"answer": "Connection Error", "contexts": []}

# 3. Main Eval Loop
def run_evaluation():
    # Ensure the path matches your folder structure
    data = load_data('eval/eval_data.json') 
    eval_rows = []

    print("--- Starting Evaluation ---")
    for item in data:
        response = get_agent_response(item['question'])
        eval_rows.append({
            "question": item['question'],
            "answer": response.get('answer', 'N/A'),
            "contexts": response.get('contexts', []),
            "ground_truth": item['ground_truth']
        })

    # 4. Prepare for Ragas
    ds = Dataset.from_list(eval_rows)
    
    # 5. Run Metrics
    print("--- Computing Ragas Metrics ---")
    results = evaluate(
        ds, 
        metrics=[faithfulness, answer_relevancy]
    )
    
    print(f"\nEvaluation Results:\n{results}")
    return results

if __name__ == "__main__":
    run_evaluation()
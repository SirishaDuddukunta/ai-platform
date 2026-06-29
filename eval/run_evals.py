import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# 1. Load the Golden Dataset
def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# 2. Mock Agent Call (Replace this with your actual API client)
def get_agent_response(question):
    # This simulates your FastAPI Agentic RAG endpoint
    # In reality, you'd use 'requests.post(...)'
    return {
        "answer": "The Acme Corp project is in the negotiation phase.",
        "contexts": ["Acme Corp project status is Contract Negotiation."]
    }

# 3. Main Eval Loop
def run_evaluation():
    data = load_data('eval_data.json')
    eval_rows = []

    for item in data:
        response = get_agent_response(item['question'])
        eval_rows.append({
            "question": item['question'],
            "answer": response['answer'],
            "contexts": response['contexts'],
            "ground_truth": item['ground_truth']
        })

    # 4. Prepare for Ragas
    ds = Dataset.from_list(eval_rows)
    
    # 5. Run Metrics
    results = evaluate(
        ds, 
        metrics=[faithfulness, answer_relevancy]
    )
    
    print(f"Evaluation Results: {results}")
    return results

if __name__ == "__main__":
    run_evaluation()
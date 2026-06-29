# Example structure for your Day 30 Benchmark Script
from ragas import evaluate
from datasets import Dataset

# 1. Load your Golden Dataset
golden_dataset = [
    {"question": "How do I update my profile?", "ground_truth": "Navigate to settings -> profile."},
    # ... add 19 more
]

# 2. Run your Agent
def get_agent_response(question):
    # This calls your FastAPI agent endpoint you built in Phase I-III
    return {"answer": "...", "contexts": ["..."]}

# 3. Prepare data for Ragas
# Ragas needs: question, answer, contexts, ground_truth
results = []
for item in golden_dataset:
    response = get_agent_response(item['question'])
    results.append({
        "question": item['question'],
        "answer": response['answer'],
        "contexts": response['contexts'],
        "ground_truth": item['ground_truth']
    })

# 4. Evaluate
ds = Dataset.from_list(results)
score = evaluate(ds, metrics=[...])
print(score)
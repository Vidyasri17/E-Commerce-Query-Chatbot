import os
import sys
import json

# Add parent directory to sys.path so agent can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from agent import app

def run_evaluation():
    dataset_path = os.path.join("evals", "dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} evaluation test cases.")
    print("Running intent classification accuracy tests...\n")
    print(f"{'ID':<4} | {'Query':<50} | {'Expected':<15} | {'Predicted':<15} | {'Result':<6}")
    print("-" * 100)

    correct_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        case_id = case["id"]
        query = case["query"]
        expected = case["expected_intent"]
        cust_id = case.get("customer_id", "")
        follow_up = case.get("follow_up_context", {})

        # Setup state
        state = {
            "customer_id": cust_id,
            "messages": [HumanMessage(content=query)],
            "intent": "",
            "active_sub_agent": "",
            "db_query_results": {},
            "follow_up_context": follow_up,
            "escalation_flag": False,
            "turn_count": 0,
            "consecutive_attempts": 0,
            "agent_response": ""
        }

        # Tracing configuration
        config = {
            "tags": ["eval_test", f"case_{case_id}"],
            "metadata": {
                "environment": "testing",
                "dataset": "intent_eval_25"
            }
        }
        if cust_id:
            config["metadata"]["customer_id"] = cust_id

        try:
            # We only need to run the intent classifier node to evaluate intent classification.
            # But invoking the full app is also completely fine and tests routing too!
            # Let's invoke the full app.
            result = app.invoke(state, config=config)
            predicted = result.get("intent", "unknown")
        except Exception as e:
            predicted = f"ERROR: {e}"

        is_correct = (predicted == expected)
        if is_correct:
            correct_count += 1
            result_str = "PASS"
        else:
            result_str = "FAIL"

        # Truncate query for neat logging
        trunc_query = query[:47] + "..." if len(query) > 50 else query
        print(f"{case_id:<4} | {trunc_query:<50} | {expected:<15} | {predicted:<15} | {result_str:<6}")

    accuracy = (correct_count / total_count) * 100
    print("-" * 100)
    print(f"\nEvaluation Summary:")
    print(f"Total Test Cases: {total_count}")
    print(f"Correct Classifications: {correct_count}")
    print(f"Accuracy Score: {accuracy:.2f}%")
    
    if accuracy >= 80.0:
        print("\nSUCCESS: Intent classification meets the performance target! (>= 80%)")
    else:
        print("\nWARNING: Intent classification accuracy is below target. Check prompts and retry.")

if __name__ == "__main__":
    run_evaluation()

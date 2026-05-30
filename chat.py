import sys
from langchain_core.messages import HumanMessage
from agent import app

def run_scripted_test():
    print("==================================================")
    print("RUNNING AUTOMATED E-2-E CONVERSATION TEST CASE")
    print("==================================================")
    
    # Initialize state
    state = {
        "customer_id": "",
        "messages": [],
        "intent": "",
        "active_sub_agent": "",
        "db_query_results": {},
        "follow_up_context": {},
        "escalation_flag": False,
        "turn_count": 0,
        "consecutive_attempts": 0,
        "agent_response": ""
    }
    
    # 1. First Turn: Identify customer & request orders
    turn_1_input = "Hi, I want to check my orders. Customer ID: C1001"
    print(f"\nUser: {turn_1_input}")
    state["messages"].append(HumanMessage(content=turn_1_input))
    state = app.invoke(state, config={"tags": ["test_turn_1"], "metadata": {"turn": 1}})
    print(f"Agent: {state['messages'][-1].content}")
    print(f"[Context]: Customer ID: {state.get('customer_id')}")
    
    # 2. Second Turn: Clarify order choice ("The jacket")
    turn_2_input = "The jacket"
    print(f"\nUser: {turn_2_input}")
    state["messages"].append(HumanMessage(content=turn_2_input))
    state = app.invoke(state, config={"tags": ["test_turn_2"], "metadata": {"turn": 2}})
    print(f"Agent: {state['messages'][-1].content}")
    print(f"[Context]: follow_up_context = {state.get('follow_up_context')}")
    
    # 3. Third Turn: Ask about returning it (utilizes follow_up_context O1002)
    turn_3_input = "Can I return it?"
    print(f"\nUser: {turn_3_input}")
    state["messages"].append(HumanMessage(content=turn_3_input))
    state = app.invoke(state, config={"tags": ["test_turn_3"], "metadata": {"turn": 3}})
    print(f"Agent: {state['messages'][-1].content}")
    print(f"[Context]: follow_up_context = {state.get('follow_up_context')}")
    
    # 4. Fourth Turn: Loop escalation test - send unresolvable queries 3 times
    print("\n--------------------------------------------------")
    print("RUNNING LOOP ESCALATION / FALLBACK TEST")
    print("--------------------------------------------------")
    
    unresolvable_state = {
        "customer_id": "C1001",
        "messages": [],
        "intent": "",
        "active_sub_agent": "",
        "db_query_results": {},
        "follow_up_context": {},
        "escalation_flag": False,
        "turn_count": 0,
        "consecutive_attempts": 0,
        "agent_response": ""
    }
    
    unresolvable_input = "Tell me the status of my order from next year"
    
    for attempt in range(1, 4):
        print(f"\nUser (Attempt {attempt}): {unresolvable_input}")
        unresolvable_state["messages"].append(HumanMessage(content=unresolvable_input))
        unresolvable_state = app.invoke(unresolvable_state, config={"tags": [f"test_loop_{attempt}"], "metadata": {"attempt": attempt}})
        print(f"Agent: {unresolvable_state['messages'][-1].content}")
        print(f"[State]: consecutive_attempts = {unresolvable_state.get('consecutive_attempts')}, escalation_flag = {unresolvable_state.get('escalation_flag')}")

    print("\n==================================================")
    print("E-2-E CONVERSATION TEST COMPLETED SUCCESSFULLY!")
    print("==================================================")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_scripted_test()
    else:
        # Standard interactive loop
        print("Starting E-Commerce Support Chatbot CLI...")
        print("Type 'exit' to quit. Type '--test' to run the scripted end-to-end conversation test.")
        print("-" * 50)
        
        state = {
            "customer_id": "",
            "messages": [],
            "intent": "",
            "active_sub_agent": "",
            "db_query_results": {},
            "follow_up_context": {},
            "escalation_flag": False,
            "turn_count": 0,
            "consecutive_attempts": 0,
            "agent_response": ""
        }
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input or user_input.lower() == "exit":
                    break
                if user_input == "--test":
                    run_scripted_test()
                    continue
                    
                state["messages"].append(HumanMessage(content=user_input))
                state = app.invoke(state, config={"tags": ["interactive"]})
                print(f"Agent: {state['messages'][-1].content}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()

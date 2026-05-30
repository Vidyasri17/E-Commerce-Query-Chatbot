import os
from langchainhub import Client
from langchain_core.prompts import ChatPromptTemplate

def upload_prompts():
    # Check for api key
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY is not set. Cannot upload to LangSmith Prompt Hub.")
        print("Our agent will run in local-fallback mode, reading files from the 'prompts/' directory.")
        return
        
    print("Uploading prompts to LangSmith Prompt Hub...")
    try:
        client = Client()
    except Exception as e:
        print(f"Error initializing LangSmith Prompt Hub Client: {e}")
        return

    prompts_to_upload = {
        "intent-classifier-prompt": "prompts/intent_classifier.txt",
        "order-agent-prompt": "prompts/order_agent.txt",
        "product-agent-prompt": "prompts/product_agent.txt",
        "returns-agent-prompt": "prompts/returns_agent.txt",
        "recommendation-agent-prompt": "prompts/recommendation_agent.txt",
        "fallback-agent-prompt": "prompts/fallback_agent.txt",
    }
    
    for hub_name, file_path in prompts_to_upload.items():
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping...")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        
        # We construct templates with structured fields
        if hub_name == "intent-classifier-prompt":
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("human", "Conversation History:\n{history}\n\nFollow-up Context:\n{follow_up_context}\n\nLatest User Message:\n{input}")
            ])
        elif hub_name == "fallback-agent-prompt":
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("human", "Conversation History:\n{history}\n\nLatest User Message:\n{input}")
            ])
        else:
            # Sub-agents
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("human", "Conversation History:\n{history}\n\nFollow-up Context:\n{follow_up_context}\n\nCustomer ID: {customer_id}\n\nDatabase Query Results:\n{db_results}\n\nUser Message:\n{input}")
            ])
        
        try:
            print(f"Pushing {hub_name} to LangSmith Hub...")
            client.push(hub_name, prompt)
            print(f"Successfully pushed {hub_name}!")
        except Exception as e:
            print(f"Error pushing {hub_name}: {e}")
            print("Make sure you are logged in to LangSmith and your LANGSMITH_API_KEY is correct.")

if __name__ == "__main__":
    upload_prompts()

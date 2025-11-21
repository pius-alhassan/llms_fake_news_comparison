import requests
import json

Base_URL = "http://127.0.0.1:5000"

print("Talk to the Gemini LLM! Type 'exit' to quit.")


def ask_llm():
    question = input("You: ")
    if not question:
        return
    try:
        response = requests.post(f"{Base_URL}/ask", json={"question":question})
        data = response.json()
        print("LLM:", data.get("answer") or data.get("error"))
    except Exception as e:
        print("Request failed", e)

def cleaned_text():
    text = input("Insert text to clean: ")
    if not text:
        return
    try:
        response = requests.post(f"{Base_URL}/clean", json={"text": text})
        data = response.json()
        print("Server Response:",json.dumps(data, indent=2))
    except Exception as e:
        print("Request failed:", e)

def extract_info():
    try:
        response = requests.get(f"{Base_URL}/extract")
        data = response.json()
        print("Extracted info:", json.dumps(data, indent=2))
    except Exception as e:
        print("Request failed:", e)

def main():
    print("Interactive LLM client!")
    print("Type 'exit' to quit at any time.\n")
    while True:
        choice = input("Choose endpoint (ask / clean / extract): ").strip().lower()
        if choice == "exit":
            break
        elif choice == "ask":
            ask_llm()
        elif choice == "clean":
            cleaned_text()
        elif choice == "extract":
            extract_info()
        else:
            print("Invalid choice. Please choose 'ask', 'clean', or 'extract'.")
    print("Goodbye!")

if __name__=="__main__":
    main()
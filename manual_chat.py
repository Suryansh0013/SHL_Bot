"""
Quick manual test loop: chat with your locally running API from the terminal.

Usage:
    python tests/manual_chat.py
    python tests/manual_chat.py --url https://your-deployed-app.onrender.com
"""
import argparse
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
args = parser.parse_args()

messages = []
print(f"Chatting with {args.url}/chat. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break
    messages.append({"role": "user", "content": user_input})

    resp = requests.post(f"{args.url}/chat", json={"messages": messages}, timeout=30)
    if resp.status_code != 200:
        print(f"[ERROR {resp.status_code}] {resp.text}")
        continue

    data = resp.json()
    print(f"\nAgent: {data['reply']}\n")
    if data.get("recommendations"):
        print("Recommendations:")
        for r in data["recommendations"]:
            print(f"  - {r['name']} ({r['test_type']}) -> {r['url']}")
        print()
    if data.get("end_of_conversation"):
        print("[agent marked end_of_conversation=true]\n")

    messages.append({"role": "assistant", "content": data["reply"]})

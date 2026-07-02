from app.llm_client import call_llm_json

system_prompt = """
You are a helpful assistant.
Always return valid JSON only.
"""

user_prompt = """
Return exactly this JSON:

{
    "status": "working",
    "provider": "groq"
}
"""

response = call_llm_json(system_prompt, user_prompt)

print(response)
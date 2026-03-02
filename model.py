# -------------------------------
# FULL AI DATING PROTOTYPE (Local GGUF)
# -------------------------------

import json
from llama_cpp import Llama

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "qwen2-7b-instruct-q4_0.gguf"

# Load the model once at startup
print("Loading model... (this may take a moment)")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,  # Use GPU if available, set to 0 for CPU only
    verbose=False
)
print("Model loaded!")

# -------------------------------
# HELPER FUNCTION TO CALL LLM
# -------------------------------
def call_llm(messages, temperature=0.7, max_tokens=24000):
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error during inference:", e)
        return None

# -------------------------------
# STAGE 1: INTAKE
# -------------------------------
def stage1_intake():
    # Stage 1 Intake system message
    system_msg = {
        "role": "system",
        "content": (
            "You are a thoughtful relationship coach helping a user understand what they want in a long-term partner. "
            "Your job right now is only to listen and extract structured preferences. "
            "Ask one open-ended question at a time. Do not overwhelm the user. "
            "DO NOT STOP ASKING UNTIL THERE IS ENOUGH INFORMATION TO FILL OUT ALL FIELDS IN THE JSON STRUCTURE. The goal is to collect enough information to fill the following JSON structure COMPLETELY:\n\n"
            "User preferences: {\n"
            "  \"core_values\": [],\n"
            "  \"emotional_needs\": [],\n"
            "  \"deal_breakers\": [],\n"
            "  \"attachment_style\": [],\n"
            "  \"love_languages\": [],\n"
            "  \"gender\": [],\n"
            "  \"age_range\": [],\n"
            "  \"appearance_preferences\": [],\n"
            "  \"lifestyle_habits\": [],\n"
            "}\n\n"
            "Stop asking questions and output this JSON ONLY when you have enough information to fill every field. "
            "Do not generate a partner profile yet. Collect and clarify information interactively. "
            "After each user response, ask a natural follow-up question until the JSON can be filled."
            "When you have gathered enough information for ALL categories above, provide a warm, conversational summary "
            "of what you've learned about their preferences. Start your summary with 'SUMMARY:' and describe their "
            "preferences naturally without any JSON or structured data. Keep it human and friendly."
            
        )
    }

    user_msg = {"role": "user", "content": "Hi, I want to explore my relationship preferences."}

    messages = [system_msg, user_msg]

    print("\nAI Relationship Coach (Stage 1 Intake):")
    while True:
        ai_response = call_llm(messages, max_tokens=300)
        print("AI:", ai_response)

        # Check if AI has completed gathering preferences
        if "SUMMARY:" in ai_response:
            break

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": ai_response})

        # Get user input
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})

    # Extract structured JSON internally (not shown to user)
    preference_json = extract_preferences_json(messages)
    return preference_json


def extract_preferences_json(conversation_messages):
    """Extract structured preferences from conversation without showing to user."""
    extraction_msg = {
        "role": "system",
        "content": (
            "Based on the conversation, extract the user's relationship preferences into JSON format. "
            "Output ONLY valid JSON, nothing else:\n"
            "{\n"
            '  "core_values": [],\n'
            '  "emotional_needs": [],\n'
            '  "deal_breakers": [],\n'
            '  "attachment_style": "",\n'
            '  "love_languages": [],\n'
            '  "gender": "",\n'
            '  "age_range": "",\n'
            '  "appearance_preferences": [],\n'
            '  "lifestyle_habits": []\n'
            "}"
        )
    }
    
    # Build context from conversation
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in conversation_messages 
        if msg['role'] in ['user', 'assistant']
    ])
    
    messages = [extraction_msg, {"role": "user", "content": f"Extract preferences from this conversation:\n{conversation_text}"}]
    
    response = call_llm(messages, temperature=0.1, max_tokens=500)
    
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_str = response[json_start:json_end]
        return json.loads(json_str)
    except Exception as e:
        print(f"Warning: Could not parse preferences: {e}")
        return {
            "core_values": [],
            "emotional_needs": [],
            "deal_breakers": [],
            "attachment_style": "",
            "love_languages": [],
            "gender": "",
            "age_range": "",
            "appearance_preferences": [],
            "lifestyle_habits": []
        }

# -------------------------------
# STAGE 2: TENSION DETECTION + CLARIFICATION
# -------------------------------
def stage2_tension(preference_json):
    system_msg = {
        "role": "system",
        "content": (
            "You are analyzing a user's relationship preferences. "
            "Detect internal contradictions (tensions) and ask one clarifying question at a time. "
            "Do not resolve them yourself yet. After tensions are resolved, scan for vague preferences "
            "and ask follow-ups to make them concrete. Stop when the JSON is detailed enough to generate a profile."
        )
    }

    messages = [system_msg, {"role": "user", "content": f"Here are the user's preferences: {json.dumps(preference_json)}"}]

    print("\nAI Relationship Coach (Stage 2 Tension Detection + Clarification):")
    while True:
        ai_response = call_llm(messages)
        print("AI:", ai_response)

        if "resolved" in ai_response.lower() or "all clarifications made" in ai_response.lower():
            break

        # User answers tension / clarification
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})

    # Simulate updated JSON (in real prototype, you would parse LLM output)
    updated_json = preference_json  # placeholder for simplicity
    return updated_json

# -------------------------------
# STAGE 3: PROFILE GENERATION
# -------------------------------
def stage3_profile(preference_json):
    system_msg = {
        "role": "system",
        "content": (
            "You are generating a fictional partner profile based on the user's resolved preferences. "
            "Create a real-feeling, textured profile grounded in the user's stated values and trade-offs. "
            "Include name, personality summary, traits, backstory, typical day, conflict style, friendship style, and version. "
            "Include a 'why_this_profile' field explaining your choices. Include a follow-up question to the user."
        )
    }

    messages = [system_msg, {"role": "user", "content": f"Generate a partner profile using these preferences: {json.dumps(preference_json)}"}]

    print("\nAI Relationship Coach (Stage 3 Profile Generation):")
    ai_response = call_llm(messages)
    print("AI Generated Profile:\n", ai_response)

    # Placeholder parsing
    profile_json = {"profile": "Generated profile JSON placeholder", "version": 1}
    return profile_json

# -------------------------------
# STAGE 4: REFINEMENT
# -------------------------------
def stage4_refinement(preference_json, profile_json):
    system_msg = {
        "role": "system",
        "content": (
            "You are refining a fictional partner profile based on user feedback. "
            "Update the profile JSON, increment version, explain what changed, and regenerate the profile."
        )
    }

    print("\nStage 4: Refinement Loop (type 'done' to finish)")

    messages = [system_msg]

    while True:
        feedback = input("Your feedback for refinement (or 'done' to finish): ")
        if feedback.lower() == "done":
            break
        messages.append({"role": "user", "content": f"Profile JSON: {json.dumps(profile_json)}, Feedback: {feedback}"})
        ai_response = call_llm(messages)
        print("AI Updated Profile:\n", ai_response)

# -------------------------------
# RUN FULL PROTOTYPE
# -------------------------------
def run_prototype():
    print("===== Stage 1: Intake =====")
    user_preferences = stage1_intake()

    print("\n===== Stage 2: Tension Detection =====")
    user_preferences = stage2_tension(user_preferences)

    print("\n===== Stage 3: Profile Generation =====")
    profile_json = stage3_profile(user_preferences)

    print("\n===== Stage 4: Refinement =====")
    stage4_refinement(user_preferences, profile_json)

# -------------------------------
# START
# -------------------------------
run_prototype()
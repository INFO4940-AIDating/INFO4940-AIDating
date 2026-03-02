# -------------------------------
# FULL AI DATING PROTOTYPE (Local GGUF)
# -------------------------------

import json
from llama_cpp import Llama

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "../qwen2-7b-instruct-q4_0.gguf"

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
def call_llm(messages, temperature=0.7, max_tokens=400):
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
            "  \"attachment_style\": \"\",\n"
            "  \"love_languages\": [],\n"
            "}\n\n"
            "Stop asking questions and output this JSON ONLY when you have enough information to fill every field. "
            "Do not generate a partner profile yet. Collect and clarify information interactively. "
            "After each user response, ask a natural follow-up question until the JSON can be filled."
            
        )
    }

    user_msg = {"role": "user", "content": "Hi, I want to explore my relationship preferences."}

    messages = [system_msg, user_msg]

    print("\nAI Relationship Coach (Stage 1 Intake):")
    preference_json = None
    while True:
        ai_response = call_llm(messages)
        print("AI:", ai_response)

        # Check if AI signals JSON output
        if "User preferences:" in ai_response:
            try:
                # Extract JSON substring - find matching braces
                json_start = ai_response.find("{")
                json_end = ai_response.rfind("}") + 1  # Find LAST closing brace
                json_str = ai_response[json_start:json_end]
                preference_json = json.loads(json_str)
            except Exception as e:
                print("Error parsing JSON:", e)
            break

        # Get user input
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "system", "content": "Continue asking the next question based on the user's answers."})

    return preference_json

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
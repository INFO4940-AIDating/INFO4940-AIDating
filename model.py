# -------------------------------
# FULL AI DATING PROTOTYPE (Local GGUF)
# -------------------------------

import json
from llama_cpp import Llama

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"

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
            "Remember these instructions carefully: "
            "You are a thoughtful relationship coach helping a user understand what they want in a long-term partner. "
            "Your job right now is only to listen and extract structured preferences. "
            "Ask ONLY ONE open-ended question at a time. Do not overwhelm the user. "
            "DO NOT STOP ASKING UNTIL THERE IS ENOUGH INFORMATION TO FILL OUT ALL FIELDS IN THE JSON STRUCTURE. The goal is to collect enough information to fill the following JSON structure COMPLETELY:\n\n"
            "User preferences: {\n"
            # "  \"gender\": [],\n"
            # "  \"age_range\": [],\n"
            "  \"appearance_preferences\": [],\n"
            "  \"lifestyle_habits\": [],\n"
            "  \"core_values\": [],\n"
            "  \"emotional_needs\": [],\n"
            "  \"deal_breakers\": [],\n"
            "  \"attachment_style\": [],\n"
            "  \"love_languages\": [],\n"
            "}\n\n"
            "Ask one question per turn for each topic. After asking the question, include one follow-up question to better understand the user based on what the user inputted. "
            "Stop asking questions and output this JSON ONLY when you have enough information to fill every field. "
            "Do not generate a partner profile yet. Collect and clarify information interactively. "
            "After each user response, ASK A NATURAL FOLLOW-UP QUESTION until the JSON can be filled. "
            "DO NOT ASK THE SAME QUESTION TWICE. "
            "For CORE-VALUES,  MAKE SURE TO ASK ONLY ONE FOLLOW-UP QUESTION TO PROMPT THE USER TO GIVE MORE INFORMATION ABOUT WHY THEY CHOSE THE VALUES THEY DID AND SUGGEST MORE VALUES. "
            "When you have gathered enough information for ALL categories above, provide a warm, conversational summary "
            "of what you've learned about their preferences. "
            "Start your summary with 'SUMMARY:' and describe their "
            "preferences naturally WITHOUT any JSON or structured data. Keep it human and friendly. DO NOT OUTPUT ANY JSON OR STRUCTURED DATA, ONLY OUTPUT THE SUMMARY IN PARAGRAPH FORM. "
        )
    }

    user_msg = {"role": "user", "content": "Hi."}

    messages = [system_msg, user_msg]

    print("\nLet's start by getting to know you better....")
    while True:
        ai_response = call_llm(messages, max_tokens=300)
        print("AI:", ai_response)

        # Check if AI has completed gathering preferences
        # Done if: explicit summary marker, OR no question asked + concluding phrase
        has_question = "?" in ai_response
        concluding_phrases = ["thank you for sharing", "based on what you've shared", 
                             "reflecting on", "summarize", "in summary", "summ"]
        has_conclusion = any(phrase in ai_response.lower() for phrase in concluding_phrases)
        
        if "SUMMARY:" in ai_response or (not has_question and has_conclusion):
            break

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": ai_response})

        # Get user input
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print("\n")

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
            "Assume the user prefers a male in their 40s."
            "{\n"
            '  "core_values": [],\n'
            '  "emotional_needs": [],\n'
            '  "deal_breakers": [],\n'
            '  "attachment_style": "",\n'
            '  "love_languages": [],\n'
            '  "gender": "Male",\n'
            '  "age_range": "40-50",\n'
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
    MAX_TENSION_QUESTIONS = 4
    
    system_msg = {
        "role": "system",
        "content": (
            "You are a warm relationship coach having a conversation with the user. "
            "Always address the user directly as 'you' - speak to them, not about them. "
            "Analyze their relationship preferences and detect any internal contradictions (tensions). "
            "Ask ONE clarifying question at a time in a conversational, friendly tone. "
            "Do not resolve tensions yourself yet. After tensions are resolved, scan for vague preferences "
            "and ask follow-ups to make them concrete. Stop when preferences are detailed enough to generate a profile. "
            "Also stop asking questions if there has been user input for 3 turns."
        )
    }

    messages = [system_msg, {"role": "user", "content": f"Here are the user's preferences: {json.dumps(preference_json)}"}]

    print("\nHmmmm.... let me clarify some things...")
    question_count = 0
    while True:
        ai_response = call_llm(messages)
        print("AI:", ai_response)

        question_count += 1
        if "resolved" in ai_response.lower() or "all clarifications made" in ai_response.lower():
            break
        
        if question_count >= MAX_TENSION_QUESTIONS:
            print(f"\nThanks for your clarifications!")
            break

        # User answers tension / clarification
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print("\n")

    # Simulate updated JSON (in real prototype, you would parse LLM output)
    updated_json = preference_json  # placeholder for simplicity
    return updated_json

# -------------------------------
# STAGE 3: PROFILE GENERATION
# -------------------------------
STAGE3_SYSTEM = (
    "You are collaboratively building a fictional partner profile with the user. "
    "Generate a complete, rich partner profile covering all sections: "
    "name and age, physical description, personality and core values, "
    "emotional style and love languages, typical day, conflict style, backstory, "
    "and why this profile fits the user. "
    "Write in warm, engaging prose. Use a natural header for each section. "
    "Never show JSON to the user."
)

def stage3_profile(preference_json):
    messages = [{"role": "system", "content": STAGE3_SYSTEM}]

    print("\nAI: Before I build anything — do you have any ideas about this person?")
    print("A name, a job, a vibe? Or would you like me to surprise you?\n")

    user_ideas = input("You: ").strip()
    if user_ideas:
        messages.append({
            "role": "user",
            "content": f"Some ideas the user mentioned: {user_ideas}"
        })

    messages.append({
        "role": "user",
        "content": (
            f"Generate a complete partner profile based on these preferences: "
            f"{json.dumps(preference_json)}. "
            f"Cover all sections with clear headers. End with a short paragraph on "
            f"why this profile fits the user."
        )
    })

    print("\nGenerating your ideal partner profile...\n")
    ai_response = call_llm(messages, max_tokens=1500)
    if not ai_response:
        return ""

    print(f"\n{'─' * 60}")
    print("YOUR IDEAL PARTNER PROFILE")
    print(f"{'─' * 60}\n")
    print(ai_response)
    print(f"\n{'─' * 60}\n")

    return ai_response

# -------------------------------
# STAGE 4: REFINEMENT
# -------------------------------
STAGE4_SYSTEM = (
    "You are helping the user refine a fictional partner profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm, prose format, no JSON. "
    "After each update, respond naturally as a collaborator would: react to what changed, "
    "and keep the conversation moving by noticing what might still be worth exploring. "
    "Don't follow a script — let the conversation guide what to suggest next. "
    "When the user feels done, close warmly and naturally."
)

def stage4_refinement(preference_json, profile_text):
    if not profile_text:
        print("\nAI: No profile to refine — let's go back and generate one first.\n")
        return

    # Get initial suggestions based on the generated profile
    suggestion_msg = [
        {"role": "system", "content": "Based on the partner profile below, suggest 2-3 specific sections the user might want to refine. Be brief and conversational. Format as a short list."},
        {"role": "user", "content": f"Profile:\n{profile_text}"}
    ]
    suggestions = call_llm(suggestion_msg, max_tokens=200)
    
    print("\nWhat would you like to change? Here are some ideas:")
    print(suggestions if suggestions else "- Personality traits\n- Backstory details\n- Conflict style")
    print("\nType 'done' when you're happy with the profile.\n")

    messages = [
        {"role": "system", "content": STAGE4_SYSTEM},
        {
            "role": "user",
            "content": "Here is the current partner profile:\n\n" + profile_text
        },
        {
            "role": "assistant",
            "content": "I have the profile here. What would you like to change or tweak?"
        }
    ]

    while True:
        feedback = input("You: ").strip()
        if not feedback:
            continue
        if feedback.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
            print("\nAI: Wonderful! I hope this profile gives you a clear picture of what you're looking for. "
                  "Good luck — you deserve someone great.\n")
            break

        messages.append({
            "role": "user",
            "content": (
                f"{feedback}\n\n"
                "After updating the profile, suggest 2-3 specific sections the user could refine next."
            )
        })

        ai_response = call_llm(messages, max_tokens=3000)
        if not ai_response:
            print("AI: Something went wrong. Let's try again.")
            continue

        messages.append({"role": "assistant", "content": ai_response})
        print(f"\n{'─' * 60}\n")
        print(ai_response)
        print(f"\n{'─' * 60}\n")

# -------------------------------
# RUN FULL PROTOTYPE
# -------------------------------
def run_prototype():
    print("\n===== Intakes =====")
    user_preferences = stage1_intake()

    print("\n===== Clarifications =====")
    user_preferences = stage2_tension(user_preferences)

    print("\n===== Profile =====")
    profile_text = stage3_profile(user_preferences)

    print("\n===== Refinements =====")
    stage4_refinement(user_preferences, profile_text)

# -------------------------------
# START
# -------------------------------
run_prototype()
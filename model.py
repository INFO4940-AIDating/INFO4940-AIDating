# -------------------------------
# AI RELATIONSHIP PROFILE BUILDER
# -------------------------------

import json

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"
TEST_MODE = False  # Set to True to run flow test without the real model

# -------------------------------
# TEST MODE — mock LLM responses and user inputs
# These simulate one full run through the flow to verify every stage.
# -------------------------------
_test_llm_idx = 0
_test_input_idx = 0

# 16 mock LLM responses — ordered to match the exact call sequence:
#   Stage 1: 6 questions (indices 0-5) + 1 SUMMARY (6)
#     (relationship type is asked upfront in run_prototype, not by the AI)
#   extract_preferences_json: 1 JSON response (7)
#   stage_ranking round 1: rankings + review question (8)
#   stage_ranking round 2: RANKINGS CONFIRMED (9)
#   stage_ranking extract: JSON (10)
#   stage2_tension: 3 questions (11-13)
#   stage3_profile: profile text (14)
#   stage4_refinement suggestions: suggestion text (15)
_TEST_LLM_RESPONSES = [
    # Stage 1 — learning about the user (relationship type already known, so not asked again)
    "Tell me a bit about yourself — who are you as a person?",
    "What kinds of things bring you the most joy or meaning in your day-to-day life?",
    "How do you tend to show up in close relationships — what role do you usually play?",
    "Do you have any preferences when it comes to age or gender?",
    "What qualities do you most hope this person brings to the relationship?",
    "Have there been any red flags or deal breakers in past relationships that you'd want to avoid this time?",
    # SUMMARY triggers end of Stage 1 loop
    "SUMMARY: You are a thoughtful, introverted software engineer in your early 30s who values genuine connection and intellectual depth. You are looking for a romantic partner — a woman in her late 20s to mid-30s — who is emotionally intelligent, kind, and curious about the world. Past experiences have shown you that emotional unavailability is a firm deal breaker for you.",
    # extract_preferences_json — must be valid JSON
    '{"user_profile": {"personality": "introverted, analytical, thoughtful", "lifestyle": "enjoys hiking, reading, long conversations", "values": ["authenticity", "growth", "connection"], "relationship_style": "supportive, values depth"}, "relationship_type": "romantic", "gender_preference": "female", "age_range": "27-35", "appearance_preferences": [], "lifestyle_habits": [], "core_values": ["kindness", "emotional intelligence", "intellectual curiosity"], "emotional_needs": ["emotional availability", "deep conversation", "stability"], "deal_breakers": ["emotional unavailability", "dismissiveness"], "attachment_style": "", "love_languages": ["quality time", "words of affirmation"]}',
    # stage_ranking — round 1: present ranked lists and ask for review
    "Based on who you are and what you have shared, here is what I believe you value most in a partner:\n\n**Core Values (ranked):**\n1. Emotional intelligence\n2. Kindness\n3. Intellectual curiosity\n\n**Emotional Needs (ranked):**\n1. Emotional availability\n2. Deep, meaningful conversation\n3. Stability and consistency\n\n**Personality Traits:**\n1. Empathetic\n2. Curious\n3. Grounded\n\n**Love Languages:**\n1. Quality time\n2. Words of affirmation\n\nDoes this feel right to you, or would you like to shift, add, or remove anything?",
    # stage_ranking — round 2: user confirms, RANKINGS CONFIRMED triggers exit
    "RANKINGS CONFIRMED",
    # stage_ranking extract — valid JSON
    '{"ranked_core_values": ["emotional intelligence", "kindness", "intellectual curiosity"], "ranked_emotional_needs": ["emotional availability", "deep conversation", "stability"], "ranked_personality_traits": ["empathetic", "curious", "grounded"], "ranked_love_languages": ["quality time", "words of affirmation"], "ranked_lifestyle_habits": ["values quiet evenings", "intellectually active", "enjoys nature"], "deal_breakers": ["emotional unavailability", "dismissiveness"]}',
    # stage2_tension — 3 clarifying questions (3rd triggers MAX break, no user input needed)
    "I noticed you value both deep intellectual connection and emotional warmth — sometimes those pull in different directions. How important is it that this person matches your intellectual pace versus simply being emotionally present?",
    "That makes sense. You also mentioned stability alongside curiosity and growth. How do you balance wanting someone grounded with wanting someone who keeps evolving?",
    "That gives me a much clearer picture — I think you have a strong sense of what you need.",
    # stage3_profile — full profile text
    "## Meet Nora\n\n**Name & Age:** Nora, 31\n\n**Physical Description:** Nora carries herself with quiet warmth — unhurried and comfortable in her own skin.\n\n**Personality & Core Values:** At her core, Nora is deeply empathetic and emotionally attuned. She values authenticity above all else.\n\n**Emotional Style & Love Languages:** Nora's primary love language is quality time. She is fully present when she is with someone she cares about.\n\n**A Typical Day:** Nora starts her mornings slowly with coffee and whatever book she is halfway through.\n\n**Conflict Style:** Nora avoids drama. When tensions arise she prefers to take a breath, then talk things through calmly and honestly.\n\n**Backstory:** Nora grew up in a mid-sized city, the middle child in a close-knit family.\n\n**Why This Fits You:** Nora embodies the emotional availability and intellectual depth you value most. She will not shut down when things get hard — she shows up.",
    # stage4_refinement — initial suggestions
    "A few things you might want to personalize: Nora's specific career, her relationship with her family, or a small quirk that makes her feel real.",
]

# 12 mock user inputs — ordered to match the exact input() call sequence
_TEST_USER_INPUTS = [
    # Upfront relationship type question (run_prototype)
    "A romantic relationship — I am looking for a serious long-term partner.",
    # Stage 1 — 6 answers (one per question before SUMMARY; relationship type already known)
    "I am a 30-year-old software engineer. Pretty introverted and analytical.",
    "Reading, hiking, and long conversations about ideas that go nowhere useful.",
    "I am usually the supportive one. I care deeply about the people close to me.",
    "Women — probably late 20s to mid 30s.",
    "Emotional intelligence, kindness, someone genuinely curious about the world.",
    "People who shut down emotionally or become dismissive when things get hard.",
    # stage_ranking — 1 answer after seeing ranked lists
    "That looks right. I would put emotional availability as number one under emotional needs.",
    # stage2_tension — 2 answers (3rd question hits MAX and breaks without input)
    "Emotional presence matters more to me than matching my intellect exactly.",
    "I want someone grounded but still open to growth. They do not need to be at my pace.",
    # stage3_profile — user ideas prompt
    "Surprise me.",
    # stage4_refinement — user says done
    "done",
]


def _mock_input(prompt=""):
    global _test_input_idx
    print(prompt, end="", flush=True)
    if _test_input_idx < len(_TEST_USER_INPUTS):
        val = _TEST_USER_INPUTS[_test_input_idx]
        print(val)
        _test_input_idx += 1
        return val
    print("done")
    return "done"


if TEST_MODE:
    input = _mock_input

# -------------------------------
# MODEL LOADING (skipped in test mode)
# -------------------------------
if not TEST_MODE:
    from llama_cpp import Llama
    print("Loading model... (this may take a moment)")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False
    )
    print("Model loaded!")

# -------------------------------
# HELPER FUNCTION TO CALL LLM
# -------------------------------
def call_llm(messages, temperature=0.7, max_tokens=24000):
    if TEST_MODE:
        global _test_llm_idx
        if _test_llm_idx < len(_TEST_LLM_RESPONSES):
            response = _TEST_LLM_RESPONSES[_test_llm_idx]
            print(f"  [TEST: mock LLM response #{_test_llm_idx}]")
            _test_llm_idx += 1
            return response
        return "I think we have everything we need."
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
# First learn who the user is as a person, then what they are looking for.
# -------------------------------
def stage1_intake(relationship_type=""):
    relationship_context = (
        f"The user has already told you they are looking for: {relationship_type}. "
        "Do NOT ask about relationship type again — that has already been established. "
        "You may reference it naturally when relevant.\n\n"
    ) if relationship_type else ""

    system_msg = {
        "role": "system",
        "content": (
            "You are a warm, curious conversational coach helping someone understand what they want "
            "in a relationship. Your first goal is to get to know the USER as a person — "
            "their personality, how they move through life, what they care about, "
            "and how they tend to show up in relationships. "
            "Only after you know who they are should you ask about their preferences "
            "for the other person (such as age range and gender).\n\n"
            + relationship_context +
            "Follow this natural conversational arc:\n"
            "1. Warmly invite the user to tell you about themselves as a person.\n"
            "2. Ask thoughtful follow-up questions to understand their personality, lifestyle, and values.\n"
            "3. Ask about their preferences for the other person (such as age range and gender).\n"
            "4. Ask about the qualities, traits, and emotional needs they hope to find.\n"
            "5. Ask about past red flags or deal breakers they have experienced.\n\n"
            "Strict rules:\n"
            "- Ask ONLY ONE question per turn. Never stack multiple questions in a single message.\n"
            "- NEVER include examples, suggestions, or anchor words in your questions. "
            "Let the user answer entirely in their own words without any prompting.\n"
            "- Be warm and conversational — like a thoughtful friend, not an interviewer with a checklist.\n"
            "- Do not revisit topics the user has already addressed.\n"
            "- When you have gathered enough about both the user AND what they are looking for, "
            "write a warm, natural summary. "
            "Start the summary with exactly the word 'SUMMARY:' on its own and write only in "
            "paragraph form — no lists, no JSON, no structured data.\n"
        )
    }

    initial_user_content = (
        f"Hi. I am looking to explore: {relationship_type}." if relationship_type else "Hi."
    )
    messages = [system_msg, {"role": "user", "content": initial_user_content}]

    print("\nGreat — now let's get to know you a bit better before we build your profile.\n")
    while True:
        ai_response = call_llm(messages, max_tokens=300)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "SUMMARY:" in ai_response:
            break

        concluding_phrases = ["thank you for sharing", "based on what you've shared", "in summary"]
        has_conclusion = any(p in ai_response.lower() for p in concluding_phrases)
        if "?" not in ai_response and has_conclusion:
            break

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print()

    preference_json = extract_preferences_json(messages)
    return preference_json


def extract_preferences_json(conversation_messages):
    """Extract structured preferences from conversation — never shown to user."""
    extraction_msg = {
        "role": "system",
        "content": (
            "Based on the conversation, extract structured information into JSON format. "
            "Output ONLY valid JSON, nothing else. Use empty strings or empty lists "
            "if something was not mentioned.\n"
            "{\n"
            '  "user_profile": {\n'
            '    "personality": "",\n'
            '    "lifestyle": "",\n'
            '    "values": [],\n'
            '    "relationship_style": ""\n'
            '  },\n'
            '  "relationship_type": "",\n'
            '  "gender_preference": "",\n'
            '  "age_range": "",\n'
            '  "appearance_preferences": [],\n'
            '  "lifestyle_habits": [],\n'
            '  "core_values": [],\n'
            '  "emotional_needs": [],\n'
            '  "deal_breakers": [],\n'
            '  "attachment_style": "",\n'
            '  "love_languages": []\n'
            "}"
        )
    }

    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_messages
        if msg['role'] in ['user', 'assistant']
    ])

    messages = [
        extraction_msg,
        {"role": "user", "content": f"Extract preferences from this conversation:\n{conversation_text}"}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=600)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception as e:
        print(f"Warning: Could not parse preferences: {e}")
        return {
            "user_profile": {"personality": "", "lifestyle": "", "values": [], "relationship_style": ""},
            "relationship_type": "",
            "gender_preference": "",
            "age_range": "",
            "appearance_preferences": [],
            "lifestyle_habits": [],
            "core_values": [],
            "emotional_needs": [],
            "deal_breakers": [],
            "attachment_style": "",
            "love_languages": []
        }

# -------------------------------
# STAGE 2: FRAMEWORK-BASED RANKING
# Use MBTI, attachment style, and love language theory to infer ranked
# priorities from who the user is. User proofreads and corrects.
# -------------------------------
def stage_ranking(preference_json):
    system_msg = {
        "role": "system",
        "content": (
            "You are a relationship coach who uses personality frameworks to help people understand "
            "what they truly need in a connection. "
            "You have detailed information about the user — who they are as a person — "
            "and what they have said they are looking for. "
            "Using MBTI personality theory, attachment style theory, and love language frameworks, "
            "generate ranked priority lists of what you believe this specific user would value most. "
            "Base the rankings primarily on who the user IS — their personality, values, and "
            "relationship style — not only on what they explicitly said they want. "
            "Look for compatibility and deeper unspoken needs.\n\n"
            "Present the rankings conversationally. Briefly explain the reasoning behind the top items "
            "so the user understands why you placed them there. "
            "Then ask the user to review: they can reorder, remove, or add anything that feels off or missing. "
            "After they respond, update the rankings and confirm. "
            "Once the rankings feel accurate to the user, output exactly 'RANKINGS CONFIRMED' and stop.\n\n"
            "Strict rules:\n"
            "- NEVER give examples when asking for corrections. Let the user tell you what to change.\n"
            "- Frame these as your best inferences for the user to validate — collaborative, not prescriptive.\n"
            "- Maximum 3 rounds of back-and-forth before finalizing.\n"
        )
    }

    user_context = (
        f"Here is what I know about the user and what they are looking for:\n"
        f"{json.dumps(preference_json, indent=2)}"
    )

    messages = [system_msg, {"role": "user", "content": user_context}]

    print("\nLet me use what you've shared to map out what matters most to you...\n")
    round_count = 0
    while round_count < 3:
        ai_response = call_llm(messages, max_tokens=600)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "RANKINGS CONFIRMED" in ai_response:
            break

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ").strip()
        messages.append({"role": "user", "content": user_input})
        print()
        round_count += 1

    if round_count == 3:
        print("AI: Great — I have noted your adjustments. Let's move on.\n")

    # Extract the final agreed rankings and merge into preference_json
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if m['role'] in ['user', 'assistant']
    ])

    ranking_extract_msg = [
        {
            "role": "system",
            "content": (
                "Based on the ranking conversation, extract the final agreed-upon ranked priorities "
                "into JSON format. Output ONLY valid JSON:\n"
                "{\n"
                '  "ranked_core_values": [],\n'
                '  "ranked_emotional_needs": [],\n'
                '  "ranked_personality_traits": [],\n'
                '  "ranked_love_languages": [],\n'
                '  "ranked_lifestyle_habits": [],\n'
                '  "deal_breakers": []\n'
                "}"
            )
        },
        {"role": "user", "content": conversation_text}
    ]

    ranking_response = call_llm(ranking_extract_msg, temperature=0.1, max_tokens=400)
    try:
        json_start = ranking_response.find("{")
        json_end = ranking_response.rfind("}") + 1
        updated_rankings = json.loads(ranking_response[json_start:json_end])
        preference_json.update(updated_rankings)
    except Exception as e:
        print(f"Warning: Could not parse rankings: {e}")

    return preference_json

# -------------------------------
# STAGE 3: TENSION DETECTION + CLARIFICATION
# -------------------------------
def stage2_tension(preference_json):
    MAX_TENSION_QUESTIONS = 3

    system_msg = {
        "role": "system",
        "content": (
            "You are a warm relationship coach having a conversation with the user. "
            "Always address the user directly as 'you' — speak to them, not about them. "
            "Analyze their relationship preferences and detect any internal contradictions or tensions. "
            "Ask ONE clarifying question at a time in a conversational, friendly tone. "
            "Do not resolve tensions yourself — let the user think through them. "
            "Stop when the preferences are clear enough to generate a profile, or after 3 turns."
        )
    }

    messages = [
        system_msg,
        {"role": "user", "content": f"Here are the preferences: {json.dumps(preference_json)}"}
    ]

    print("\nLet me think through a couple of things with you...\n")
    question_count = 0
    while True:
        ai_response = call_llm(messages, max_tokens=300)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        question_count += 1
        if "resolved" in ai_response.lower() or question_count >= MAX_TENSION_QUESTIONS:
            if question_count >= MAX_TENSION_QUESTIONS:
                print("Thanks for working through that with me!\n")
            break

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print()

    return preference_json

# -------------------------------
# STAGE 4: PROFILE GENERATION
# -------------------------------
STAGE3_SYSTEM = (
    "You are collaboratively building a relationship profile with the user. "
    "Generate a complete, rich profile covering all relevant sections based on the relationship type: "
    "name and age, physical description, personality and core values, "
    "emotional style and love languages, a typical day, conflict style, backstory, "
    "and why this profile fits the user specifically. "
    "Write in warm, engaging prose. Use a clear header for each section. "
    "Never show JSON to the user. "
    "Reflect the ranked priorities throughout the profile — the top-ranked traits should come through clearly. "
    "Ensure the profile does NOT include any of the user's stated deal breakers or past red flags."
)

def stage3_profile(preference_json):
    messages = [{"role": "system", "content": STAGE3_SYSTEM}]

    print("AI: Before I build the profile — do you have anything specific in mind?")
    print("A name, a vibe, a detail you definitely want included? Or should I surprise you?\n")

    user_ideas = input("You: ").strip()
    if user_ideas and user_ideas.lower() not in {"surprise me", "surprise", "no", "nope", ""}:
        messages.append({
            "role": "user",
            "content": f"The user wants to include these ideas: {user_ideas}"
        })

    messages.append({
        "role": "user",
        "content": (
            f"Generate a complete profile based on these preferences: {json.dumps(preference_json)}. "
            f"Cover all sections with clear headers. End with a short paragraph explaining "
            f"why this profile is a strong match for this specific user. "
            f"Reflect the ranked priorities and exclude all deal breakers."
        )
    })

    print("\nGenerating your profile...\n")
    ai_response = call_llm(messages, max_tokens=1500)
    if not ai_response:
        return ""

    print(f"\n{'─' * 60}")
    print("YOUR IDEAL PROFILE")
    print(f"{'─' * 60}\n")
    print(ai_response)
    print(f"\n{'─' * 60}\n")

    return ai_response

# -------------------------------
# STAGE 5: REFINEMENT
# -------------------------------
STAGE4_SYSTEM = (
    "You are helping the user refine a relationship profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm prose format, no JSON. "
    "React naturally as a collaborator: acknowledge what changed, and notice what else might be worth exploring. "
    "When the user is done, close warmly. "
    "Never include anything the user has flagged as a deal breaker or red flag."
)

def stage4_refinement(preference_json, profile_text):
    if not profile_text:
        print("\nAI: No profile to refine yet.\n")
        return

    suggestion_msg = [
        {
            "role": "system",
            "content": (
                "Based on the profile below, suggest 2-3 specific things the user might want "
                "to personalize or adjust. Be brief and conversational."
            )
        },
        {"role": "user", "content": f"Profile:\n{profile_text}"}
    ]
    suggestions = call_llm(suggestion_msg, max_tokens=150)

    print("What would you like to change? A few things you might consider:")
    print(suggestions if suggestions else "The backstory, the career, or a small quirk that makes the person feel real.")
    print("\nType 'done' when you're happy with it.\n")

    messages = [
        {"role": "system", "content": STAGE4_SYSTEM},
        {"role": "user", "content": "Here is the current profile:\n\n" + profile_text},
        {"role": "assistant", "content": "I have the profile here. What would you like to tweak?"}
    ]

    while True:
        feedback = input("You: ").strip()
        if not feedback:
            continue
        if feedback.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
            print(
                "\nAI: Wonderful! I hope this profile gives you a clear sense of what you're looking for. "
                "Good luck — you deserve someone great.\n"
            )
            break

        messages.append({
            "role": "user",
            "content": (
                f"{feedback}\n\n"
                "After updating the profile, briefly note what changed and suggest "
                "one or two things that might still be worth refining."
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
    print("\n" + "=" * 60)
    print("  WELCOME TO THE AI RELATIONSHIP PROFILE BUILDER")
    print("=" * 60)
    print(
        "\nThis tool builds a detailed profile of your ideal connection "
        "through a guided conversation. Your responses are analyzed using "
        "three established psychological frameworks:\n\n"
        "  - MBTI personality theory\n"
        "  - Attachment style theory\n"
        "  - Love language framework\n\n"
    )
    print("Before we begin — what kind of relationship are you hoping to explore?")
    print("(e.g., romantic partner, close friendship, professional mentor, etc.)\n")
    relationship_type = input("You: ").strip()
    print()

    print("\n===== Getting to Know You =====")
    user_preferences = stage1_intake(relationship_type)

    print("\n===== Mapping Your Priorities =====")
    user_preferences = stage_ranking(user_preferences)

    print("\n===== A Few Clarifications =====")
    user_preferences = stage2_tension(user_preferences)

    print("\n===== Building Your Profile =====")
    profile_text = stage3_profile(user_preferences)

    print("\n===== Refine Your Profile =====")
    stage4_refinement(user_preferences, profile_text)

# -------------------------------
# START
# -------------------------------
if __name__ == "__main__":
    run_prototype()

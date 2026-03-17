# -------------------------------
# AI RELATIONSHIP PROFILE BUILDER (v2)
# -------------------------------
# 4-stage architecture:
#   Stage 1: About You — personality-informed questions, no partner preferences
#   Stage 2: The Proposition — trait map + inferred priorities, user reacts
#   Stage 3: Profile Generation — dynamic sections based on relationship type
#   Stage 4: Refinement — quick polish loop
# -------------------------------

import json

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"
TEST_MODE = False  # Set to True to run flow test without the real model

# -------------------------------
# TEST MODE — mock LLM responses and user inputs
# -------------------------------
_test_llm_idx = 0
_test_input_idx = 0

_TEST_LLM_RESPONSES = [
    # TODO: update mocks to match new flow once prompts are finalized
]

_TEST_USER_INPUTS = [
    # TODO: update mocks to match new flow once prompts are finalized
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
# HELPER: CALL LLM
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


# ===============================
# DIMENSION MENU
# ===============================
# The LLM selects 4-6 of these based on relationship type.
# Tier 1 (universal) should almost always be included.
# Tier 2/3 are pulled in based on context.

DIMENSION_MENU = """
TIER 1 — UNIVERSAL (include for almost any relationship type):
- Personality traits (e.g., warmth, humor, directness, patience)
- Communication style (e.g., expressive vs. reserved, direct vs. diplomatic)
- Core values (e.g., honesty, loyalty, ambition, kindness)
- Deal breakers (behaviors or traits that are non-negotiable)

TIER 2 — RELATIONAL (romantic partners, close friendships, mentors):
- Emotional needs (e.g., emotional availability, reassurance, independence)
- Attachment style (secure, anxious, avoidant tendencies)
- Love languages (quality time, words of affirmation, acts of service, physical touch, gifts)
- Conflict style (e.g., avoidant, confrontational, collaborative)

TIER 3 — ACTIVITY / CONTEXT-SPECIFIC (sports partners, study buddies, cofounders):
- Skill level and competitiveness
- Scheduling and reliability
- Work or play style (structured vs. improvisational)
- Growth orientation (casual vs. always improving)
"""


# ===============================
# STAGE 1: ABOUT YOU
# ===============================
# Learn about the user as a person. Questions quietly map Big Five / MBTI
# trait dimensions without feeling like a personality quiz.
# NEVER ask about partner preferences — that's Stage 2's job.
# ===============================

STAGE1_SYSTEM = """You are a warm, perceptive conversational guide helping someone understand themselves \
better so you can eventually help them find the right person for a specific kind of relationship.

RIGHT NOW your ONLY job is to learn about the USER as a person. You are NOT asking what they \
want in someone else — that comes later. You are building a portrait of who they are.

The user is looking for: {relationship_type}

Your questions should naturally surface where the user falls on these personality dimensions \
(but NEVER name these dimensions or make it feel like a quiz):
- Introversion vs. Extraversion — social energy, how they recharge, what connection looks like
- Sensing vs. Intuition — concrete vs. abstract thinker, what they find interesting
- Thinking vs. Feeling — decision-making style, how they handle conflict, how they express care
- Judging vs. Perceiving — structured vs. spontaneous, planning style, flexibility
- Openness to experience — curiosity, novelty-seeking, comfort with change

QUESTION BUDGET:
- For deep relational types (romantic partner, close friend, mentor): ask 5-6 questions
- For moderate relational types (roommate, creative collaborator): ask 4-5 questions
- For activity/context types (sports partner, study buddy): ask 3-4 questions
- Use your judgment. Lighter relationship types need fewer questions.

STRICT RULES:
- Ask ONLY ONE question per turn.
- NEVER ask what the user wants in another person, a partner, or a match. Not even indirectly.
- NEVER include examples, suggestions, or anchor words in your questions. \
Let the user answer entirely in their own words.
- Questions should feel like a thoughtful friend getting to know someone — not a therapist, \
not an interviewer, not a quiz.
- Each question should cover different ground. Do not revisit topics already addressed.
- Keep questions short. One or two sentences max.
- When you have asked enough questions (per the budget above), write a warm summary of who \
this person is. Start the summary with exactly 'SUMMARY:' and write only in paragraph form. \
The summary should capture personality, values, lifestyle, and relational tendencies — \
NO partner preferences, NO lists, NO JSON."""


def stage1_about_you(relationship_type):
    system_msg = {
        "role": "system",
        "content": STAGE1_SYSTEM.format(relationship_type=relationship_type)
    }

    messages = [
        system_msg,
        {"role": "user", "content": f"Hi — I'm looking for: {relationship_type}"}
    ]

    print("\nLet's start by getting to know you.\n")

    while True:
        ai_response = call_llm(messages, max_tokens=300)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "SUMMARY:" in ai_response:
            break

        # Safety valve: if LLM stops asking questions without a summary
        concluding_phrases = ["thank you for sharing", "based on what you've shared", "i have a good sense"]
        has_conclusion = any(p in ai_response.lower() for p in concluding_phrases)
        if "?" not in ai_response and has_conclusion:
            break

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print()

    # Extract structured user portrait (internal — never shown to user)
    user_portrait = extract_user_portrait(messages)
    return user_portrait, messages


def extract_user_portrait(conversation_messages):
    """Extract a structured portrait of the USER (not their preferences). Internal only."""
    extraction_msg = {
        "role": "system",
        "content": (
            "Based on the conversation, extract a structured portrait of who this user is. "
            "This is about THE USER — not what they want in someone else. "
            "Output ONLY valid JSON, nothing else.\n"
            "{\n"
            '  "personality_traits": [],\n'
            '  "communication_style": "",\n'
            '  "values": [],\n'
            '  "lifestyle": "",\n'
            '  "social_energy": "",\n'
            '  "thinking_style": "",\n'
            '  "decision_making": "",\n'
            '  "structure_vs_spontaneity": "",\n'
            '  "openness_to_experience": "",\n'
            '  "relationship_tendencies": "",\n'
            '  "big_five_estimates": {\n'
            '    "extraversion": "low/medium/high",\n'
            '    "openness": "low/medium/high",\n'
            '    "agreeableness": "low/medium/high",\n'
            '    "conscientiousness": "low/medium/high",\n'
            '    "neuroticism": "low/medium/high"\n'
            "  }\n"
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
        {"role": "user", "content": f"Extract the user portrait from this conversation:\n{conversation_text}"}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=600)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception as e:
        print(f"Warning: Could not parse user portrait: {e}")
        return {
            "personality_traits": [], "communication_style": "", "values": [],
            "lifestyle": "", "social_energy": "", "thinking_style": "",
            "decision_making": "", "structure_vs_spontaneity": "",
            "openness_to_experience": "", "relationship_tendencies": "",
            "big_five_estimates": {
                "extraversion": "", "openness": "", "agreeableness": "",
                "conscientiousness": "", "neuroticism": ""
            }
        }


# ===============================
# STAGE 2: THE PROPOSITION
# ===============================
# Show the user their trait map, then infer what they'd benefit from
# in their stated relationship type. Present as organized categories.
# User reacts, adjusts, confirms. Max 3 rounds.
# ===============================

STAGE2_SYSTEM = """You are a warm, insightful relationship coach. You have just learned a lot about \
the user as a person. Now your job is to do TWO things in your FIRST message:

1. TRAIT MAP — Show the user a brief, readable summary of the personality traits you picked up on. \
Frame it warmly: "Based on what you've shared, here's what I see in you..." \
Map them along these dimensions (use plain language, not jargon):
   - Social energy (introvert ↔ extravert)
   - Thinking style (concrete/practical ↔ abstract/big-picture)
   - Decision-making (head-first ↔ heart-first)
   - Structure (planner ↔ spontaneous)
   - Openness (comfort-seeking ↔ novelty-seeking)
Keep it to a short paragraph — not a list, not a quiz result. Make it feel like a friend \
reflecting back what they've noticed.

2. INFERRED PRIORITIES — Based on who this person IS (their trait map), infer what they would \
benefit from in their {relationship_type}. Do NOT just repeat what they said — make genuine \
inferences based on personality-compatibility principles.

Select 4-6 relevant dimension categories from this menu, based on the relationship type:

{dimension_menu}

For each selected category, list 2-4 ranked items with a ONE-LINE explanation of why you \
placed it there, tied to what you know about the user. Format as clear, organized categories \
with numbered items.

End by asking the user to react: reorder, add, remove, or confirm.

STRICT RULES:
- Present the trait map FIRST, then the inferred priorities.
- The categories you choose must make sense for the relationship type. Do NOT use love languages \
for a squash partner. Do NOT use competitiveness for a romantic partner (unless it came up).
- Frame everything as inference, not prescription: "I think..." / "Based on who you are..." \
not "You need..." / "You should look for..."
- NEVER give examples when asking for feedback. Let the user tell you what to change.
- Maximum 3 rounds of back-and-forth. After the user confirms (or after 3 rounds), \
output exactly 'PROPOSITION CONFIRMED' and stop."""


def stage2_proposition(user_portrait, relationship_type):
    system_msg = {
        "role": "system",
        "content": STAGE2_SYSTEM.format(
            relationship_type=relationship_type,
            dimension_menu=DIMENSION_MENU
        )
    }

    user_context = (
        f"The user is looking for: {relationship_type}\n\n"
        f"Here is the structured portrait of who they are:\n"
        f"{json.dumps(user_portrait, indent=2)}"
    )

    messages = [system_msg, {"role": "user", "content": user_context}]

    print("\nBased on what I've learned about you, let me map out what I think matters most...\n")

    round_count = 0
    while round_count < 3:
        ai_response = call_llm(messages, max_tokens=800)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "PROPOSITION CONFIRMED" in ai_response:
            break

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ").strip()
        messages.append({"role": "user", "content": user_input})
        print()
        round_count += 1

    if round_count == 3:
        print("AI: Great — I've noted your adjustments. Let's build your profile.\n")

    # Extract the final confirmed proposition as structured data (internal)
    proposition_data = extract_proposition(messages, relationship_type)
    return proposition_data


def extract_proposition(conversation_messages, relationship_type):
    """Extract the final confirmed proposition as structured JSON. Internal only."""
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_messages
        if m['role'] in ['user', 'assistant']
    ])

    extraction_msg = [
        {
            "role": "system",
            "content": (
                "Based on the proposition conversation, extract the final agreed-upon priorities "
                "into JSON format. The categories should reflect what was actually discussed — "
                "they will vary based on the relationship type. "
                "Output ONLY valid JSON.\n"
                "{\n"
                f'  "relationship_type": "{relationship_type}",\n'
                '  "user_trait_summary": "",\n'
                '  "selected_dimensions": [\n'
                '    {\n'
                '      "category": "",\n'
                '      "ranked_items": [\n'
                '        {"item": "", "reasoning": ""}\n'
                '      ]\n'
                '    }\n'
                '  ],\n'
                '  "deal_breakers": []\n'
                "}"
            )
        },
        {"role": "user", "content": conversation_text}
    ]

    response = call_llm(extraction_msg, temperature=0.1, max_tokens=600)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception as e:
        print(f"Warning: Could not parse proposition: {e}")
        return {
            "relationship_type": relationship_type,
            "user_trait_summary": "",
            "selected_dimensions": [],
            "deal_breakers": []
        }


# ===============================
# STAGE 3: PROFILE GENERATION
# ===============================
# Build a full profile. Sections are dynamic based on relationship type.
# The LLM selects which sections make sense.
# ===============================

STAGE3_SYSTEM = """You are collaboratively building a profile of the user's ideal {relationship_type} \
based on everything you know about them.

The user's trait summary: {trait_summary}

Their confirmed priorities:
{proposition_json}

SECTION SELECTION — Choose the sections that make sense for this relationship type. \
Here are your options (pick 5-8):

FOR ANY RELATIONSHIP TYPE:
- Name & Age
- Personality & Core Traits
- Communication Style
- A Typical Interaction (what spending time together looks like)
- Why This Person Fits You

FOR ROMANTIC / CLOSE EMOTIONAL RELATIONSHIPS:
- Physical Description
- Emotional Style & Love Languages
- Conflict Style
- Backstory
- A Typical Day in Their Life

FOR ACTIVITY / CONTEXT-BASED RELATIONSHIPS:
- Play Style or Work Style
- Skill Level & Approach
- Scheduling & Reliability
- Growth Orientation

STRICT RULES:
- Write in warm, engaging prose. Use a clear header for each section.
- Never show JSON to the user.
- The top-ranked priorities from the proposition should come through clearly in the profile.
- The profile must NOT include any of the user's stated deal breakers.
- Keep the profile grounded and specific — this should feel like a real person, not a wish list.
- End with a short 'Why This Person Fits You' section that ties the profile back to \
the user's personality and needs."""


def stage3_profile(proposition_data):
    relationship_type = proposition_data.get("relationship_type", "connection")
    trait_summary = proposition_data.get("user_trait_summary", "")

    system_content = STAGE3_SYSTEM.format(
        relationship_type=relationship_type,
        trait_summary=trait_summary,
        proposition_json=json.dumps(proposition_data.get("selected_dimensions", []), indent=2)
    )

    messages = [{"role": "system", "content": system_content}]

    print("AI: Before I build the profile — do you have anything specific in mind?")
    print("A name, a vibe, a detail you want included? Or should I surprise you?\n")

    user_ideas = input("You: ").strip()
    if user_ideas and user_ideas.lower() not in {"surprise me", "surprise", "no", "nope", ""}:
        messages.append({
            "role": "user",
            "content": f"The user wants to include these ideas: {user_ideas}"
        })

    messages.append({
        "role": "user",
        "content": (
            f"Generate a complete profile based on these confirmed priorities: "
            f"{json.dumps(proposition_data, indent=2)}.\n"
            f"Select appropriate sections for this relationship type. "
            f"End with a 'Why This Person Fits You' section. "
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


# ===============================
# STAGE 4: REFINEMENT
# ===============================
# Suggest tweaks, loop until user says done.
# ===============================

STAGE4_SYSTEM = """You are helping the user refine a {relationship_type} profile through natural conversation.

When the user gives feedback, update the profile and reprint it in full — same warm prose format, \
no JSON. React naturally as a collaborator: acknowledge what changed, and notice what else \
might be worth exploring.

When the user is done, close warmly.

Never include anything the user has flagged as a deal breaker."""


def stage4_refinement(proposition_data, profile_text):
    if not profile_text:
        print("\nAI: No profile to refine yet.\n")
        return

    relationship_type = proposition_data.get("relationship_type", "connection")

    # Generate initial suggestions
    suggestion_msg = [
        {
            "role": "system",
            "content": (
                "Based on the profile below, suggest 2-3 specific things the user might want "
                "to personalize or adjust. Be brief and conversational. "
                "Do NOT give generic suggestions — tie them to the specific profile content."
            )
        },
        {"role": "user", "content": f"Profile:\n{profile_text}"}
    ]
    suggestions = call_llm(suggestion_msg, max_tokens=150)

    print("What would you like to change? A few things you might consider:")
    print(suggestions if suggestions else "Any details that would make this person feel more real to you.")
    print("\nType 'done' when you're happy with it.\n")

    messages = [
        {
            "role": "system",
            "content": STAGE4_SYSTEM.format(relationship_type=relationship_type)
        },
        {"role": "user", "content": "Here is the current profile:\n\n" + profile_text},
        {"role": "assistant", "content": "I have the profile ready. What would you like to tweak?"}
    ]

    while True:
        feedback = input("You: ").strip()
        if not feedback:
            continue
        if feedback.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
            print(
                f"\nAI: Great — I hope this gives you a clear picture of the {relationship_type} "
                f"you're looking for. Good luck out there.\n"
            )
            break

        messages.append({
            "role": "user",
            "content": (
                f"{feedback}\n\n"
                "After updating the profile, reprint it in full. "
                "Briefly note what changed and suggest one or two things "
                "that might still be worth refining."
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


# ===============================
# RUN FULL PROTOTYPE
# ===============================
def run_prototype():
    print("\n" + "=" * 60)
    print("  AI RELATIONSHIP PROFILE BUILDER")
    print("=" * 60)
    print(
        "\nThis tool builds a detailed profile of your ideal connection — "
        "whether that's a romantic partner, a close friend, a squash buddy, "
        "or anything else — through a guided conversation.\n"
        "\nFirst I'll get to know you as a person. Then I'll use what I've learned "
        "to propose what I think you'd benefit from in that connection. "
        "You'll review, adjust, and we'll build the profile together.\n"
    )
    print("What kind of relationship or connection are you looking for?\n")
    relationship_type = input("You: ").strip()
    print()

    # Stage 1: About You
    print("\n" + "=" * 40)
    print("  GETTING TO KNOW YOU")
    print("=" * 40)
    user_portrait, stage1_messages = stage1_about_you(relationship_type)

    # Stage 2: The Proposition
    print("\n" + "=" * 40)
    print("  WHAT I THINK YOU NEED")
    print("=" * 40)
    proposition_data = stage2_proposition(user_portrait, relationship_type)

    # Stage 3: Profile Generation
    print("\n" + "=" * 40)
    print("  BUILDING YOUR PROFILE")
    print("=" * 40)
    profile_text = stage3_profile(proposition_data)

    # Stage 4: Refinement
    print("\n" + "=" * 40)
    print("  REFINE YOUR PROFILE")
    print("=" * 40)
    stage4_refinement(proposition_data, profile_text)


# ===============================
# START
# ===============================
if __name__ == "__main__":
    run_prototype()

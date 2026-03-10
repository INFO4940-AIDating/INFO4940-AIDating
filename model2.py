# -------------------------------
# AI DATING PROTOTYPE — IMPROVED
# -------------------------------
# Changes from v1:
#  - JSON never shown to the user; only human-readable messages are printed
#  - Expanded schema: added age_range, appearance_preferences, lifestyle_habits
#  - Stronger stage gate: ALL fields must be populated before moving to Stage 2
#  - AI summarises collected info and asks for confirmation before transitioning
#  - Removed all internal stage headers / technical labels from user-facing output
#  - Stage 3 properly extracts the generated profile text for use in Stage 4
#  - Stage 4 passes the real profile text so refinements are grounded
#  - Error-retry loops are hidden; only graceful fallback messages reach the user
# -------------------------------

import json
import re
from llama_cpp import Llama

# -------------------------------
# TESTING FLAGS
# -------------------------------
INJECT_BUG_AGE  = False  # Bug 1 — disabled (was: silently override age_range after Stage 1)
INJECT_BUG_NAME = False  # Bug 2 — disabled (was: ignore user's name suggestion in Stage 3)


# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"

print("Starting up… this may take a moment.")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False
)
print("Ready!\n")

# -------------------------------
# SCHEMA
# Full preference schema — moral AND physical traits
# -------------------------------
EMPTY_PREFERENCES = {
      # Physical / lifestyle
    "gender":                "",   # male / female / unspecified
    "age_range":             "",   # e.g. "25–35"
    "appearance_preferences": [],  # e.g. ["tall", "athletic build", "doesn't matter"]
    "lifestyle_habits":      [],   # e.g. ["non-smoker", "active", "social drinker ok"]
    # Personality / moral
    "core_values":           [],   # e.g. ["honesty", "ambition"]
    "emotional_needs":       [],   # e.g. ["emotional availability", "stability"]
    "deal_breakers":         [],   # hard no's
    "love_languages":        []    # words of affirmation, acts of service, etc.
}

ALL_FIELDS = list(EMPTY_PREFERENCES.keys())

# -------------------------------
# LLM HELPER
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
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        return None


def extract_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    return text[start:end]


def safe_parse_json(text):
    """Try to parse JSON, fix common LLM formatting issues (trailing commas)."""
    raw = extract_json_object(text)
    if not raw:
        return None
    for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            return json.loads(attempt)
        except Exception:
            pass
    return None


def has_meaningful_data(value):
    if value is None:
        return False
    if isinstance(value, list):
        cleaned = [str(v).strip().lower() for v in value if str(v).strip()]
        return any(v not in {"unknown", "n/a", "none", ""} for v in cleaned)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "n/a", "none"}
    return True


# -------------------------------
# NORMALISE
# -------------------------------
def normalize_preferences(raw):
    if not isinstance(raw, dict):
        raw = {}

    def norm_list(v):
        if isinstance(v, list):
            items = [str(x).strip() for x in v if str(x).strip()]
            return items if items else []
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    def norm_str(v):
        return str(v).strip() if isinstance(v, str) and v.strip() else ""

    return {
        "gender":                 norm_str(raw.get("gender")) or "male",
        "age_range":              norm_str(raw.get("age_range")) or "40",
        "appearance_preferences": norm_list(raw.get("appearance_preferences")),
        "lifestyle_habits":       norm_list(raw.get("lifestyle_habits")),
        "core_values":            norm_list(raw.get("core_values")),
        "emotional_needs":        norm_list(raw.get("emotional_needs")),
        "deal_breakers":          norm_list(raw.get("deal_breakers")),
        "love_languages":         norm_list(raw.get("love_languages"))
    }


def missing_fields(prefs):
    return [k for k in ALL_FIELDS if not has_meaningful_data(prefs.get(k))]


def all_fields_populated(prefs):
    return len(missing_fields(prefs)) == 0


STAGE1_SYSTEM = (
    "You are a warm, perceptive relationship coach having a real conversation — not filling out a form. "
    "Your job is to make the user feel genuinely heard and understood.\n\n"

    "TOPIC ORDER:\n"
    # "Start with asking about the user's ideal partner's gender and then ask about their ideal partner's age range. Then work through in order — do not skip ahead.\n"
    "  A) Ideal partner's physical appearance (ask how they want their partner to look like)\n"
    "  B) Ideal partner's lifestyle (fitness, smoking, drinking, diet)\n"
    "  C) Ideal partner's core values and character\n"
    "  D) How the user wants to feel in the relationship\n"
    "  E) Deal-breakers\n"
    "  F) Love languages — ask gently only if unclear from earlier answers\n\n"

    "CONVERSATION RULES:\n"
    "1. Ask one question per turn for each topic. After asking the question, include one follow-up question to better understand the user based on what the user inputted. Please look at FOLLOW-UP RULES to fully understand.\n"
    "2. Every response MUST contain both an acknowledgement AND a question — never send one without the other.\n"
    "   Acknowledge what they said with one warm, specific phrase, then immediately ask your next question.\n"
    "3. Never summarise or repeat back what they said.\n"
    "4. Never suggest examples or give binary choices — ask open questions only.\n"
    "   Wrong: 'Do you want light and playful, or deep and intense?'\n"
    "   Right: 'How would you want to feel day to day in the relationship?'\n"
    "5. Never show lists, summaries, or JSON mid-conversation.\n"
    "6. Keep tone warm, curious, and non-judgmental. Light humour when it fits naturally.\n\n"

    "FOLLOW-UP RULES:\n"
    "Topics are either factual, non-factual, or special.\n\n"
    "- FACTUAL (gender, age range, appearance):\n"
    "  Accept the answer and move on immediately. No follow-up.\n\n"
    "- NON-FACTUAL (lifestyle, values, feelings, deal-breakers, love languages):\n"
    "  The follow-up must only reference words or ideas the user actually used — never introduce new concepts.\n"
    "  Once the user answers the follow-up, the topic is CLOSED. Move to the next topic immediately.\n"
    "  Do not reflect, bridge, or summarise before moving on — just move on.\n"
    "  Even if the answer is vague or short, treat it as complete and closed.\n\n"

    "WRAPPING UP — only when all 9 fields have at least one value from the user's own words:\n"
    "  Accept whatever the user gave. Do not re-open a topic to get a better answer.\n"
    "  Output ONLY this raw JSON:\n"
    "PREFERENCES_FINAL: { "
    "\"core_values\": [], \"emotional_needs\": [], \"deal_breakers\": [], "
    "\"love_languages\": [], \"gender\": \"\", "
    "\"age_range\": \"\", \"appearance_preferences\": [], \"lifestyle_habits\": [] "
    "}"
)

def stage1_intake():
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": "Hi, I'd like your help figuring out what I want in a partner."}
    ]

    preference_json = None
    turn_count      = 0
    max_turns       = 20
    force_wrap      = False
    retry_count     = 0
    max_retries     = 3

    while True:
        ai_response = call_llm(messages, max_tokens=700)

        if not ai_response:
            print("\nAI: (I seem to have lost my train of thought — could you repeat that?)\n")
            user_input = input("You: ").strip()
            if user_input:
                messages.append({"role": "user", "content": user_input})
            continue

        # ── Final JSON signal ──
        if "PREFERENCES_FINAL:" in ai_response:
            raw_json_part = ai_response.split("PREFERENCES_FINAL:", 1)[1]
            parsed = safe_parse_json(raw_json_part)
            if parsed:
                preference_json = normalize_preferences(parsed)
                missing = missing_fields(preference_json)
                if missing and not force_wrap:
                    messages.append({"role": "assistant", "content": ai_response})
                    messages.append({
                        "role": "system",
                        "content": (
                            f"The following fields are still empty: {', '.join(missing)}. "
                            "Do NOT output JSON yet. Ask one natural question to fill one missing field."
                        )
                    })
                    continue

                break
            else:
                retry_count += 1
                if retry_count > max_retries:
                    preference_json = normalize_preferences({})
                    break
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({
                    "role": "system",
                    "content": (
                        "Your JSON was malformed. Output ONLY 'PREFERENCES_FINAL:' followed by "
                        "a valid JSON block. No other text. Double-quoted keys, no trailing commas."
                    )
                })
                continue

        # ── Leaked JSON guard ──
        if "{" in ai_response and "core_values" in ai_response:
            messages.append({"role": "assistant", "content": ai_response})
            messages.append({
                "role": "system",
                "content": (
                    "Do not output any JSON yet — not all fields are complete. "
                    "Continue the conversation and ask one more question."
                )
            })
            continue

        # ── Normal turn ──
        print(f"\nAI: {ai_response.strip()}\n")
        messages.append({"role": "assistant", "content": ai_response})

        turn_count += 1
        if turn_count >= max_turns and not force_wrap:
            force_wrap = True
            messages.append({
                "role": "system",
                "content": (
                    "You've gathered enough information. Do a final check for contradictions "
                    "or vague fields, resolve in one question if needed, then output "
                    "PREFERENCES_FINAL JSON immediately — no summary, no confirmation step."
                )
            })

        user_input = input("You: ").strip()
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})

    if preference_json is None:
        preference_json = normalize_preferences({})

    return preference_json


# ---------------------------------------------------------------
# STAGE 3 — PROFILE GENERATION
# ---------------------------------------------------------------
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
    system_prompt = STAGE3_SYSTEM
    messages = [{"role": "system", "content": system_prompt}]

    print("\nAI: Before I build anything — do you have any ideas about this person?")
    print("A name, a job, a vibe? Or would you like me to surprise you?\n")

    user_ideas = input("You: ").strip()
    if user_ideas:
        messages.append({
            "role": "user",
            "content": f"Some ideas the user mentioned: {user_ideas}"
        })

    # Exclude age_range — the model should invent its own age here.
    # Age is only applied if the user explicitly states one during Stage 4.
    prefs_without_age = {k: v for k, v in preference_json.items() if k != "age_range"}

    messages.append({
        "role": "user",
        "content": (
            f"Generate a complete partner profile based on these preferences: "
            f"{json.dumps(prefs_without_age)}. "
            f"IMPORTANT: Do not use any age or age range from the preferences or conversation. "
            f"Invent a completely new age for this person yourself that feels natural. "
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


# ---------------------------------------------------------------
# STAGE 4 — COLLABORATIVE REFINEMENT
# ---------------------------------------------------------------
STAGE4_SYSTEM = (
    "You are helping the user refine a fictional partner profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm, prose format, no JSON, no headers. "
    "After each update, respond naturally as a collaborator would: react to what changed, "
    "and keep the conversation moving by noticing what might still be worth exploring. "
    "Don't follow a script — let the conversation guide what to suggest next. "
    "AGE RULE: Do not apply any age or age range to the profile unless the user explicitly states one in their feedback "
    "(e.g. 'I want him to be 22' or 'make him 25'). Never infer or assume an age from context. "
    "When the user feels done, close warmly and naturally."
)

def stage4_refinement(preference_json, profile_text):
    if not profile_text:
        print("\nAI: No profile to refine — let's go back and generate one first.\n")
        return

    print("\nWhat would you like to change? (e.g. 'make her more adventurous', 'change the name to Sofia')")
    print("Type 'done' when you're happy with the profile.\n")

    messages = [
        {"role": "system", "content": STAGE4_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here is the current partner profile:\n\n"
                + profile_text
            )
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
                  "Good luck — you deserve someone great. 💛\n")
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
            print("\nAI: (I had a hiccup — could you repeat that?)\n")
            continue

        print(f"\n{'─' * 60}")
        print(ai_response.strip())
        print(f"{'─' * 60}\n")

        messages.append({"role": "assistant", "content": ai_response})


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def run_prototype():
    print("=" * 60)
    print("  Welcome — let's find out what you're looking for in a partner.")
    print("=" * 60)
    print()

    # Stage 1 — Intake
    user_preferences = stage1_intake()

    print("\nAI: Okay, I think I've got a good picture of you now. Give me a moment...\n")

    # Stage 3 — Profile generation
    profile_text = stage3_profile(user_preferences)

    # Stage 4 — Refinement loop
    stage4_refinement(user_preferences, profile_text)


run_prototype()

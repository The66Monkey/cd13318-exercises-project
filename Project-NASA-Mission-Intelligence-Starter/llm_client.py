from typing import Dict, List
from openai import OpenAI

# Connect to local KoboldCpp OpenAI-compatible API
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:5001/v1"
)

def generate_response(
    openai_key: str,                 # ignored (kept for compatibility)
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "local-model"       # ignored, KoboldCpp chooses model
) -> str:
    """Generate response using local KoboldCpp with context"""

    # System prompt for NASA RAG
    system_prompt = (
        "You are a NASA mission operations expert. "
        "Use the provided context to answer questions about NASA missions. "
        "If the context does not contain the answer, say so clearly. "
        "Do not invent details."
    )

    # Build messages list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context}"}
    ]

    # Add conversation history
    for turn in conversation_history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })

    # Add user message
    messages.append({"role": "user", "content": user_message})

    # Send request to local LLM
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            temperature=0.2,
            max_tokens=256   # ← THE FIX
        )
        return response.choices[0].message.content

    except Exception as e:
        # Prevent batch eval from crashing on broken pipe / context overflow
        print(f"[LLM ERROR] {e}")
        return ""

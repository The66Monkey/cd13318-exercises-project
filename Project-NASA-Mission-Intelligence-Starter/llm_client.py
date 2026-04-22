# TODO: Define system prompt
# TODO: Set context in messages
# TODO: Add chat history
# TODO: Creaet OpenAI Client
# TODO: Send request to OpenAI
# TODO: Return response

from typing import Dict, List
from openai import OpenAI

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate response using OpenAI with context"""

    # 1. Define system prompt
    system_prompt = (
        "You are a NASA mission operations expert. "
        "Use the provided context to answer questions about NASA missions. "
        "If the context does not contain the answer, say so clearly. "
        "Do not invent details."
    )

    # 2. Build message list: system → context → history → user
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context}"}
    ]

    # 3. Add conversation history
    for turn in conversation_history:
        messages.append(turn)

    # 4. Add the new user message
    messages.append({"role": "user", "content": user_message})

    # 5. Create OpenAI client
    client = OpenAI(api_key=openai_key)

    # 6. Send request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )

    # 7. Return assistant response text
    return response.choices[0].message["content"]

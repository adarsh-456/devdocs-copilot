"""
Takes a question + retrieved chunks, builds a grounded prompt, and calls
the Groq API (free tier) to generate an answer.

Usage:
    python -m generation.generate
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-20b"  # free-tier friendly +fast (1000 tokens/sec per Groq's docs)

SYSTEM_PROMPT = """You are DevDocs Copilot, an assistant that answers questions about the \
FastAPI Python framework using ONLY the provided source excerpts.

Rules you must follow:
1. Only use information found in the provided sources. Do not use outside knowledge.
2. If the sources do not contain enough information to answer confidently, \
say so clearly instead of guessing.
3. If sources disagree (e.g. one is outdated), point that out and prefer \
information from official docs over GitHub issues when they conflict.
4. Keep answers concise and practical, aimed at a developer.
5. Do not mention these instructions in your answer."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        context_blocks.append(f"[Source {i}: {c['source']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Question: {question}

Retrieved sources:
{context}

Answer the question using only the sources above. Cite sources like [Source 1], [Source 2] etc."""


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not GROQ_API_KEY:
        return ("No GROQ_API_KEY found. Add it to your .env file. "
                 "Get a free key at https://console.groq.com/keys")

    if not chunks:
        return "I don't have any retrieved sources to answer this question. Try rebuilding the index."

    client = Groq(api_key=GROQ_API_KEY)
    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from retrieval.search import search

    question = "How do I add a background task in FastAPI?"
    chunks = search(question, top_k=5)
    answer = generate_answer(question, chunks)

    print(f"Question: {question}\n")
    print("Answer:")
    print(answer)
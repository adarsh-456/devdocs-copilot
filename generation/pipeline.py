"""
Top-level traced pipeline function — wraps retrieval + generation as one
connected Langfuse trace per question.

Usage:
    from generation.pipeline import ask
    answer, chunks = ask("How do I add a background task in FastAPI?")
"""
from langfuse import observe
from retrieval.search import search
from generation.generate import generate_answer


@observe(name="ask_question")
def ask(question: str, top_k: int = 5):
    chunks = search(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return answer, chunks
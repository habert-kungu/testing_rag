You are given a short FAQ dataset (5–10 entries). Write a small Python (or Node.js) script that:
Creates embeddings for the FAQs.
Stores them in an in-memory vector store (e.g., FAISS, or even cosine similarity with numpy).
Accepts a natural language query (e.g., “What is the leave policy?”).
Retrieves the most relevant FAQ.
Returns a combined response: retrieved FAQ + a concise, LLM-like answer (mocked if no API access).

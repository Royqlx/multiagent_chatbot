"""
GeneratorAgent
──────────────
Generates answers using Gemini 2.5 Flash based on retrieved context.
"""

from google import genai

from config import GOOGLE_API_KEY, LLM_MODEL


class GeneratorAgent:
    """Generates answers using Gemini 2.5 Flash with multi-turn history support."""

    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def generate(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """
        Parameters
        ----------
        query   : current user question
        chunks  : retrieved chunks from RetrieverAgent
        history : list of {role, content} dicts
                  role must be 'user' or 'model'
        """

        history = history or []

        context = "\n\n---\n\n".join(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in chunks
        )

        system_prompt = f"""
You are an expert assistant answering questions from AIS documents.

Instructions:
- Use ONLY the provided context.
- If the answer is not found in the context, clearly state that.
- Cite the source document when possible.
- Be concise and accurate.

Context:
{context}
"""

        contents = []

        for turn in history:
            contents.append({
                "role": turn["role"],
                "parts": [{"text": turn["content"]}],
            })

        contents.append({
            "role": "user",
            "parts": [{
                "text": f"{system_prompt}\n\nQuestion:\n{query}"
            }],
        })

        response = self.client.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
        )

        return response.text if response.text else "No response generated." 
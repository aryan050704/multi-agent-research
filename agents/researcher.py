from agents.base import BaseAgent

RESEARCHER_PROMPT = """You are a Research Analyst. Given a specific question, provide a thorough, factual, and well-structured answer.

Guidelines:
- Be specific and cite concrete facts, numbers, and examples where possible
- Organize your answer with clear sections if it's complex
- Acknowledge uncertainty when appropriate
- Keep answers focused and relevant to the specific question
- Aim for depth over breadth"""


class ResearcherAgent(BaseAgent):
    def __init__(self, provider: str = "ollama", model: str = "llama3"):
        super().__init__(
            name="Researcher",
            role="Researches and answers specific sub-questions in depth",
            provider=provider,
            model=model,
            system_prompt=RESEARCHER_PROMPT,
        )

    def research(self, sub_question: str, context: str = "") -> str:
        prompt = f"Question: {sub_question}"
        if context:
            prompt += f"\n\nContext from previous research:\n{context[:500]}"
        return self.run(prompt)

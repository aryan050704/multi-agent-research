from agents.base import BaseAgent

SUMMARIZER_PROMPT = """You are a Research Synthesizer. Your job is to take multiple research findings and synthesize them into a single, coherent, well-structured final answer.

Guidelines:
- Integrate all findings into a unified narrative
- Remove redundancy
- Highlight the most important insights
- Structure with: Executive Summary, Key Findings, Detailed Analysis, Conclusion
- Use markdown formatting for readability"""


class SummarizerAgent(BaseAgent):
    def __init__(self, provider: str = "ollama", model: str = "llama3"):
        super().__init__(
            name="Summarizer",
            role="Synthesizes all research into a final coherent answer",
            provider=provider,
            model=model,
            system_prompt=SUMMARIZER_PROMPT,
        )

    def synthesize(self, original_query: str, sub_questions: list[str], findings: list[str]) -> str:
        combined = "\n\n".join([
            f"Sub-question {i+1}: {q}\nFinding: {a}"
            for i, (q, a) in enumerate(zip(sub_questions, findings))
        ])
        prompt = f"""Original research question: {original_query}

Research findings:
{combined}

Please synthesize these findings into a comprehensive final answer."""
        return self.run(prompt)

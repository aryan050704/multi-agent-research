from agents.base import BaseAgent

PLANNER_PROMPT = """You are a Research Planner. Your job is to break down a complex research question into 3-5 clear, specific sub-questions that together will fully answer the original question.

Output format (strictly follow this):
PLAN:
1. [specific sub-question 1]
2. [specific sub-question 2]
3. [specific sub-question 3]
...

Be precise and make each sub-question independently answerable. Do not answer the questions — only plan them."""


class PlannerAgent(BaseAgent):
    def __init__(self, provider: str = "ollama", model: str = "llama3"):
        super().__init__(
            name="Planner",
            role="Breaks research questions into structured sub-questions",
            provider=provider,
            model=model,
            system_prompt=PLANNER_PROMPT,
        )

    def plan(self, query: str) -> list[str]:
        response = self.run(f"Research question: {query}")
        lines = response.split("\n")
        sub_questions = []
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                q = line.split(".", 1)[-1].strip()
                if q:
                    sub_questions.append(q)
        if not sub_questions:
            sub_questions = [query]
        return sub_questions

"""
Orchestrator: coordinates Planner → Researcher(s) → Summarizer pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.summarizer import SummarizerAgent
from agents.base import Message


@dataclass
class ResearchSession:
    query: str
    sub_questions: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    final_answer: str = ""
    log: list[dict] = field(default_factory=list)

    def add_log(self, agent: str, action: str, content: str):
        self.log.append({"agent": agent, "action": action, "content": content})


class Orchestrator:
    def __init__(self, provider: str = "ollama", model: str = "llama3"):
        self.provider = provider
        self.model = model
        self.planner = PlannerAgent(provider, model)
        self.researcher = ResearcherAgent(provider, model)
        self.summarizer = SummarizerAgent(provider, model)

    def run(self, query: str, on_step=None) -> ResearchSession:
        session = ResearchSession(query=query)

        # Step 1: Plan
        if on_step:
            on_step("Planner", "Planning research approach...")
        sub_questions = self.planner.plan(query)
        session.sub_questions = sub_questions
        session.add_log("Planner", "plan", f"Generated {len(sub_questions)} sub-questions")

        # Step 2: Research each sub-question
        context = ""
        for i, sq in enumerate(sub_questions):
            if on_step:
                on_step("Researcher", f"Researching sub-question {i+1}/{len(sub_questions)}: {sq[:60]}...")
            finding = self.researcher.research(sq, context=context)
            session.findings.append(finding)
            session.add_log("Researcher", f"research_{i+1}", finding)
            context = finding[:300]

        # Step 3: Synthesize
        if on_step:
            on_step("Summarizer", "Synthesizing all findings into final answer...")
        final = self.summarizer.synthesize(query, sub_questions, session.findings)
        session.final_answer = final
        session.add_log("Summarizer", "synthesize", final)

        return session

    def reset(self):
        self.planner.reset()
        self.researcher.reset()
        self.summarizer.reset()

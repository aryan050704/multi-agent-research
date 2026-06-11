"""
Base agent class. Supports Ollama (local) and OpenAI backends.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str       # "system", "user", "assistant"
    content: str
    agent: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class BaseAgent:
    def __init__(self, name: str, role: str, provider: str = "ollama", model: str = "llama3", system_prompt: str = ""):
        self.name = name
        self.role = role
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.memory: list[Message] = []

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "demo":
            return self._demo_response(prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, prompt: str) -> str:
        import requests
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for m in self.memory[-6:]:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post("http://localhost:11434/api/chat",
            json={"model": self.model, "messages": messages, "stream": False}, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for m in self.memory[-6:]:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content

    def _demo_response(self, prompt: str) -> str:
        return f"[{self.name} — demo mode] Processed: '{prompt[:80]}...'"

    def run(self, input_text: str) -> str:
        response = self._call_llm(input_text)
        self.memory.append(Message(role="user", content=input_text, agent=self.name))
        self.memory.append(Message(role="assistant", content=response, agent=self.name))
        return response

    def reset(self):
        self.memory = []

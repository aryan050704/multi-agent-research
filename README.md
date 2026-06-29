# Multi-Agent Research Assistant

A small multi-agent pipeline I built to play around with agent orchestration. It splits a research question into sub-questions (Planner), answers each one (Researcher), and then merges everything into one final answer (Summarizer).

## How it works

```
User Query
    │
    ▼
PLANNER      -> breaks the question into 3-5 sub-questions
    │
    ▼
RESEARCHER   -> answers each sub-question, carries forward context from the previous one
    │
    ▼
SUMMARIZER   -> combines all the findings into one structured answer
```

Each agent is just a thin wrapper with its own system prompt — see `agents/`.

## Features
- 3 agents (Planner, Researcher, Summarizer), each with a separate role/prompt
- Context from each research step is passed into the next, so answers build on each other
- Works with Ollama (local), OpenAI, or a no-API "demo" mode for testing the UI
- Streamlit UI with live status updates while the agents are running, plus tabs for the final answer / breakdown / full log

## Running it

Demo mode (no LLM needed, just to see the flow):
```bash
pip install -r requirements.txt
streamlit run app.py
# pick "demo" in the sidebar
```

With Ollama:
```bash
ollama pull llama3
ollama serve
streamlit run app.py
# pick "ollama" in the sidebar
```

With OpenAI:
```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
# pick "openai" in the sidebar
```

## Using it as a library

```python
from agents.orchestrator import Orchestrator

orc = Orchestrator(provider="ollama", model="llama3")
session = orc.run("What are the key challenges in deploying LLMs in production?")

print(session.sub_questions)
print(session.final_answer)
```

## Stack
Python, Streamlit, Ollama, OpenAI API (optional)

## Notes / things I'd improve
- Researcher only carries forward the last 300 chars of context — works fine for short chains but loses detail on longer ones
- No caching, so re-running the same query hits the LLM again every time
- Demo mode is just canned text, useful for testing the UI without burning API calls

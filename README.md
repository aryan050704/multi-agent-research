# Multi-Agent Research Assistant

A **Planner → Researcher → Summarizer** multi-agent pipeline that decomposes complex research questions into sub-tasks and synthesizes comprehensive answers.

## Architecture

```
User Query
    │
    ▼
┌─────────┐     Breaks query into 3-5 specific sub-questions
│ PLANNER │
└─────────┘
    │ sub-questions
    ▼
┌────────────┐   Researches each sub-question independently (with context passing)
│ RESEARCHER │
└────────────┘
    │ findings
    ▼
┌─────────────┐  Synthesizes all findings → structured final answer
│  SUMMARIZER │
└─────────────┘
```

## Features
- **3 specialized agents**: Planner, Researcher, Summarizer — each with a distinct system prompt
- **Context passing**: each researcher result informs the next query
- **Multi-provider**: Ollama (local), OpenAI, or demo mode (no API key needed)
- **Streamlit UI** with live agent status, tabbed output (final answer / breakdown / log)
- **Modular design**: swap out any agent independently

## Run Locally

### Demo mode (no LLM required)
```bash
pip install -r requirements.txt
streamlit run app.py
# Select "demo" in the sidebar
```

### With Ollama (fully local)
```bash
ollama pull llama3
ollama serve
streamlit run app.py
# Select "ollama" in the sidebar
```

### With OpenAI
```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
# Select "openai" in the sidebar
```

## Use as a Library

```python
from agents.orchestrator import Orchestrator

orc = Orchestrator(provider="ollama", model="llama3")
session = orc.run("What are the key challenges in deploying LLMs in production?")

print(session.sub_questions)    # ['What are the compute challenges?', ...]
print(session.final_answer)     # synthesized answer
```

## Tech Stack
`Python` `Streamlit` `Ollama` `OpenAI API` (optional)

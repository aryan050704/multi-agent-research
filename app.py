import streamlit as st
import time
from agents.orchestrator import Orchestrator

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")
st.markdown("A **Planner → Researcher → Summarizer** agent pipeline that breaks complex questions into sub-tasks and synthesizes a final answer.")

# --- Agent flow diagram ---
with st.expander("How it works", expanded=False):
    st.markdown("""
```
User Query
    │
    ▼
┌─────────┐     Breaks query into 3-5 specific sub-questions
│ PLANNER │ ──────────────────────────────────────────────────
└─────────┘
    │
    ▼ (sub-questions)
┌────────────┐   Researches each sub-question independently
│ RESEARCHER │ ─────────────────────────────────────────────
└────────────┘
    │
    ▼ (findings)
┌─────────────┐  Synthesizes all findings into final answer
│  SUMMARIZER │ ─────────────────────────────────────────────
└─────────────┘
    │
    ▼
Final Comprehensive Answer
```
    """)

# --- Sidebar ---
st.sidebar.header("Configuration")
provider = st.sidebar.selectbox("LLM Provider", ["demo", "ollama", "openai"])
model = st.sidebar.text_input(
    "Model",
    value="llama3" if provider == "ollama" else ("gpt-3.5-turbo" if provider == "openai" else "demo")
)

if provider == "demo":
    st.sidebar.info("Demo mode shows the agent structure without a real LLM. Switch to Ollama or OpenAI for real answers.")
elif provider == "ollama":
    st.sidebar.info("Make sure Ollama is running: `ollama serve` and model is pulled: `ollama pull llama3`")
elif provider == "openai":
    st.sidebar.info("Set your OPENAI_API_KEY environment variable before running.")

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Query ---
query = st.text_area("Research question", placeholder="e.g. What are the key challenges and solutions in deploying large language models in production?", height=80)
run_btn = st.button("Start Research", type="primary")

if run_btn and query.strip():
    orchestrator = Orchestrator(provider=provider, model=model)
    status_container = st.empty()
    agent_log = []

    def on_step(agent: str, msg: str):
        agent_log.append((agent, msg))
        with status_container.container():
            for a, m in agent_log:
                color = {"Planner": "#3498db", "Researcher": "#2ecc71", "Summarizer": "#9b59b6"}.get(a, "#888")
                st.markdown(f'<span style="color:{color}">**[{a}]**</span> {m}', unsafe_allow_html=True)

    with st.spinner("Agents working..."):
        try:
            session = orchestrator.run(query, on_step=on_step)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    status_container.empty()

    # Results tabs
    tab1, tab2, tab3 = st.tabs(["Final Answer", "Research Breakdown", "Agent Log"])

    with tab1:
        st.markdown(session.final_answer)

    with tab2:
        for i, (sq, finding) in enumerate(zip(session.sub_questions, session.findings)):
            with st.expander(f"Sub-question {i+1}: {sq}"):
                st.markdown(finding)

    with tab3:
        for entry in session.log:
            agent = entry["agent"]
            color = {"Planner": "#3498db", "Researcher": "#2ecc71", "Summarizer": "#9b59b6"}.get(agent, "#888")
            st.markdown(f'<span style="color:{color}">**[{agent}]** {entry["action"]}</span>', unsafe_allow_html=True)
            with st.expander("View output"):
                st.markdown(entry["content"])

    st.session_state.history.append({"query": query, "answer": session.final_answer})

# --- History ---
if st.session_state.history:
    with st.sidebar.expander(f"History ({len(st.session_state.history)})"):
        for item in reversed(st.session_state.history):
            st.caption(item["query"][:60] + "...")
    if st.sidebar.button("Clear history"):
        st.session_state.history = []

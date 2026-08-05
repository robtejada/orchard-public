#!/usr/bin/env python
"""
Minimal Streamlit web widget for the ORCHARD documentation assistant.

Run locally:
    streamlit run docs_bot/app.py
Deploy free on Streamlit Community Cloud or Hugging Face Spaces (set the
ANTHROPIC_API_KEY secret). Requires a built index (build_corpus.py + build_index.py).
"""
import streamlit as st

import config
from ask import answer, DISCLAIMER

st.set_page_config(page_title="ORCHARD docs assistant", page_icon="🪐")

st.title("🪐 ORCHARD documentation assistant")
st.caption(
    "Ask how to run or configure ORCHARD for your project. Answers are grounded "
    "in the public docs (README, FAQ, parameter descriptions, tutorials, examples)."
)

with st.sidebar:
    st.markdown("### Need a human?")
    st.markdown(f"- [Ask in Discussions]({config.DISCUSSIONS_URL}) (usage)")
    st.markdown(f"- [Open an Issue]({config.ISSUES_URL}) (bugs / features)")
    st.markdown("---")
    st.markdown(
        "Also see `parameter_descriptions.md`, the `tutorials/` notebooks, and "
        "`parameter_examples/` for templates."
    )

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("e.g. How do I model a sub-Neptune with an H/He envelope?")

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the ORCHARD docs..."):
            try:
                result = answer(question, history=st.session_state.history)
            except Exception as exc:  # surface config/key errors gracefully
                st.error(f"Assistant error: {exc}")
                st.stop()
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"- `{s['source']}` — {s['title']}  *(score {s['score']})*")
        st.caption(DISCLAIMER)
    st.session_state.history.append((question, result["answer"]))

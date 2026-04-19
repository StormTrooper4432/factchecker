import streamlit as st

from foodrag.factcheck import run_fact_check_llm_only
from foodrag.hardcoded_prompts import get_available_claims


st.set_page_config(page_title="Nutrition Fact Checker", page_icon="🥗", layout="centered")


def _render_source(source: dict, index: int) -> None:
    title = source.get("title") or f"Source {index}"
    url = source.get("url", "")
    quote = source.get("quote", "")

    st.markdown(f"**{index}. {title}**")
    if quote:
        st.write(quote)
    if url:
        st.markdown(f"[Open source]({url})")


st.title("Nutrition Fact Checker")
st.write("Check a nutrition or fitness claim using the API-only fact-check flow.")

example_claims = get_available_claims()
selected_example = st.selectbox(
    "Try an example claim",
    [""] + example_claims,
    index=0,
    help="Optional: choose a sample claim to autofill the text box.",
)

default_claim = selected_example if selected_example else ""
claim = st.text_area(
    "Enter your claim",
    value=default_claim,
    placeholder="Type a nutrition or fitness claim here",
    height=140,
)

check_clicked = st.button("Check Claim", type="primary", use_container_width=True)

if check_clicked:
    claim_text = claim.strip()
    if not claim_text:
        st.error("Enter a claim before checking.")
    else:
        with st.spinner("Checking claim..."):
            verdict, reasoning, sources = run_fact_check_llm_only(claim_text)

        st.subheader("Verdict")
        st.markdown(f"### {verdict.title()}")

        st.subheader("Reasoning")
        st.write(reasoning or "No reasoning available.")

        st.subheader("Sources")
        if sources:
            for index, source in enumerate(sources, start=1):
                _render_source(source, index)
        else:
            st.write("No sources were returned for this claim.")

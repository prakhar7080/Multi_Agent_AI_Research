import time
import streamlit as st

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .step-title {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def to_text(content) -> str:
    """Some models return a list of content blocks instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def run_pipeline_with_ui(topic: str) -> dict:
    """Same steps as pipeline.py, but reports progress to the Streamlit UI."""
    state = {}
    timings = {}

    # Step 1 - search agent
    with st.status("Step 1 · Search agent is searching the web...", expanded=False) as status:
        t0 = time.time()
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about : {topic}")]
        })
        state["search_results"] = to_text(search_result["messages"][-1].content)
        timings["search"] = time.time() - t0
        status.update(label=f"Step 1 · Search complete ({timings['search']:.1f}s)", state="complete")

    # Step 2 - reader agent
    with st.status("Step 2 · Reader agent is scraping top resources...", expanded=False) as status:
        t0 = time.time()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content. \n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = to_text(reader_result["messages"][-1].content)
        timings["reader"] = time.time() - t0
        status.update(label=f"Step 2 · Scraping complete ({timings['reader']:.1f}s)", state="complete")

    # Step 3 - writer chain
    with st.status("Step 3 · Writer is drafting the report...", expanded=False) as status:
        t0 = time.time()
        research_combined = (
            f"SEARCH RESEARCH : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = to_text(writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        }))
        timings["writer"] = time.time() - t0
        status.update(label=f"Step 3 · Report drafted ({timings['writer']:.1f}s)", state="complete")

    # Step 4 - critic chain
    with st.status("Step 4 · Critic is reviewing the report...", expanded=False) as status:
        t0 = time.time()
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        timings["critic"] = time.time() - t0
        status.update(label=f"Step 4 · Review complete ({timings['critic']:.1f}s)", state="complete")

    state["timings"] = timings
    return state


# ---------------------------------------------------------------- session state
if "result" not in st.session_state:
    st.session_state.result = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of (topic, result)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔎 Research Assistant")
    st.caption("A multi-agent pipeline that searches, reads, writes and critiques.")

    st.markdown("### How it works")
    st.markdown(
        """
        1. **Search agent** finds recent sources  
        2. **Reader agent** scrapes the best one  
        3. **Writer** drafts the report  
        4. **Critic** reviews the draft
        """
    )

    st.markdown("### Examples")
    examples = [
        "Latest advances in solid-state batteries",
        "Impact of GLP-1 drugs on healthcare costs",
        "State of open-source LLMs in 2026",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.topic_input = ex

    if st.session_state.history:
        st.markdown("### Past runs")
        for i, (past_topic, past_result) in enumerate(reversed(st.session_state.history)):
            if st.button(f"📄 {past_topic[:40]}", use_container_width=True, key=f"hist_{i}"):
                st.session_state.result = past_result
                st.session_state.topic = past_topic
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()


# ---------------------------------------------------------------- main area
st.title("Multi-Agent Research System")
st.write("Enter a topic and let the agents do the research for you.")

topic_input = st.text_input(
    "Research topic",
    key="topic_input",
    placeholder="e.g. Latest advances in quantum computing",
)

col_run, _ = st.columns([1, 5])
run_clicked = col_run.button("🚀 Run research", type="primary", use_container_width=True)

if run_clicked:
    topic = topic_input.strip()
    if not topic:
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.result = None
        try:
            result = run_pipeline_with_ui(topic)
            st.session_state.result = result
            st.session_state.topic = topic
            st.session_state.history.append((topic, result))
            st.success("Research complete!")
        except Exception as e:
            st.error(f"Something went wrong while running the pipeline: {e}")
            with st.expander("Error details"):
                st.exception(e)

# ---------------------------------------------------------------- results
result = st.session_state.result
if result:
    st.divider()
    st.subheader(f"Results: {st.session_state.topic}")

    timings = result.get("timings", {})
    if timings:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Search", f"{timings.get('search', 0):.1f}s")
        m2.metric("Reader", f"{timings.get('reader', 0):.1f}s")
        m3.metric("Writer", f"{timings.get('writer', 0):.1f}s")
        m4.metric("Critic", f"{timings.get('critic', 0):.1f}s")
        m5.metric("Total", f"{sum(timings.values()):.1f}s")

    tab_report, tab_feedback, tab_search, tab_scrape = st.tabs(
        ["📝 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "🌐 Scraped Content"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(result["feedback"])

    with tab_search:
        st.markdown(result["search_results"])

    with tab_scrape:
        st.markdown(result["scraped_content"])

    full_export = (
        f"# Research Report: {st.session_state.topic}\n\n{result['report']}\n\n"
        f"---\n\n## Critic Feedback\n\n{result['feedback']}\n"
    )
    st.download_button(
        "⬇️ Download report + feedback (.md)",
        data=full_export,
        file_name="research_report_full.md",
        mime="text/markdown",
    )
"""SkillLens - Streamlit interface. Run:  streamlit run app.py"""
import html
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from core import llm                                   # noqa: E402  (after load_dotenv so the key is visible)
from core.config import DATA_DIR, DOCS_DIR             # noqa: E402
from core.explain import explain_skill                 # noqa: E402
from core.normalize import Vocabulary                  # noqa: E402
from core.pipeline import analyse, build_report, to_markdown   # noqa: E402
from core.resume import Embedder                       # noqa: E402

st.set_page_config(page_title="SkillLens", page_icon="🔍", layout="wide")
st.markdown("""<style>
mark.hard{background:#bfe0ff;color:#0b1f33;padding:1px 4px;border-radius:4px}
mark.soft{background:#ffd9b8;color:#33200b;padding:1px 4px;border-radius:4px}
mark sup{font-size:.62em;margin-left:3px;font-weight:700}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the five-model ensemble (first run only)...")
def get_extractor():
    from core.extractor import SkillExtractor
    return SkillExtractor()


@st.cache_resource
def get_vocab():
    return Vocabulary()


@st.cache_resource
def get_embedder():
    return Embedder()


def load_sample():
    st.session_state["jd"] = (DATA_DIR / "samples" / "sample_jd.txt").read_text(encoding="utf-8")


def highlight(text, skills):
    out, pos = [], 0
    for s in sorted(skills, key=lambda x: x["start"]):
        if s["start"] < pos:
            continue
        out.append(html.escape(text[pos:s["start"]]))
        tag = s["category"]
        out.append(f'<mark class="{tag}">{html.escape(text[s["start"]:s["end"]])}<sup>{tag.upper()}</sup></mark>')
        pos = s["end"]
    out.append(html.escape(text[pos:]))
    return '<div style="white-space:pre-wrap;line-height:1.9">' + "".join(out) + "</div>"


STATUS_LABEL = {"detected": "✅ Detected in resume", "needs_review": "🟡 Needs review", "not_detected": "❌ Not detected in resume",
                "user_confirmed": "☑️ You confirmed"}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("SkillLens")
    st.caption("Explainable skill extraction and gap analysis for student self-assessment. Not for hiring decisions.")
    use_llm = st.toggle("Use LLM layer (normalisation, inference, roadmap)", value=True,
                        help="Needs LLM_API_KEY. Without it SkillLens falls back to the alias dictionary and rules.")
    st.write("LLM key:", "✅ configured" if llm.available() else "⚪ not set (fallback mode)")
    num_samples = st.slider("LIME samples", 40, 300, 100, 20, help="More samples = steadier explanation, slower.")
    st.info("Privacy: resumes are processed in memory and not stored. If the LLM layer is on, skill phrases and job-ad "
            "sentences (not your resume text) are sent to the LLM provider.")

st.title("🔍 SkillLens")
st.caption("Paste a job advertisement, upload your resume, and get an explainable skill profile, gap report and learning roadmap.")

# ------------------------------------------------------------------ inputs
c1, c2 = st.columns([3, 2])
with c1:
    st.text_area("Job advertisement", key="jd", height=280, placeholder="Paste the full job description here...")
    st.button("Load sample job ad", on_click=load_sample)
with c2:
    up = st.file_uploader("Resume (PDF, DOCX or TXT, max 5 MB)", type=["pdf", "docx", "txt"])
    sample_pdf = DATA_DIR / "samples" / "sample_resume.pdf"
    if sample_pdf.exists():
        st.download_button("Download a synthetic sample resume", sample_pdf.read_bytes(), "sample_resume.pdf", mime="application/pdf")
    st.caption("No resume? You still get the job skill profile.")

if st.button("Analyse", type="primary"):
    if not st.session_state.get("jd", "").strip():
        st.error("Please paste a job advertisement first.")
    else:
        try:
            with st.spinner("Running extraction, normalisation and matching..."):
                st.session_state["analysis"] = analyse(
                    st.session_state["jd"], (up.getvalue(), up.name) if up else None,
                    get_extractor(), get_vocab(), get_embedder(), use_llm)
                st.session_state["expl"] = {}
        except Exception as e:                           # never show a raw traceback to the student
            st.error(f"Analysis failed ({type(e).__name__}): {str(e)[:300]}")

a = st.session_state.get("analysis")
if not a:
    st.stop()

for w in a["warnings"]:
    st.warning(w)
if a["resume_error"]:
    st.warning(f"Resume problem: {a['resume_error']} The job-skill analysis below is still shown.")
if use_llm and llm.available() and llm.LAST_ERROR:
    st.caption(f"LLM note: last call failed ({llm.LAST_ERROR}); fallbacks were used where needed.")

explicit = a["skills"]
names = sorted({s["canonical"] for s in explicit})
with st.expander("Optional: correct the report (mark skills you already know, pick focus skills)", expanded=False):
    known = st.multiselect("I already know these (even if my resume doesn't show it)", names)
    focus = st.multiselect("Prioritise these for me", names)
rep = build_report(a, focus, known, use_llm)

tabs = st.tabs(["1 Extracted skills", "2 Explanations (LIME)", "3 Normalised names", "4 Inferred skills",
                "5 Resume evidence", "6 Gap report", "7 Roadmap", "8 Performance"])

with tabs[0]:
    st.caption(f"{len(explicit)} explicit skill mentions from {a['n_models']} models voting (≥ majority). Labels are text, not just colour.")
    st.markdown(highlight(a["text"], explicit), unsafe_allow_html=True)
    if explicit:
        st.dataframe(pd.DataFrame([{"Phrase (as written)": s["phrase"], "Type": s["category"].upper(), "Canonical": s["canonical"],
                                    "Ad says": s["level"], "Model score": s["score"], "Model agreement": s["agreement"],
                                    "Sentence": s["sentence"]} for s in explicit]), width="stretch", hide_index=True)
    st.caption("Model score is an ensemble confidence, not a calibrated probability.")

with tabs[1]:
    st.caption("LIME is a LOCAL approximation: it shows which words in this sentence push the ensemble towards 'skill'. "
               "It is diagnostic evidence, not proof of how the models reason.")
    if explicit:
        pick = st.selectbox("Choose a skill to explain", range(len(explicit)),
                            format_func=lambda i: f"{explicit[i]['phrase']}  ({explicit[i]['category']})")
        if st.button("Explain this prediction"):
            sk = explicit[pick]
            try:
                with st.spinner("Perturbing the sentence and querying the ensemble..."):
                    st.session_state["expl"][sk["id"]] = explain_skill(get_extractor(), a["sentences"][sk["sent_id"]], sk, num_samples)
            except Exception as e:
                st.error(f"Explanation unavailable ({type(e).__name__}). The extraction results are unaffected.")
        ex = st.session_state.get("expl", {}).get(explicit[pick]["id"])
        if ex:
            mx = max(abs(e["weight"]) for e in ex) or 1
            chips = []
            for e in ex:
                alpha = min(abs(e["weight"]) / mx, 1) * 0.75
                col = f"rgba(34,160,90,{alpha:.2f})" if e["weight"] >= 0 else f"rgba(220,60,60,{alpha:.2f})"
                bold = "font-weight:700;" if e["is_target"] else ""
                chips.append(f'<span style="background:{col};{bold}padding:2px 3px;border-radius:3px;margin:1px" '
                             f'title="{e["weight"]:+.3f}">{html.escape(e["word"])}</span>')
            st.markdown(" ".join(chips), unsafe_allow_html=True)
            st.caption("Green supports 'skill', red opposes it; bold = the predicted skill itself; hover for the weight.")
            top = sorted(ex, key=lambda e: -abs(e["weight"]))[:8]
            st.bar_chart(pd.DataFrame({"weight": [e["weight"] for e in top]}, index=[e["word"] for e in top]))
    else:
        st.info("No skills were extracted from this advertisement.")

with tabs[2]:
    st.caption("Original wording is always kept. Statuses: exact alias, likely equivalent, ambiguous (review), no match.")
    st.dataframe(pd.DataFrame([{"Original phrase": s["phrase"], "Canonical": s["canonical"], "Status": s["norm_status"].replace("_", " "),
                                "Note": s["norm_note"]} for s in explicit]), width="stretch", hide_index=True)

with tabs[3]:
    st.caption("Suggestions only. These are NOT stated requirements and are kept apart from the explicit extraction.")
    if a["inferred"]:
        st.dataframe(pd.DataFrame([{"Inferred skill": i["canonical"], "Type": i["category"].upper(), "Confidence": i["confidence"],
                                    "Evidence sentence": i["evidence"], "Why": i["rationale"], "Method": i["method"]}
                                   for i in a["inferred"]]), width="stretch", hide_index=True)
    else:
        st.info("Nothing was inferred beyond the explicit skills.")

with tabs[4]:
    if a["sections"]:
        with st.expander("Extracted resume text (check for parsing problems)"):
            st.text(a["resume_text"][:6000])
        rows = [{"Skill": g["canonical"], "Status": STATUS_LABEL[g["status"]], "Evidence type": e["type"], "Resume snippet": e["text"]}
                for g in rep["gaps"] for e in (g["evidence"] or [{"type": "-", "text": "-"}])]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Upload a readable resume to see evidence.")

with tabs[5]:
    s = rep["summary"]
    if s:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Explicit skills in ad", s["explicit_total"])
        k2.metric("With resume evidence", s["with_evidence"])
        k3.metric("Needs review", s["needs_review"])
        k4.metric("Not detected in resume", s["not_detected"])
        st.caption("This is textual alignment with the advertisement, not a measure of your ability. "
                   "'Not detected' means the resume text does not mention it - you may still have the skill; use the correction box above.")
        st.dataframe(pd.DataFrame([{"Skill": g["canonical"], "Type": g["category"].upper(), "Origin": g["origin"], "Level in ad": g["level"],
                                    "Status": STATUS_LABEL[g["status"]], "Priority": g["tier"], "How the score was computed": g["score_note"]}
                                   for g in rep["gaps"]]), width="stretch", hide_index=True)
        st.download_button("Download report (Markdown)", to_markdown(a, rep), "skilllens_report.md")
    else:
        st.info("Upload a readable resume to get the gap report.")

with tabs[6]:
    if rep["roadmap"]:
        st.caption("Suggestions, not guarantees. Effort figures are approximate - adjust to your experience.")
        for tier, items in rep["roadmap"].items():
            st.subheader(tier.capitalize())
            if not items:
                st.write("Nothing here.")
            for it in items:
                with st.expander(f"{it['skill']}  ·  {it['kind']}"):
                    st.write(f"**Goal:** {it['objective']}")
                    st.markdown("\n".join(f"{n}. {t}" for n, t in enumerate(it["steps"], 1)))
                    st.write(f"**Practice task:** {it['practice']}")
                    st.write(f"**Show it:** {it['demonstrate']}")
                    if it["prerequisites"]:
                        st.write("**Prerequisites:** " + ", ".join(it["prerequisites"]))
                    st.write(f"**Effort (approximate):** {it['effort']}")
                    st.caption(f"Priority basis: {it['score_note']}")
    else:
        st.info("Upload a readable resume to get a roadmap.")

with tabs[7]:
    st.caption("Stage timings for THIS run (CPU, shared host). Use them for the latency row in your report.")
    st.dataframe(pd.DataFrame([{"Stage": k, "Seconds": round(v, 2)} for k, v in a["timings"].items()]), hide_index=True)
    csv = DOCS_DIR / "results.csv"
    if csv.exists():
        st.subheader("Measured extraction results (SkillSpan test split, exact-span)")
        st.dataframe(pd.read_csv(csv, index_col=0), width="stretch")

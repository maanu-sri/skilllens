"""SkillLens - Gradio interface for Hugging Face Spaces (ZeroGPU). Same core/ pipeline as the Streamlit app."""
import os, tempfile
import gradio as gr
import pandas as pd
import spaces
from dotenv import load_dotenv
load_dotenv()
from core import llm  # noqa: E402
from core.config import DATA_DIR, DOCS_DIR  # noqa: E402
from core.explain import explain_skill  # noqa: E402
from core.extractor import SkillExtractor  # noqa: E402
from core.normalize import Vocabulary  # noqa: E402
from core.pipeline import analyse, build_report, to_markdown  # noqa: E402
from core.resume import Embedder  # noqa: E402

EXT, VOC, EMB = SkillExtractor(), Vocabulary(), Embedder()   # loaded once at start-up
STATUS = {"detected": "Detected in resume", "needs_review": "Needs review", "not_detected": "Not detected in resume", "user_confirmed": "You confirmed"}
SAMPLE_JD = DATA_DIR / "samples" / "sample_jd.txt"
SAMPLE_PDF = DATA_DIR / "samples" / "sample_resume.pdf"

def table(rows, cols): return pd.DataFrame(rows, columns=cols)

def segments(text, skills):
    out, pos = [], 0
    for s in sorted(skills, key=lambda x: x["start"]):
        if s["start"] < pos: continue
        if s["start"] > pos: out.append((text[pos:s["start"]], None))
        out.append((text[s["start"]:s["end"]], s["category"].upper())); pos = s["end"]
    if pos < len(text): out.append((text[pos:], None))
    return out

def render(a, focus, known, use_llm):
    rep, sk = build_report(a, focus, known, use_llm), a["skills"]
    warns = [f"- {w}" for w in a["warnings"]]
    if a["resume_error"]: warns.append(f"- Resume problem: {a['resume_error']} The job-skill analysis is still shown.")
    if use_llm and llm.available() and llm.LAST_ERROR: warns.append(f"- LLM note: last call failed ({llm.LAST_ERROR}); fallbacks were used.")
    s = rep["summary"]
    gap_md = (f"**{s['explicit_total']}** explicit skills in the ad, **{s['with_evidence']}** with resume evidence, **{s['needs_review']}** need review, "
              f"**{s['not_detected']}** not detected. This is text alignment with the ad, not a measure of your ability. 'Not detected' only means the resume "
              "does not mention it. Use the correction box if you have the skill.") if s else "Upload a readable resume to get the gap report."
    road = ""
    for tier, items in (rep["roadmap"] or {}).items():
        road += f"### {tier.capitalize()}\n" + ("Nothing here.\n" if not items else "")
        for it in items:
            steps = "\n".join(f"  {n}. {t}" for n, t in enumerate(it["steps"], 1))
            pre = f"\n- **Prerequisites:** {', '.join(it['prerequisites'])}" if it["prerequisites"] else ""
            road += (f"\n**{it['skill']}** ({it['kind']})\n- **Goal:** {it['objective']}\n{steps}\n- **Practice task:** {it['practice']}\n"
                     f"- **Show it:** {it['demonstrate']}{pre}\n- **Effort (approximate):** {it['effort']}\n- *Priority basis: {it['score_note']}*\n\n")
    road = road or "Upload a readable resume to get a roadmap. Suggestions only, not guarantees."
    rfile = None
    if s:
        rfile = os.path.join(tempfile.mkdtemp(), "skilllens_report.md")
        open(rfile, "w", encoding="utf-8").write(to_markdown(a, rep))
    res_csv = DOCS_DIR / "results.csv"
    return (
        "\n".join(warns) or "No warnings.",
        segments(a["text"], sk),
        table([[x["phrase"], x["category"].upper(), x["canonical"], x["level"], x["score"], x["agreement"], x["sentence"]] for x in sk],
              ["Phrase (as written)", "Type", "Canonical", "Ad says", "Model score", "Model agreement", "Sentence"]),
        table([[x["phrase"], x["canonical"], x["norm_status"].replace("_", " "), x["norm_note"]] for x in sk], ["Original phrase", "Canonical", "Status", "Note"]),
        table([[i["canonical"], i["category"].upper(), i["confidence"], i["evidence"], i["rationale"], i["method"]] for i in a["inferred"]],
              ["Inferred skill", "Type", "Confidence", "Evidence sentence", "Why", "Method"]),
        a["resume_text"][:6000] if a["sections"] else "",
        table([[g["canonical"], STATUS[g["status"]], e["type"], e["text"]] for g in rep["gaps"] for e in (g["evidence"] or [{"type": "-", "text": "-"}])],
              ["Skill", "Status", "Evidence type", "Resume snippet"]) if a["sections"] else table([], ["Skill", "Status", "Evidence type", "Resume snippet"]),
        gap_md,
        table([[g["canonical"], g["category"].upper(), g["origin"], g["level"], STATUS[g["status"]], g["tier"], g["score_note"]] for g in rep["gaps"]],
              ["Skill", "Type", "Origin", "Level in ad", "Status", "Priority", "How the score was computed"]),
        rfile, road,
        table([[k, round(v, 2)] for k, v in a["timings"].items()], ["Stage", "Seconds"]),
        pd.read_csv(res_csv, index_col=0).reset_index() if res_csv.exists() else table([], ["Metric"]),
    )

@spaces.GPU(duration=120)
def run(jd, resume, use_llm):
    if not (jd or "").strip(): raise gr.Error("Please paste a job advertisement first.")
    up = None
    if resume:
        with open(resume, "rb") as f: up = (f.read(), os.path.basename(resume))
    try: a = analyse(jd, up, EXT, VOC, EMB, use_llm)
    except Exception as e: raise gr.Error(f"Analysis failed ({type(e).__name__}): {str(e)[:300]}")
    names = sorted({s["canonical"] for s in a["skills"]})
    picks = [(f"{s['phrase']} ({s['category']})", i) for i, s in enumerate(a["skills"])]
    return (a, *render(a, [], [], use_llm), gr.update(choices=names, value=[]), gr.update(choices=names, value=[]),
            gr.update(choices=picks, value=picks[0][1] if picks else None))

def refresh(a, focus, known, use_llm):
    if not a: raise gr.Error("Run an analysis first.")
    return render(a, focus or [], known or [], use_llm)

@spaces.GPU(duration=120)
def explain(a, pick, n):
    if not a or pick is None: raise gr.Error("Run an analysis and choose a skill first.")
    sk = a["skills"][int(pick)]
    try: ex = explain_skill(EXT, a["sentences"][sk["sent_id"]], sk, int(n))
    except Exception as e: raise gr.Error(f"Explanation unavailable ({type(e).__name__}). The extraction results are unaffected.")
    top = sorted(ex, key=lambda e: -abs(e["weight"]))[:8]
    return ([(e["word"] + " ", round(e["weight"], 3)) for e in ex],
            pd.DataFrame({"word": [e["word"] for e in top], "weight": [round(e["weight"], 3) for e in top]}))

with gr.Blocks(title="SkillLens", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    state = gr.State()
    gr.Markdown("# SkillLens\nPaste a job advertisement, upload your resume, and get an explainable skill profile, gap report and learning roadmap. "
                "For student self-assessment, not for hiring decisions.")
    with gr.Row():
        with gr.Column(scale=3):
            jd = gr.Textbox(label="Job advertisement", lines=12, placeholder="Paste the full job description here...")
            if SAMPLE_JD.exists():
                gr.Button("Load sample job ad", size="sm").click(lambda: SAMPLE_JD.read_text(encoding="utf-8"), None, jd)
        with gr.Column(scale=2):
            resume = gr.File(label="Resume (PDF, DOCX or TXT, max 5 MB)", type="filepath", file_types=[".pdf", ".docx", ".txt"])
            if SAMPLE_PDF.exists(): gr.DownloadButton("Download a synthetic sample resume", value=str(SAMPLE_PDF), size="sm")
            use_llm = gr.Checkbox(value=True, label="Use LLM layer (normalisation, inference, roadmap)",
                                  info="Needs LLM_API_KEY as a Space secret. Without it, SkillLens uses the alias dictionary and rules.")
            go = gr.Button("Analyse", variant="primary")
    gr.Markdown("Privacy: resumes are processed in memory and not stored. If the LLM layer is on, skill phrases and job-ad sentences (not your resume text) go to the LLM provider.")
    warn = gr.Markdown()
    with gr.Accordion("Optional: correct the report", open=False):
        known = gr.Dropdown(multiselect=True, label="I already know these (even if my resume doesn't show it)")
        focus = gr.Dropdown(multiselect=True, label="Prioritise these for me")
        upd = gr.Button("Update report")
    with gr.Tabs():
        with gr.Tab("1 Extracted skills"):
            hl = gr.HighlightedText(color_map={"HARD": "#bfe0ff", "SOFT": "#ffd9b8"}, show_legend=True, combine_adjacent=False, label="Job ad (labels are text, not only colour)")
            sk = gr.Dataframe(wrap=True, label="Model score is an ensemble confidence, not a calibrated probability.")
        with gr.Tab("2 Explanations (LIME)"):
            gr.Markdown("LIME is a **local** approximation. Positive weights push the ensemble towards 'skill', negative weights push against it. It is diagnostic evidence, not proof of how the models reason.")
            pick = gr.Dropdown(label="Choose a skill to explain")
            nsamp = gr.Slider(40, 300, value=100, step=20, label="LIME samples (more = steadier, slower)")
            exb = gr.Button("Explain this prediction")
            exhl = gr.HighlightedText(label="Word weights", show_legend=True)
            exbar = gr.BarPlot(x="word", y="weight", title="Top word weights")
        with gr.Tab("3 Normalised names"): norm = gr.Dataframe(wrap=True)
        with gr.Tab("4 Inferred skills"):
            gr.Markdown("Suggestions only. These are not stated requirements and are kept apart from the explicit extraction.")
            inf = gr.Dataframe(wrap=True)
        with gr.Tab("5 Resume evidence"):
            ev = gr.Dataframe(wrap=True)
            rtext = gr.Textbox(label="Extracted resume text (check for parsing problems)", lines=6)
        with gr.Tab("6 Gap report"):
            gapmd, gap, rfile = gr.Markdown(), gr.Dataframe(wrap=True), gr.File(label="Download report (Markdown)")
        with gr.Tab("7 Roadmap"): road = gr.Markdown()
        with gr.Tab("8 Performance"):
            tim = gr.Dataframe(label="Stage timings for this run")
            res = gr.Dataframe(label="Measured extraction results (SkillSpan test split, exact-span)")
    OUT = [warn, hl, sk, norm, inf, rtext, ev, gapmd, gap, rfile, road, tim, res]
    go.click(run, [jd, resume, use_llm], [state] + OUT + [known, focus, pick])
    upd.click(refresh, [state, focus, known, use_llm], OUT)
    exb.click(explain, [state, pick, nsamp], [exhl, exbar])

demo.launch()
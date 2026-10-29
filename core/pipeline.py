"""Orchestration. analyse() runs the heavy models once; build_report() is cheap and re-runs when the user
marks skills as known / sets focus skills (FR-11)."""
import time
from . import llm
from .gap import analyse_gaps, build_profile
from .normalize import clean_key, infer_implied, normalize_skills
from .preprocess import normalize_text, segment
from .resume import ResumeError, extract_text, match_skills, sectionize
from .roadmap import build_roadmap


def analyse(jd_text, resume, extractor, vocab, embedder=None, use_llm=True):
    """resume: (bytes, filename) or None."""
    text = normalize_text(jd_text)
    if not text.strip():
        raise ValueError("The job advertisement is empty.")
    timings, warnings = {}, []
    t = time.perf_counter()
    sentences = segment(text)
    timings["preprocessing"] = time.perf_counter() - t

    t = time.perf_counter()
    ext = extractor.extract(text, sentences)
    skills, warnings = ext["skills"], warnings + ext["warnings"]
    timings["extraction (ensemble)"] = time.perf_counter() - t

    llm_on = bool(use_llm and llm.available())
    if use_llm and not llm.available():
        warnings.append("No LLM key configured: using alias dictionary and rule-based inference only.")
    t = time.perf_counter()
    normalize_skills(skills, vocab, llm_on)
    inferred = infer_implied(sentences, {clean_key(s["canonical"]) for s in skills}, vocab, llm_on)
    timings["normalisation + inference"] = time.perf_counter() - t

    resume_text = sections = resume_error = None
    t = time.perf_counter()
    if resume:
        try:
            resume_text = extract_text(*resume)
            sections = sectionize(resume_text)
        except ResumeError as e:
            resume_error = str(e)
    profile = build_profile(skills, inferred, vocab)
    matches = match_skills(profile, sections, vocab, embedder) if sections else {}
    timings["resume matching"] = time.perf_counter() - t
    timings["total (before gap/roadmap)"] = sum(timings.values())
    return {"text": text, "sentences": sentences, "skills": skills, "inferred": inferred, "profile": profile,
            "resume_text": resume_text, "sections": sections, "resume_error": resume_error, "matches": matches,
            "warnings": warnings, "timings": timings, "n_models": ext["n_models"], "llm_used": llm_on}


def build_report(a, focus=(), known=(), use_llm=True):
    if not a["sections"]:
        return {"gaps": [], "summary": None, "roadmap": None}
    gaps, summary = analyse_gaps(a["profile"], a["matches"], set(focus), set(known))
    return {"gaps": gaps, "summary": summary, "roadmap": build_roadmap(gaps, use_llm and a["llm_used"])}


def to_markdown(a, rep):
    L = ["# SkillLens report", "", "_Automated estimates for student self-assessment - not a hiring decision._", ""]
    if rep["summary"]:
        s = rep["summary"]
        L += [f"Textual alignment: {s['with_evidence']} of {s['explicit_total']} explicit skills have resume evidence "
              f"({s['needs_review']} need review, {s['not_detected']} not detected in the resume text).", ""]
    L += ["## Gap report", "", "| Skill | Category | Origin | Level in ad | Status | Priority |", "|---|---|---|---|---|---|"]
    for g in rep["gaps"]:
        L.append(f"| {g['canonical']} | {g['category']} | {g['origin']} | {g['level']} | {g['status'].replace('_', ' ')} | {g['tier']} |")
    for tier, items in (rep["roadmap"] or {}).items():
        if items:
            L += ["", f"## Roadmap - {tier}"]
            for it in items:
                L += ["", f"### {it['skill']} ({it['kind']})", f"Goal: {it['objective']}"] + [f"- {s}" for s in it["steps"]]
                L += [f"Practice: {it['practice']}", f"Show it: {it['demonstrate']}", f"Effort: {it['effort']} (approximate)"]
    return "\n".join(L)

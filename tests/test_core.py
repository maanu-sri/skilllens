"""Run with:  python -m pytest -q   (no GPU / model download needed)"""
import re, sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import LABELS, DATA_DIR
from core.extractor import decode_spans, repair, vote
from core.explain import explain_skill
from core.gap import priority
from core.normalize import Vocabulary, clean_key, infer_implied, normalize_skills
from core.pipeline import analyse, build_report, to_markdown
from core.preprocess import normalize_text, segment
from core.resume import ResumeError, alias_regex, extract_text, sectionize

VOCAB = Vocabulary()
SAMPLES = DATA_DIR / "samples"
JD = (SAMPLES / "sample_jd.txt").read_text()


def onehot(labels):
    a = np.zeros((len(labels), 5), dtype=np.float32)
    for i, l in enumerate(labels):
        a[i, LABELS.index(l)] = 1
    return a


class FakeExtractor:
    """Dictionary-based stand-in so the whole pipeline can be tested without the trained models."""
    def __init__(self):
        self.regs = [(a, alias_regex(a), c) for c, e in VOCAB.entries.items() for a in [c] + e["aliases"]]

    def extract(self, text, sentences):
        skills = []
        for s in sentences:
            for alias, reg, canon in self.regs:
                for m in reg.finditer(s["text"]):
                    a = s["starts"][0] + m.start()
                    words = [i for i, (st, en) in enumerate(zip(s["starts"], s["ends"])) if st >= a and en <= a + len(m.group(0))]
                    if words:
                        skills.append({"id": len(skills), "phrase": text[a:a + len(m.group(0))], "category": VOCAB.category(canon),
                                       "start": a, "end": a + len(m.group(0)), "sent_id": s["id"], "word_range": (words[0], words[-1]),
                                       "sentence": s["text"], "level": s["level"], "score": 0.9, "agreement": 1.0, "origin": "explicit"})
        return {"skills": skills, "warnings": [], "n_models": 5}

    def skill_prob(self, words_list, idx):
        known = {t for k in VOCAB.alias2canon for t in k.split()}
        return np.array([0.9 if w[idx].lower() in known else 0.1 for w in words_list])


def test_vote_majority_and_tiebreak():
    a, b = onehot(["B-HARD", "O"]), onehot(["O", "O"])
    assert vote([a, a, a, b, b])[0] == "B-HARD"          # 3 of 5
    assert vote([a, a, b, b, b])[0] == "O"
    two_two_one = [onehot(["B-HARD"]), onehot(["B-HARD"]), onehot(["O"]), onehot(["O"]), onehot(["B-SOFT"])]
    assert vote(two_two_one)[0] in ("B-HARD", "O")        # tie broken by mean probability, never crashes


def test_repair_and_decode():
    assert repair(["I-HARD", "I-HARD", "O", "I-SOFT"]) == ["B-HARD", "I-HARD", "O", "B-SOFT"]
    assert decode_spans(["B-HARD", "I-HARD", "O", "B-SOFT"]) == [("HARD", 0, 1), ("SOFT", 3, 3)]


def test_segment_offsets_and_levels():
    text = normalize_text(JD)
    sents = segment(text)
    assert sents
    for s in sents:
        for w, st, en in zip(s["words"], s["starts"], s["ends"]):
            assert text[st:en] == w
    levels = {s["text"]: s["level"] for s in sents}
    assert levels[next(t for t in levels if t.startswith("Strong knowledge of SQL"))] == "required"
    assert levels[next(t for t in levels if t.startswith("Exposure to Tableau"))] == "preferred"
    assert levels[next(t for t in levels if t.startswith("Write SQL queries"))] == "unspecified"


def test_normalisation_statuses():
    sk = [{"phrase": p, "sentence": p, "category": "hard"} for p in ["MS SQL", "Microsoft SQL Server", "team player",
          "Strong knowledge of Python", "Kubernates", "FooBarTool"]]
    normalize_skills(sk, VOCAB, use_llm=False)
    got = {s["phrase"]: (s["canonical"], s["norm_status"]) for s in sk}
    assert got["MS SQL"] == ("SQL Server", "exact_alias") and got["Microsoft SQL Server"][0] == "SQL Server"
    assert got["team player"][0] == "Teamwork" and got["Strong knowledge of Python"][0] == "Python"
    assert got["Kubernates"] == ("Kubernetes", "likely_equivalent")
    assert got["FooBarTool"][1] == "no_match" and sk[-1]["phrase"] == "FooBarTool"      # original preserved


def test_alias_regex_edge_cases():
    assert alias_regex("R").search("Python, R, SQL") and not alias_regex("R").search("R&D budget")
    assert alias_regex("C").search("C, C++") and not alias_regex("C").search("C++ only")
    assert alias_regex("SQL").search("sql.") and not alias_regex("SQL").search("NoSQL")


def test_resume_sections_and_errors():
    secs = sectionize((SAMPLES / "sample_resume.txt").read_text())
    kinds = {s for s, _ in secs}
    assert {"skills", "projects", "coursework", "certifications"} <= kinds
    with pytest.raises(ResumeError):
        extract_text(b"x", "cv.exe")
    with pytest.raises(ResumeError):
        extract_text(b"%PDF-1.4 broken", "cv.pdf")
    assert "Pandas" in extract_text((SAMPLES / "sample_resume.pdf").read_bytes(), "sample_resume.pdf")


def test_priority_is_visible_and_not_invented():
    unspec = {"level": "unspecified", "mentions": 1, "origin": "explicit", "canonical": "X"}
    assert priority(unspec)[1] == "longer-term"
    req = {"level": "required", "mentions": 2, "origin": "explicit", "canonical": "X"}
    s, tier, note = priority(req)
    assert (s, tier) == (5, "immediate") and "required in ad" in note


def test_inference_rule_fallback_marks_inferred(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    inf = infer_implied(segment(normalize_text(JD)), {"python"}, VOCAB, use_llm=False)
    names = {i["canonical"] for i in inf}
    assert "Communication" in names and all(i["origin"] == "inferred" for i in inf)


def test_full_pipeline_with_fake_extractor(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    res = (SAMPLES / "sample_resume.pdf").read_bytes(), "sample_resume.pdf"
    a = analyse(JD, res, FakeExtractor(), VOCAB, embedder=None, use_llm=True)
    rep = build_report(a)
    st = {g["canonical"]: g["status"] for g in rep["gaps"]}
    assert st["Python"] == "detected" and st["Pandas"] == "detected" and st["Excel"] == "detected"
    assert st["Power BI"] == "not_detected" and st["Tableau"] == "not_detected"
    assert st["Statistics"] == "needs_review"                   # only in coursework -> weak evidence
    assert rep["summary"]["explicit_total"] > 5
    assert any(i["skill"] == "Power BI" for tier in rep["roadmap"].values() for i in tier)
    again = build_report(a, focus={"Power BI"}, known={"Tableau"})
    assert {g["canonical"]: g["status"] for g in again["gaps"]}["Tableau"] == "user_confirmed"
    assert "Gap report" in to_markdown(a, rep)


def test_resume_failure_does_not_block_extraction():
    a = analyse(JD, (b"x", "scan.exe"), FakeExtractor(), VOCAB, use_llm=False)
    assert a["resume_error"] and a["skills"] and build_report(a)["gaps"] == []


def test_lime_wiring():
    fx = FakeExtractor()
    sents = segment(normalize_text(JD))
    sk = next(s for s in fx.extract(normalize_text(JD), sents)["skills"] if s["phrase"] == "Power BI")
    exp = explain_skill(fx, sents[sk["sent_id"]], sk, num_samples=60)
    assert len(exp) == len(sents[sk["sent_id"]]["words"])
    top = max(exp, key=lambda e: abs(e["weight"]))
    assert top["is_target"]                                    # the skill word itself must drive the prediction

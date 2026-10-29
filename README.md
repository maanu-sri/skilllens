---
title: SkillLens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
---

# SkillLens

Explainable hybrid Transformer-LLM framework for skill extraction and skill-gap analysis from job descriptions
(mini project, V.S.B. College of Engineering Technical Campus, AI & Data Science).

**Pipeline:** job ad -> spaCy sentence/word segmentation -> 5-model BERT-family ensemble (word-level majority vote)
-> LIME local explanations -> alias/fuzzy/LLM normalisation -> inferred skills (kept separate) -> resume parsing
-> evidence-based matching -> gap report with visible priority score -> learning roadmap.

Trained on the public SkillSpan dataset (Zhang et al., NAACL 2022, CC-BY-4.0). `knowledge` tags are mapped to HARD, `skill` tags to SOFT.
Base paper: Akkasi (2024), Natural Language Processing Journal 9, 100102.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                     # then fill in LLM_API_KEY and SKILLLENS_MODEL_REPO
python -m pytest -q                                      # 11 tests, no models needed
streamlit run app.py
```
Without `LLM_API_KEY` the app still works (alias dictionary + rule-based inference + generic roadmaps).

## Layout
`app.py` UI | `core/` pipeline modules | `data/` alias vocabulary, roadmap templates, synthetic samples |
`training/` Colab training script | `docs/` measured results (`results.csv`, `errors.json`) | `tests/`

## Limits
Extraction metrics come from SkillSpan's test split, not from your own ads. "Not detected in resume" means the resume
text does not mention the skill. Inferred skills are advisory. Not intended for hiring decisions.

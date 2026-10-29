"""Learning roadmap: curated templates -> LLM for unknown skills (validated) -> generic fallback.
Effort figures are approximate and never a promise of employability."""
import json
from . import llm
from .config import DATA_DIR

TEMPLATES = json.load(open(DATA_DIR / "roadmap_templates.json", encoding="utf-8"))
TIERS = ["immediate", "short-term", "longer-term"]

RM_SYSTEM = ("Write a beginner-friendly learning plan for each skill. Reply as JSON: "
             '{"items":[{"skill":str,"objective":str,"steps":[3-5 short strings],"practice":str,"demonstrate":str,'
             '"prerequisites":[str],"effort":str}]}. Effort must be an approximate range. No promises about jobs.')


def _generic(name, category):
    if category == "soft":
        return {"objective": f"Show concrete evidence of {name}",
                "steps": [f"Pick a real situation (class, club, project) to practise {name}", "Ask a peer or mentor for feedback",
                          "Write a short STAR-format example"], "practice": f"Use {name} deliberately in one team task this month",
                "demonstrate": "Add one resume bullet with a measurable outcome", "prerequisites": [], "effort": "ongoing"}
    return {"objective": f"Use {name} on one small real task",
            "steps": [f"Read the official getting-started guide for {name}", "Follow one end-to-end tutorial",
                      "Rebuild it on your own data"], "practice": f"Build a mini project with {name} and push it to GitHub",
            "demonstrate": "Add a resume bullet that links the repo and states what you built", "prerequisites": [], "effort": "~1-2 weeks (approximate)"}


def _evidence_item(name):
    return {"objective": f"Make your existing {name} experience visible and verifiable",
            "steps": [f"Find where you used {name} (project, course, internship)", f"Add one concrete resume bullet naming {name} and the outcome",
                      "Link evidence (repo, certificate or report)"], "practice": "Rewrite one project description to name the tools used",
            "demonstrate": "Evidence link next to the resume bullet", "prerequisites": [], "effort": "~1 hour"}


def _valid(it, name):
    return (isinstance(it, dict) and str(it.get("skill", "")).lower() == name.lower() and isinstance(it.get("objective"), str)
            and isinstance(it.get("steps"), list) and 2 <= len(it["steps"]) <= 6 and all(isinstance(s, str) for s in it["steps"])
            and isinstance(it.get("practice"), str) and isinstance(it.get("demonstrate"), str)
            and isinstance(it.get("prerequisites", []), list) and isinstance(it.get("effort"), str))


def build_roadmap(gaps, use_llm=True):
    todo = [g for g in gaps if g["status"] in ("not_detected", "needs_review")]
    unknown = [g for g in todo if g["status"] == "not_detected" and g["canonical"] not in TEMPLATES][:6]
    llm_items = {}
    if unknown and use_llm and llm.available():
        resp = llm.chat_json(RM_SYSTEM, json.dumps({"skills": [{"skill": g["canonical"], "category": g["category"]} for g in unknown]}))
        for it in (resp or {}).get("items", []) if isinstance(resp, dict) else []:
            for g in unknown:
                if _valid(it, g["canonical"]):
                    llm_items[g["canonical"]] = {k: it.get(k, []) if k == "prerequisites" else it[k]
                                                 for k in ("objective", "steps", "practice", "demonstrate", "prerequisites", "effort")}
    road = {t: [] for t in TIERS}
    for g in todo:
        name = g["canonical"]
        if g["status"] == "needs_review":
            plan, kind = _evidence_item(name), "verify / strengthen evidence"
        elif name in TEMPLATES:
            plan, kind = TEMPLATES[name], "curated plan"
        elif name in llm_items:
            plan, kind = llm_items[name], "LLM-drafted plan (review before use)"
        else:
            plan, kind = _generic(name, g["category"]), "generic plan"
        road[g["tier"]].append({"skill": name, "category": g["category"], "origin": g["origin"], "status": g["status"],
                                "score_note": g["score_note"], "kind": kind, **plan})
    return road

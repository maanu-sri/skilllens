"""Canonical skill names: alias dictionary -> fuzzy match -> LLM (validated) -> keep original.
The original phrase is NEVER overwritten; normalisation only adds fields. Also proposes implied (inferred) skills."""
import json
import re
from rapidfuzz import fuzz, process
from . import llm
from .config import DATA_DIR

_LEAD = re.compile(r"^(?:strong|good|excellent|solid|proven|hands on|working|basic|advanced|in depth|deep|demonstrated|expert|"
                   r"experience (?:with|in|of)|knowledge (?:of|in)|proficiency (?:in|with)|understanding of|"
                   r"familiarity with|ability to|skills? in)\s+")
_TRAIL = re.compile(r"\s+(?:skills?|experience|knowledge|expertise|abilities|ability)$")


def clean_key(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^\w\s+#.&/-]", " ", s)
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    for _ in range(2):
        s = _LEAD.sub("", s)
    return _TRAIL.sub("", s).strip()


class Vocabulary:
    def __init__(self, path=DATA_DIR / "skill_aliases.json"):
        self.entries = json.load(open(path, encoding="utf-8"))
        self.alias2canon = {}
        for canon, e in self.entries.items():
            for a in [canon] + e.get("aliases", []):
                self.alias2canon[clean_key(a)] = canon
        self._keys = list(self.alias2canon)

    def category(self, canon):
        return self.entries.get(canon, {}).get("category")

    def aliases(self, canon):
        return [canon] + self.entries.get(canon, {}).get("aliases", [])


NORM_SYSTEM = ("You map skill phrases from job ads to canonical skill names. For each item choose a name ONLY from "
               "canonical_list when the phrase denotes the SAME skill (not merely a related one); otherwise null. "
               'Reply as JSON: {"results":[{"id":int,"canonical":string|null,"confidence":"high"|"medium"|"low"}]}')


def normalize_skills(skills, vocab, use_llm=True):
    """Adds canonical, norm_status, norm_note to each skill dict (in place)."""
    pending = []
    for sk in skills:
        key = clean_key(sk["phrase"])
        canon, status = vocab.alias2canon.get(key), "exact_alias"
        if not canon and len(key) >= 5:
            m = process.extractOne(key, vocab._keys, scorer=fuzz.ratio, score_cutoff=90)
            if m:
                canon, status = vocab.alias2canon[m[0]], "likely_equivalent"
        if canon:
            sk.update(canonical=canon, norm_status=status, norm_note="")
        else:
            pending.append(sk)
    resolved = {}
    if pending and use_llm and llm.available():
        payload = {"canonical_list": sorted(vocab.entries),
                   "items": [{"id": i, "phrase": p["phrase"], "sentence": p["sentence"]} for i, p in enumerate(pending)]}
        resp = llm.chat_json(NORM_SYSTEM, json.dumps(payload))
        for r in (resp or {}).get("results", []) if isinstance(resp, dict) else []:
            if isinstance(r, dict) and isinstance(r.get("id"), int) and 0 <= r["id"] < len(pending) \
                    and r.get("canonical") in vocab.entries and r.get("confidence") in ("high", "medium"):
                resolved[r["id"]] = (r["canonical"], r["confidence"])
    for i, sk in enumerate(pending):
        if i in resolved:
            canon, conf = resolved[i]
            if conf == "high":
                sk.update(canonical=canon, norm_status="likely_equivalent", norm_note="LLM-suggested, validated against vocabulary")
            else:
                sk.update(canonical=canon, norm_status="ambiguous", norm_note="LLM unsure - please review")
        else:
            p = sk["phrase"].strip()
            sk.update(canonical=p if any(c.isupper() for c in p) else p.title(), norm_status="no_match",
                      norm_note="not in vocabulary; kept as written")
    return skills


# ---------- implied skills (always marked "inferred", never merged into explicit extraction) ----------
INFER_SYSTEM = ("Given a job ad, list up to 6 skills it IMPLIES but does not name. For each give: skill (prefer names from "
                "canonical_list), category ('hard' or 'soft'), evidence (ONE sentence copied exactly from the ad), "
                "rationale (max 20 words), confidence ('low'|'medium'|'high'). Do not repeat explicit_skills. "
                'Reply as JSON: {"inferred":[{"skill":...,"category":...,"evidence":...,"rationale":...,"confidence":...}]}')

CUES = [(r"stakeholder|non[- ]technical|business (?:users|teams)|present(?:ing)? (?:findings|results|insights)", "Communication", "soft"),
        (r"cross[- ]functional|work(?:ing)? closely with|collaborat", "Teamwork", "soft"),
        (r"deadline|multiple projects|prioriti[sz]e|fast[- ]paced", "Time Management", "soft"),
        (r"mentor|lead(?:ing)? a team|team lead", "Leadership", "soft"),
        (r"troubleshoot|debug|root cause|diagnos", "Problem Solving", "soft"),
        (r"dashboards?|kpis?\b", "Data Visualization", "hard"),
        (r"deploy(?:ing|ment)?\b|production environment", "CI/CD", "hard"),
        (r"ambiguous|learn new|evolving", "Adaptability", "soft")]


def _norm_space(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def infer_implied(sentences, explicit_keys, vocab, use_llm=True):
    jd = "\n".join(s["text"] for s in sentences)
    out, seen = [], set(explicit_keys)

    def add(skill, cat, ev, why, conf, how):
        k = clean_key(skill)
        k = clean_key(vocab.alias2canon.get(k, skill))
        if k in seen or len(out) >= 6:
            return
        seen.add(k)
        out.append({"canonical": vocab.alias2canon.get(clean_key(skill), skill), "category": cat, "origin": "inferred",
                    "evidence": ev, "rationale": why, "confidence": conf, "method": how,
                    "status": "inferred - advisory, not a stated requirement"})

    if use_llm and llm.available():
        resp = llm.chat_json(INFER_SYSTEM, json.dumps({"job_ad": jd[:6000], "canonical_list": sorted(vocab.entries),
                                                       "explicit_skills": sorted(explicit_keys)}))
        items = resp.get("inferred", []) if isinstance(resp, dict) else []
        for it in items if isinstance(items, list) else []:
            try:
                ev_ok = _norm_space(it["evidence"]) in _norm_space(jd)      # evidence must really be in the ad
                if ev_ok and it["category"] in ("hard", "soft") and it["confidence"] in ("low", "medium", "high"):
                    add(str(it["skill"]), it["category"], it["evidence"], str(it["rationale"])[:200], it["confidence"], "llm")
            except (KeyError, TypeError):
                continue
        if out:
            return out
    for pat, skill, cat in CUES:                                   # offline fallback
        for s in sentences:
            m = re.search(pat, s["text"], re.I)
            if m:
                add(skill, cat, s["text"], f"cue phrase '{m.group(0)}' appears in the evidence sentence", "low", "rule")
                break
    return out

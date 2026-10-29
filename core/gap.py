"""Job skill profile, gap statuses and a VISIBLE priority score (nothing is invented when the ad is silent)."""
from .normalize import clean_key

LEVEL_RANK = {"required": 2, "preferred": 1, "unspecified": 0}
STATUS_ORDER = {"not_detected": 0, "needs_review": 1, "detected": 2, "user_confirmed": 3}


def build_profile(skills, inferred, vocab):
    prof = {}
    for s in skills:
        k = clean_key(s["canonical"])
        it = prof.setdefault(k, {"key": k, "canonical": s["canonical"], "category": s["category"], "origin": "explicit",
                                 "mentions": 0, "level": "unspecified", "phrases": [], "sentences": []})
        it["mentions"] += 1
        if s["phrase"] not in it["phrases"]:
            it["phrases"].append(s["phrase"])
        if s["sentence"] not in it["sentences"]:
            it["sentences"].append(s["sentence"])
        if LEVEL_RANK[s["level"]] > LEVEL_RANK[it["level"]]:
            it["level"] = s["level"]
    for s in inferred:
        k = clean_key(s["canonical"])
        if k not in prof:
            prof[k] = {"key": k, "canonical": s["canonical"], "category": s["category"], "origin": "inferred", "mentions": 0,
                       "level": "unspecified", "phrases": [], "sentences": [s["evidence"]], "rationale": s["rationale"]}
    return prof


def priority(item, focus=False):
    """required +3, preferred +1, mentions (max +3), user focus +3, inferred -1. >=5 immediate, 3-4 short-term, else longer-term."""
    parts = []
    if item["level"] == "required":
        parts.append(("required in ad", 3))
    elif item["level"] == "preferred":
        parts.append(("preferred in ad", 1))
    if item["mentions"]:
        parts.append((f"{item['mentions']} mention(s)", min(item["mentions"], 3)))
    if focus:
        parts.append(("your focus skill", 3))
    if item["origin"] == "inferred":
        parts.append(("inferred (advisory)", -1))
    score = sum(v for _, v in parts)
    tier = "immediate" if score >= 5 else "short-term" if score >= 3 else "longer-term"
    note = " + ".join(f"{n} ({v:+d})" for n, v in parts) or "no priority signal in the ad"
    return score, tier, f"{note} = {score}"


def analyse_gaps(profile, matches, focus=(), known=()):
    gaps = []
    for key, it in profile.items():
        m = matches.get(key, {"status": "not_detected", "evidence": [], "method": None})
        status = "user_confirmed" if it["canonical"] in known else m["status"]
        score, tier, note = priority(it, it["canonical"] in focus)
        gaps.append({**it, "status": status, "evidence": m["evidence"], "method": m["method"],
                     "score": score, "tier": tier, "score_note": note})
    gaps.sort(key=lambda g: (STATUS_ORDER[g["status"]], -g["score"], g["canonical"]))
    exp = [g for g in gaps if g["origin"] == "explicit"]
    summary = {"explicit_total": len(exp)}
    for st in STATUS_ORDER:
        summary[st] = sum(g["status"] == st for g in exp)
    summary["with_evidence"] = summary["detected"] + summary["user_confirmed"]
    return gaps, summary

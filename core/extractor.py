"""Five-model BERT-family ensemble. Word-level majority voting (>=3 of 5), ties -> highest mean probability.
torch/transformers are imported lazily so the rest of the package (and the tests) work without them."""
from pathlib import Path
import numpy as np
from .config import LABELS, MODEL_NAMES, MODEL_REPO, MODEL_LOCAL_DIR, MAXLEN


def repair(tags):
    """An I-X that does not continue a B-X/I-X becomes B-X (documented repair rule)."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def vote(prob_list):
    """prob_list: one (n_words, 5) array per model -> repaired label list (identical to the Colab evaluation)."""
    M = len(prob_list)
    votes = np.stack([p.argmax(-1) for p in prob_list])
    mean_p = np.mean(prob_list, axis=0)
    out = []
    for w in range(votes.shape[1]):
        counts = np.bincount(votes[:, w], minlength=len(LABELS))
        tied = np.flatnonzero(counts == counts.max())
        lab = tied[0] if counts.max() > M // 2 else tied[np.argmax(mean_p[w, tied])]
        out.append(LABELS[lab])
    return repair(out)


def decode_spans(labels):
    """BIO labels -> [(type, first_word, last_word)]"""
    spans, start, typ = [], None, None
    for i, l in enumerate(list(labels) + ["O"]):
        if l == "O" or l.startswith("B-") or (l.startswith("I-") and l[2:] != typ):
            if start is not None:
                spans.append((typ, start, i - 1))
                start = typ = None
            if l != "O":
                start, typ = i, l[2:]
    return spans


def _run(torch, tok, model, words_list, bs=32):
    out = []
    with torch.inference_mode():
        for i in range(0, len(words_list), bs):
            chunk = words_list[i:i + bs]
            enc = tok(chunk, is_split_into_words=True, truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits.float(), -1).numpy()
            for j, words in enumerate(chunk):
                wp = np.zeros((len(words), len(LABELS)), dtype=np.float32)
                wp[:, 0] = 1.0                       # truncated words default to "O"
                seen = set()
                for t, w in enumerate(enc.word_ids(j)):
                    if w is not None and w not in seen:
                        wp[w] = probs[j, t]
                        seen.add(w)
                out.append(wp)
    return out


class SkillExtractor:
    def __init__(self, repo=MODEL_REPO, names=MODEL_NAMES, local_dir=MODEL_LOCAL_DIR):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.torch, self.models, self.failed = torch, {}, {}
        for n in names:
            try:
                src, kw = (str(Path(local_dir) / n), {}) if local_dir else (repo, {"subfolder": n})
                tkw = dict(kw, **({"add_prefix_space": True} if n == "roberta" else {}))
                tok = AutoTokenizer.from_pretrained(src, **tkw)
                model = AutoModelForTokenClassification.from_pretrained(src, **kw).eval()
                self.models[n] = (tok, model)
            except Exception as e:                    # one broken member must not kill the app
                self.failed[n] = f"{type(e).__name__}: {str(e)[:150]}"
        if not self.models:
            raise RuntimeError(f"No ensemble member could be loaded: {self.failed}")

    def predict_probs(self, words_list, bs=32):
        return {n: _run(self.torch, tok, m, words_list, bs) for n, (tok, m) in self.models.items()}

    def extract(self, text, sentences):
        warnings = []
        if self.failed:
            warnings.append(f"Partial ensemble: {len(self.models)} of {len(self.models) + len(self.failed)} "
                            f"models running (failed: {', '.join(self.failed)}).")
        probs = self.predict_probs([s["words"] for s in sentences])
        skills = []
        for i, s in enumerate(sentences):
            plist = [probs[n][i] for n in probs]
            labels = vote(plist)
            mean_p = np.mean(plist, axis=0)
            votes = np.stack([p.argmax(-1) for p in plist])
            for typ, a, b in decode_spans(labels):
                idx = [LABELS.index(labels[w]) for w in range(a, b + 1)]
                score = float(np.mean([mean_p[w, k] for w, k in zip(range(a, b + 1), idx)]))
                agree = float(np.mean([(votes[:, w] == k).mean() for w, k in zip(range(a, b + 1), idx)]))
                st, en = s["starts"][a], s["ends"][b]
                skills.append({"id": len(skills), "phrase": text[st:en], "category": typ.lower(),
                               "start": st, "end": en, "sent_id": s["id"], "word_range": (a, b),
                               "sentence": s["text"], "level": s["level"], "score": round(score, 3),
                               "agreement": round(agree, 2), "origin": "explicit"})
        return {"skills": skills, "warnings": warnings, "n_models": len(self.models)}

    def skill_prob(self, words_list, target_idx):
        """Ensemble P(word is part of a skill) for one word position across many perturbed sentences (for LIME)."""
        probs = self.predict_probs(words_list)
        return np.mean([[1.0 - p[target_idx, 0] for p in probs[n]] for n in probs], axis=0)

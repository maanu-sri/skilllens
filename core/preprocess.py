"""Text preparation: normalise without changing offsets, split into sentences/words, tag required/preferred context."""
import re
import spacy
from .config import MAX_WORDS_PER_CHUNK

_nlp = None
BULLET = re.compile(r"^\s*(?:[•·▪●◦\-–—*]+|\d+[.)])\s+")
PREFERRED = re.compile(r"preferred|nice[- ]to[- ]have|good[- ]to[- ]have|a plus|bonus|desirable|advantage", re.I)
REQUIRED = re.compile(r"required|requirements|must[- ]have|mandatory|qualifications|what you.?ll need|you (?:have|bring)|essential", re.I)
INLINE_REQ = re.compile(r"\b(must|required|mandatory|essential)\b", re.I)


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.blank("en")          # tokenizer + rule-based sentence splitter (no model download)
        _nlp.add_pipe("sentencizer")
    return _nlp


def normalize_text(text: str) -> str:
    """Unify line breaks/spaces. Everything downstream uses character offsets into THIS string."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\t", " ")


def _is_heading(line: str) -> bool:
    s = line.strip()
    if BULLET.match(line):
        return False
    n = len(s.split())
    return n <= 8 and (s.endswith(":") or (s.isupper() and n <= 6))


def _heading_level(heading: str):
    if PREFERRED.search(heading):
        return "preferred"
    if REQUIRED.search(heading):
        return "required"
    return None


def _sentence_level(sent: str):
    if PREFERRED.search(sent):
        return "preferred"
    if INLINE_REQ.search(sent):
        return "required"
    return None


def segment(text: str):
    """Return sentences: {id, words, starts, ends, text, level}. level in required/preferred/unspecified
    (never invented: only set when the ad itself says so, via a heading or a cue word)."""
    nlp, sentences, heading_level, pos = _get_nlp(), [], None, 0
    for line in text.split("\n"):
        line_start, pos = pos, pos + len(line) + 1
        if not line.strip():
            continue
        if _is_heading(line):
            heading_level = _heading_level(line)
            continue
        for sent in nlp(line).sents:
            toks = [t for t in sent if not t.is_space]
            if toks and BULLET.match(toks[0].text + " "):      # drop the bullet marker itself
                toks = toks[1:]
            for i in range(0, len(toks), MAX_WORDS_PER_CHUNK):
                chunk = toks[i:i + MAX_WORDS_PER_CHUNK]
                if not chunk:
                    continue
                words = [t.text for t in chunk]
                starts = [line_start + t.idx for t in chunk]
                ends = [s + len(w) for s, w in zip(starts, words)]
                stext = text[starts[0]:ends[-1]]
                sentences.append({"id": len(sentences), "words": words, "starts": starts, "ends": ends,
                                  "text": stext, "level": _sentence_level(stext) or heading_level or "unspecified"})
    return sentences

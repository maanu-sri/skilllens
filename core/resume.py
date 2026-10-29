"""Resume text extraction (PDF/DOCX/TXT), section detection and evidence-based matching."""
import io
import re
from pathlib import Path
import numpy as np
from .config import ALLOWED_EXT, EMBED_MODEL, MAX_UPLOAD_MB, RELATED_THRESHOLD
from .preprocess import normalize_text


class ResumeError(Exception):
    """Raised with a message that is safe to show to the user."""


def extract_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ResumeError(f"Unsupported file type '{ext}'. Supported: {', '.join(ALLOWED_EXT)}.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ResumeError(f"File is larger than {MAX_UPLOAD_MB} MB.")
    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception as e:
            raise ResumeError("Could not read this PDF (it may be encrypted or corrupted).") from e
        if len(text.strip()) < 40:
            raise ResumeError("No selectable text found - this looks like a scanned PDF. "
                              "Please upload a text-based PDF or a DOCX (OCR is not enabled).")
    elif ext == ".docx":
        try:
            import docx
            d = docx.Document(io.BytesIO(data))
            parts = [p.text for p in d.paragraphs]
            for t in d.tables:
                parts += [" | ".join(c.text for c in row.cells) for row in t.rows]
            text = "\n".join(parts)
        except Exception as e:
            raise ResumeError("Could not read this DOCX file.") from e
        if len(text.strip()) < 40:
            raise ResumeError("This DOCX contains almost no text.")
    else:
        text = data.decode("utf-8", errors="ignore")
        if len(text.strip()) < 40:
            raise ResumeError("This text file contains almost no text.")
    return normalize_text(text)


SECTION_KEYS = [("coursework", r"coursework|relevant courses"), ("skills", r"skills?|technologies|tech stack|competenc"),
                ("projects", r"projects?"), ("experience", r"experience|internships?|employment"),
                ("certifications", r"certifications?|certificates?|courses|training"), ("education", r"education|academics?"),
                ("summary", r"summary|objective|profile|about me")]


def _heading(line):
    s = line.strip().rstrip(":").strip()
    if not s or len(s.split()) > 4 or re.search(r"[.,;]$", s) or re.match(r"^[•·\-*–]", s):
        return None
    for name, pat in SECTION_KEYS:
        if re.search(pat, s, re.I):
            return name
    return None


def sectionize(text):
    """-> list of (section, line). 'Skills: Python, SQL' style lines are handled too."""
    out, cur = [], "other"
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        h = _heading(s)
        if h:
            cur = h
            continue
        m = re.match(r"^([A-Za-z &/]{3,30})\s*[:\-–]\s*(.+)$", s)
        if m and _heading(m.group(1)) and len(m.group(2)) > 2:
            cur = _heading(m.group(1))
            s = m.group(2)
        out.append((cur, s))
    return out


def alias_regex(alias):
    flags = 0 if len(alias) <= 2 else re.I          # short aliases (R, C) are case-sensitive
    return re.compile(r"(?<![\w+#.&\-])" + re.escape(alias) + r"(?![\w+#&])", flags)


STRONG = {"skills", "projects", "experience", "certifications", "summary"}
EVIDENCE_LABEL = {"skills": "listed in skills section", "projects": "supported by project",
                  "experience": "supported by experience/internship", "certifications": "certification / course",
                  "summary": "mentioned in summary", "education": "mentioned in education",
                  "coursework": "mentioned in coursework", "other": "mentioned elsewhere"}


class Embedder:
    """Optional semantic matcher (MiniLM). Loads lazily; if it cannot load, matching silently stays keyword-only."""
    def __init__(self):
        self.model, self.ok = None, True

    def encode(self, texts):
        if not self.ok:
            return None
        try:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EMBED_MODEL)
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            self.ok = False
            return None


def match_skills(profile, sections, vocab, embedder=None):
    """profile: {key: item}. Returns {key: {status, evidence:[{section,type,text}], method}}.
    detected = found in skills/projects/experience/certifications; needs_review = only weak or semantic evidence."""
    matches, unmatched = {}, []
    for key, it in profile.items():
        names = vocab.aliases(it["canonical"]) if it["canonical"] in vocab.entries else [it["canonical"]] + list(it.get("phrases", []))
        regs = [alias_regex(a) for a in dict.fromkeys(names)]
        ev = [{"section": sec, "type": EVIDENCE_LABEL[sec], "text": line[:220]}
              for sec, line in sections if any(r.search(line) for r in regs)]
        if ev:
            strong = any(e["section"] in STRONG for e in ev)
            matches[key] = {"status": "detected" if strong else "needs_review", "evidence": ev[:3], "method": "keyword"}
        else:
            matches[key] = {"status": "not_detected", "evidence": [], "method": None}
            unmatched.append(key)
    if embedder and unmatched:
        lines = [(sec, l) for sec, l in sections if len(l) >= 15][:400]
        if lines:
            le = embedder.encode([l for _, l in lines])
            se = embedder.encode([profile[k]["canonical"] for k in unmatched]) if le is not None else None
            if le is not None and se is not None:
                sims = se @ le.T
                for row, key in zip(sims, unmatched):
                    j = int(np.argmax(row))
                    if row[j] >= RELATED_THRESHOLD:
                        sec, line = lines[j]
                        matches[key] = {"status": "needs_review", "method": "semantic",
                                        "evidence": [{"section": sec, "type": f"possibly related (similarity {row[j]:.2f})", "text": line[:220]}]}
    return matches

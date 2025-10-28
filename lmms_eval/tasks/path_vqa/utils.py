from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from PIL import Image
from loguru import logger as eval_logger

# ---------------------------
# normalization & parsing
# ---------------------------
_ARTICLES = {"a", "an", "the"}
_NUM_MAP = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
}
_YES = {"yes","y","yeah","yep","true","1"}
_NO  = {"no","n","nope","false","0"}

def _normalize_vqa(s: str) -> str:
    """VQA-ish normalization for exact-match scoring."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()

    if s in _YES: return "yes"
    if s in _NO:  return "no"

    s = re.sub(r"[^\w\s%/.-]", " ", s)  # keep %, /, -, .
    toks = [t for t in re.split(r"\s+", s) if t]
    out  = []
    for t in toks:
        if t in _ARTICLES:
            continue
        out.append(_NUM_MAP.get(t, t))
    s2 = " ".join(out)
    s2 = re.sub(r"\s+", " ", s2).strip().strip(".")
    return s2

def _extract_final_answer(text: str) -> str:
    """Prefer the LAST 'Answer: ...'; else last non-empty line."""
    if not text:
        return ""
    m = list(re.finditer(r"answer\s*[:\-]\s*(.+)", text, re.IGNORECASE))
    if m:
        return m[-1].group(1).strip()
    for line in reversed([ln.strip() for ln in text.splitlines()]):
        if line:
            return line
    return text.strip()

# ---------------------------
# PathVQA task adapters
# ---------------------------
def pathvqa_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> str:
    """HF PathVQA provides 'image' (PIL), 'question' (str), 'answer' (str)."""
    pre = post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
        post = lmms_eval_specific_kwargs.get("post_prompt", "") or ""
    q = (doc.get("question") or "").strip()
    return f"{pre}{q}\n{post}".strip()

def pathvqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return (doc.get("answer") or "").strip()

def pathvqa_doc_to_visual(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> List[Image.Image]:
    """Images are embedded directly in the dataset."""
    im = doc.get("image")
    if im is None:
        eval_logger.warning("PathVQA: sample has no image")
        return []
    try:
        return [im.convert("RGB")]
    except Exception as e:
        eval_logger.warning(f"PathVQA: failed to convert image: {e}")
        return []

def pathvqa_process_results(
    doc: Dict[str, Any],
    results: List[str],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
):
    """Exact-match after VQA normalization. Accept 'Answer: ...' or raw."""
    pred_raw = (results[0] if results else "") or ""
    pred = _extract_final_answer(pred_raw)

    p = _normalize_vqa(pred)
    t = _normalize_vqa(pathvqa_doc_to_target(doc))
    score = 1.0 if (p == t and p != "") else 0.0

    qid = f"{doc.get('question','')[:200]}::{doc.get('answer','')[:40]}"
    return {"accuracy": {"question_id": qid, "score": score}}

def pathvqa_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """Aggregate accuracy (%) over emitted item dicts."""
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        acc = r.get("accuracy", {}).get("score", None)
        if acc is None:
            acc = r.get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1
    pct = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"PathVQA Accuracy: {pct:.2f}")
    return pct

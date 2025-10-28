from __future__ import annotations

import os
import re
import zipfile
from functools import lru_cache
from typing import Any, Dict, List, Optional
from PIL import Image
from loguru import logger as eval_logger
from huggingface_hub import snapshot_download

# ---------------------------
# Language filter (we keep only English)
# ---------------------------
LANG_KEEP = "en"

def _is_en(doc: Dict[str, Any]) -> bool:
    return str(doc.get("q_lang", "")).lower() == LANG_KEEP

# ---------------------------
# VQA-style normalization
# ---------------------------
_ARTICLES = {"a", "an", "the"}
_NUM_MAP = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
}
_YES = {"yes","y","yeah","yep","true","1"}
_NO  = {"no","n","nope","false","0"}

def _normalize_vqa(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip().lower()
    if s in _YES: return "yes"
    if s in _NO:  return "no"
    s = re.sub(r"[^\w\s%/.-]", " ", s)  # keep %, /, -, .
    toks = [t for t in re.split(r"\s+", s) if t]
    out: List[str] = []
    for t in toks:
        if t in _ARTICLES:
            continue
        out.append(_NUM_MAP.get(t, t))
    s2 = " ".join(out)
    s2 = re.sub(r"\s+", " ", s2).strip().strip(".")
    return s2

def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    m = list(re.finditer(r"answer\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE))
    if m:
        return m[-1].group(1).strip()
    for line in reversed([ln.strip() for ln in text.splitlines()]):
        if line:
            return line
    return text.strip()

# ---------------------------
# Image resolver (imgs.zip → imgs/)
# ---------------------------
def _dir_nonempty(p: str) -> bool:
    return os.path.isdir(p) and bool(os.listdir(p))

def _extract_zip_once(zp: str, out_dir: str) -> Optional[str]:
    try:
        if _dir_nonempty(out_dir):
            return out_dir
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            eval_logger.info(f"Extracting {os.path.basename(zp)} -> {out_dir} (one-time)...")
            zf.extractall(out_dir)
        return out_dir if _dir_nonempty(out_dir) else None
    except Exception as e:
        eval_logger.warning(f"Failed to extract {zp}: {e}")
        return None

@lru_cache(maxsize=8)
def _slake_image_root_for_repo(repo_id: str) -> str:
    if not repo_id:
        eval_logger.error("SLAKE resolver: empty repo_id")
        return ""
    if snapshot_download is None:
        eval_logger.error("Please install huggingface_hub: pip install -U huggingface_hub")
        return ""
    snap = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["imgs/*", "imgs.zip"],
        resume_download=True,
    )
    imgs_dir = os.path.join(snap, "imgs")
    if _dir_nonempty(imgs_dir):
        return imgs_dir
    zp = os.path.join(snap, "imgs.zip")
    if os.path.isfile(zp):
        out = _extract_zip_once(zp, imgs_dir)
        if out and _dir_nonempty(out):
            return out
    eval_logger.error(f"SLAKE resolver: imgs/ not found in snapshot {snap}")
    return ""

# ---------------------------
# Task adapters
# ---------------------------
def slake_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Return question text only for English items; otherwise return empty string (skipped)."""
    if not _is_en(doc):
        return ""
    pre = post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
        post = lmms_eval_specific_kwargs.get("post_prompt", "") or ""
    q = (doc.get("question") or "").strip()
    return f"{pre}{q}\n{post}".strip()

def slake_doc_to_target(doc: Dict[str, Any]) -> str:
    return (doc.get("answer") or "").strip()

def slake_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):
    """Resolve 'img_name' under imgs/; non-English docs will be ignored upstream."""
    repo_id = ""
    if lmms_eval_specific_kwargs:
        repo_id = lmms_eval_specific_kwargs.get("dataset_repo") or lmms_eval_specific_kwargs.get("dataset_path") or ""
    if not repo_id:
        repo_id = "BoKelvin/SLAKE"
    root = _slake_image_root_for_repo(repo_id)
    if not root:
        raise RuntimeError("SLAKE images unavailable; ensure network/HF token and disk space.")
    rel = str(doc.get("img_name") or "").strip().lstrip("/\\")
    if not rel:
        return []
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        base = os.path.basename(rel)
        for r, _d, files in os.walk(root):
            if base in files:
                path = os.path.join(r, base)
                break
    if not os.path.isfile(path):
        eval_logger.warning(f"SLAKE: image not found for {rel}")
        return []
    try:
        with Image.open(path) as im:
            return [im.convert("RGB")]
    except Exception as e:
        eval_logger.warning(f"SLAKE: failed to open {path}: {e}")
        return []

def slake_process_results(
    doc: Dict[str, Any],
    results: List[str],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
):
    """Emit accuracy only for English items; non-English return {} (ignored by aggregator)."""
    if not _is_en(doc):
        return {}

    # Raw model text (BEFORE any parsing)
    pred_raw = (results[0] if results else "") or ""

    # Parse & normalize
    parsed = _extract_final_answer(pred_raw)
    pred_norm = _normalize_vqa(parsed)
    target_norm = _normalize_vqa(slake_doc_to_target(doc))
    score = 1.0 if (pred_norm == target_norm and pred_norm != "") else 0.0
    qid = str(doc.get("qid", "")) or f"{doc.get('img_name','')}::{doc.get('question','')}"[:256]

    return {"accuracy": {"question_id": qid, "score": score}}

def slake_aggregate_results(results: List[Dict[str, Any]]) -> float:
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
    eval_logger.info(f"SLAKE Accuracy: {pct:.2f}")
    return pct
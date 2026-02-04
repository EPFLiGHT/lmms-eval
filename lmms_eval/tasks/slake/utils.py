from __future__ import annotations

import os
import re
import zipfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from loguru import logger as eval_logger
from huggingface_hub import snapshot_download
import os 
import numpy as np
from copy import deepcopy
from lmms_eval.llm_judge import Request, ServerConfig, get_server



SLAKE_JUDGE_METRICS = [
    "gpt_eval_slake_all",
    "gpt_eval_slake_binary",
    "gpt_eval_slake_nonbinary",
]
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

def _is_binary_target(t_norm: str) -> bool:
    return t_norm == "yes" or t_norm == "no"
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
    m = list(re.finditer(r"(?i)answer\W(?:is)*\W*([A-D])(?:\W|$)|(?:answer|boxed)?{\W*([A-D])\W+", text, flags=re.IGNORECASE))
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

    is_binary = (target_norm == "yes" or target_norm == "no")
    score = 1.0 if (pred_norm == target_norm and pred_norm != "") else 0.0
    qid = str(doc.get("qid", "")) or f"{doc.get('img_name','')}::{doc.get('question','')}"[:256]

    return {
        "overall_acc": {
            "question_id": qid, 
            "score": score
        },
        "binary_acc": {
            "question_id": qid, 
            "score": score,
            "is_binary" : is_binary
        }
    }


def slake_simple_process_results(
        doc: Dict[str, Any],
        results: List[Tuple[float, bool]],
        lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):

    if not _is_en(doc):
        return {}

    if len(results) > 1:
        raise ValueError(f"loglikelihood returned {len(results)} elements, please use batch size at 1 when evaluating on loglikelihood")

    return results[0]


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
    eval_logger.info(f"SLAKE Overall Accuracy: {pct:.2f}")
    return pct

def slake_aggregate_binary_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        if not r.get("is_binary", False):
            continue

        acc = r.get("accuracy", {}).get("score", None)
        if acc is None:
            acc = r.get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1

    pct = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"SLAKE Yes/No Accuracy: {pct:.2f}")
    return pct
 

def _parse_judge_score(text: str) -> int:
    if not text:
        return -1
    m = re.search(r"(?i)\bscore\b\W*([0-9]{1,3})\b", text)
    if m:
        v = int(m.group(1))
        return max(0, min(100, v))
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    m = re.search(r"\b([0-9]{1,3})\b", first)
    if m:
        v = int(m.group(1))
        return max(0, min(100, v))
    return -1

def _judge_prompt(question: str, gt: str, pred: str, is_binary: bool) -> str:
    binary_line = "This is a YES/NO question; accept only yes/no meaning." if is_binary else "This is NOT necessarily yes/no; accept short medical synonyms if they match the ground truth."
    return (
        "You are grading a medical VQA model output.\n"
        "Return a single integer SCORE from 0 to 100.\n"
        "100 = fully correct, 0 = incorrect.\n"
        "Do not add extra numbers besides the SCORE.\n"
        f"{binary_line}\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[GROUND TRUTH ANSWER]\n{gt}\n\n"
        f"[MODEL ANSWER]\n{pred}\n\n"
        "Format:\nSCORE: <integer 0-100>\n"
    )


    
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "Qwen/Qwen3-8B")
API_TYPE = os.getenv("API_TYPE", "openai")


_server_config = ServerConfig(model_name=JUDGE_MODEL_NAME, temperature=0.2, max_tokens=256)
_server = get_server(server_name=API_TYPE, config=_server_config)

def get_eval_judge(content: str, max_tokens: int, retries: int = 5) -> Tuple[str, str]:
    messages = [
        {"role": "system", "content": "You are a strict, helpful grader for medical visual question answering."},
        {"role": "user", "content": content},
    ]
    custom_config = ServerConfig(model_name=JUDGE_MODEL_NAME, temperature=0.2, max_tokens=max_tokens)
    for attempt in range(retries):
        try:
            request = Request(messages=messages, config=custom_config)
            response = _server.evaluate(request)
            out = response.content.strip() if response.content else ""
            if out != "":
                return out, response.model_used
            return "", response.model_used
        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt + 1 < retries:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


def slake_process_results_with_llm_judge(
    doc: Dict[str, Any],
    results: List[str],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not _is_en(doc):
        return {}

    base = slake_process_results(doc, results, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)

    question = (doc.get("question") or "").strip()
    gt = (doc.get("answer") or "").strip()
    pred_raw = (results[0] if results else "") or ""
    pred = _extract_final_answer(pred_raw).strip()

    t_norm = _normalize_vqa(gt)
    is_binary = _is_binary_target(t_norm)

    content = _judge_prompt(question=question, gt=gt, pred=pred, is_binary=is_binary)

    try:
        review, model_used = get_eval_judge(content, max_tokens=256)
        judge_score = _parse_judge_score(review)
    except Exception as e:
        eval_logger.error(f"SLAKE judge error for qid={base.get('overall_acc',{}).get('question_id','')} : {e}")
        review, model_used, judge_score = "", "", -1

    qid = base["overall_acc"]["question_id"]
    item = {
        "question_id": qid,
        "judge_score": judge_score,
        "judge_review": review,
        "eval_model": model_used,
        "question": question,
        "gt": gt,
        "pred": pred,
        "content": content,
        "is_binary": is_binary,
    }

    non_category_item = deepcopy(item)
    non_category_item["judge_score"] = -999

    out = {}
    out["gpt_eval_slake_all"] = item
    out["gpt_eval_slake_binary"] = item if is_binary else non_category_item
    out["gpt_eval_slake_nonbinary"] = item if not is_binary else non_category_item
    return out

def slake_llm_judge_aggregate(results: List[Dict[str, Any]]) -> Optional[float]:
    try:
        scores = []
        for r in results:
            s = r.get("judge_score", None)
            if s is None:
                continue
            if s == -999:
                continue
            if s < 0:
                continue
            scores.append(float(s))
        if not scores:
            return None
        return round(float(np.mean(scores)), 3)
    except Exception as e:
        eval_logger.info(f"Error in slake_llm_judge_aggregate: {e}")
        return None
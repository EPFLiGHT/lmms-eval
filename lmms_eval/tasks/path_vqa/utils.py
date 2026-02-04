from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from PIL import Image
from loguru import logger as eval_logger
import os 
import numpy as np
from copy import deepcopy
from lmms_eval.llm_judge import Request, ServerConfig, get_server
# ---------------------------
# normalization & parsing
# ---------------------------

NUM_SECONDS_TO_SLEEP = 5

PATHVQA_JUDGE_METRICS = [
    "gpt_eval_pathvqa_all",
    "gpt_eval_pathvqa_binary",
    "gpt_eval_pathvqa_nonbinary",
]

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

    m = list(re.finditer(r"(?i)answer\W(?:is)*\W*([A-D])(?:\W|$)|(?:answer|boxed)?{\W*([A-D])\W+", text, re.IGNORECASE))
    if m:
        return m[-1].group(1).strip()
    for line in reversed([ln.strip() for ln in text.splitlines()]):
        if line:
            return line
    return text.strip()

#### LLM as a judge
def _is_binary_target(t_norm: str) -> bool:
    return t_norm == "yes" or t_norm == "no"

def _parse_judge_score(text: str) -> int:
    if not text:
        return -1
    m = re.search(r"(?i)\bscore\b\W*([0-9]{1,3})\b", text)
    if m:
        v = int(m.group(1))
        return max(0, min(100, v))
    m = re.search(r"\b([0-9]{1,3})\b", text.strip().splitlines()[0] if text.strip() else "")
    if m:
        v = int(m.group(1))
        return max(0, min(100, v))
    return -1

JUDGE_MODEL_NAME = os.getenv("GPT_EVAL_MODEL_NAME", "Qwen/Qwen3-8B")
API_TYPE = os.getenv("API_TYPE", "openai")


_server_config = ServerConfig(
    model_name=JUDGE_MODEL_NAME,

    temperature=0.2,
    max_tokens=256,
)

_server = get_server(
    server_name=API_TYPE,
    config=_server_config,
)

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
    return "", ""


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
    is_binary = (t == "yes" or t == "no") 

    qid = f"{doc.get('question','')[:200]}::{doc.get('answer','')[:40]}"
    return {
        "overall_acc": {
            "question_id": qid, 
            "score": score
        }, 
        "binary_acc": {
            "is_binary" : is_binary,
            "question_id" : qid,
            "score" : score
        }
    }


def pathvqa_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """Aggregate accuracy (%) over emitted item dicts."""
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        acc = r.get("overall_acc", {}).get("score", None)
        if acc is None:
            acc = r.get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1
    pct = (total / count) * 100.0 if count else 0.0

    eval_logger.info(f"PathVQA Overall Accuracy: {pct:.2f}")
    return pct

def pathvqa_process_results_with_llm_judge(
    doc: Dict[str, Any],
    results: List[str],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    base = pathvqa_process_results(doc, results, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)

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
        eval_logger.error(f"PathVQA judge error for question: {question[:80]}... : {e}")
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
    out["gpt_eval_pathvqa_all"] = item
    out["gpt_eval_pathvqa_binary"] = item if is_binary else non_category_item
    out["gpt_eval_pathvqa_nonbinary"] = item if not is_binary else non_category_item
    return out

def pathvqa_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        acc = r.get("overall_acc", {}).get("score", None)
        if acc is None:
            acc = r.get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1
    pct = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"PathVQA Overall Accuracy: {pct:.2f}")
    return pct

def pathvqa_aggregate_binary_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    total = 0.0
    count = 0
    for r in results:
        if not r.get("is_binary", False):
            continue
        acc = r.get("score", None)
        if acc is None:
            continue
        total += float(acc)
        count += 1
    pct = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"PathVQA Yes/No Accuracy: {pct:.2f}")
    return pct

def pathvqa_llm_judge_aggregate(results: List[Dict[str, Any]]) -> Optional[float]:
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
        eval_logger.info(f"Error in pathvqa_llm_judge_aggregate: {e}")
        return None

def pathvqa_llm_judge_binary_aggregate(results: List[Dict[str, Any]]) -> Optional[float]:
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
        eval_logger.info(f"Error in pathvqa_llm_judge_binary_aggregate: {e}")
        return None

def pathvqa_llm_judge_nonbinary_aggregate(results: List[Dict[str, Any]]) -> Optional[float]:
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
        eval_logger.info(f"Error in pathvqa_llm_judge_nonbinary_aggregate: {e}")
        return None

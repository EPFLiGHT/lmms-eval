from __future__ import annotations

import os
import re
import zipfile
from functools import lru_cache
from typing import Any, Dict, List, Optional
from PIL import Image
from loguru import logger as eval_logger
from huggingface_hub import snapshot_download

CHOICE_LETTERS = ["A", "B", "C", "D"]

# ---------------------------
# FS helpers
# ---------------------------
def _dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))

def _normalize_leaf_folder(folder: str) -> str:
    """
    Some zips expand to .../images/images/* or .../images/images2/*.
    Dive to the real leaf if needed.
    """
    if not os.path.isdir(folder):
        return folder

    for d in ("images2", "images"):
        p = os.path.join(folder, d)
        if _dir_nonempty(p):
            return p

    if _dir_nonempty(folder):
        return folder

    inner = os.path.join(folder, "images")
    if _dir_nonempty(inner):
        return inner

    return folder

def _extract_zip(zp: str, out_dir: str) -> Optional[str]:
    try:
        if _dir_nonempty(out_dir):
            return _normalize_leaf_folder(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zp) as zf:
            eval_logger.info(f"Extracting {os.path.basename(zp)} -> {out_dir} (one-time)...")
            zf.extractall(out_dir)
        return _normalize_leaf_folder(out_dir)
    except Exception as e:
        eval_logger.warning(f"Failed to extract {zp}: {e}")
        return None

@lru_cache(maxsize=16)
def _resolve_image_root_for_repo(repo_id: str) -> str:
    """
    Resolve PMC-VQA images directory (prefers images2/ then images/).
    Auto-downloads to HF cache (honors HF_HOME / HF_HUB_CACHE).
    """
    if not repo_id:
        eval_logger.error("No repo_id provided to image resolver.")
        return ""
    if snapshot_download is None:
        eval_logger.error("Please install huggingface_hub: pip install -U huggingface_hub")
        return ""

    snap_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["images2/*", "images/*", "images2.zip", "images.zip"],
        resume_download=True,
    )

    for d in ("images2", "images"):
        p = os.path.join(snap_dir, d)
        if _dir_nonempty(p):
            return _normalize_leaf_folder(p)

    for zname, outdir in (("images2.zip", "images2"), ("images.zip", "images")):
        zp = os.path.join(snap_dir, zname)
        if os.path.isfile(zp):
            out = _extract_zip(zp, os.path.join(snap_dir, outdir))
            if out and _dir_nonempty(out):
                return _normalize_leaf_folder(out)

    eval_logger.error(f"Could not locate images under snapshot for {repo_id}.")
    return ""

def _find_file(root: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(root, filename),
        os.path.join(root, "images", filename),
        os.path.join(root, "images2", filename),
        os.path.join(root, "train", filename),
        os.path.join(root, "test", filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    for r, _dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None

# ---------------------------
# Text helpers
# ---------------------------
def _collect_choices(doc: Dict[str, Any]) -> List[str]:
    choices: List[str] = []
    for letter in CHOICE_LETTERS:
        for key in (f"Choice {letter}", f"Choice_{letter}", f"choice_{letter}"):
            if key in doc and doc[key] is not None:
                raw = doc[key]
                if isinstance(raw, str):
                    parts = raw.split(":", 1)  # handle 'A: text'
                    if len(parts) == 2 and len(parts[0]) <= 3:
                        raw = parts[1].strip()
                choices.append(str(raw).strip())
                break
    return choices

def _extract_choice_letter(text: str) -> str:
    """
    Extract final choice letter from model output.
    Handles: 'Answer: B', 'Option C', '(D)', trailing 'C', etc.
    """
    if not text:
        return ""
    patterns = [
        r"answer\s*[:\-]\s*([ABCD])\b",
        r"(?:choice|option)\s*([ABCD])\b",
        r"[\(\[\{]\s*([ABCD])\s*[\)\]\}]",
        r"\b([ABCD])\b[^\w]*$",
    ]
    for pat in patterns:
        m = list(re.finditer(pat, text, flags=re.IGNORECASE))
        if m:
            return m[-1].group(1).upper()
    return ""

# ---------------------------
# Task adapters
# ---------------------------
def pmc_vqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {"pre_prompt": "", "post_prompt": ""}
    pre = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
    post = lmms_eval_specific_kwargs.get("post_prompt", "") or ""

    question = (doc.get("Question") or doc.get("question") or "").strip()
    choices = _collect_choices(doc)
    formatted = "\n".join(f"{l}. {t}" for l, t in zip(CHOICE_LETTERS, choices))
    return f"{pre}{question}\n{formatted}\n{post}".strip()

def pmc_vqa_doc_to_target(doc: Dict[str, Any]) -> str:
    label = doc.get("Answer_label") or doc.get("Answer_Label") or doc.get("answer_label")
    if isinstance(label, str) and label.strip() in CHOICE_LETTERS:
        return label.strip().upper()
    # fallback: try matching Answer text to a choice
    answer_text = (doc.get("Answer") or doc.get("answer") or "").strip().lower()
    for idx, choice in enumerate(_collect_choices(doc)):
        if choice.strip().lower() == answer_text:
            return CHOICE_LETTERS[idx]
    return (doc.get("Answer") or doc.get("answer") or "").strip().upper()

def pmc_vqa_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Image.Image]:
    repo_id = ""
    if lmms_eval_specific_kwargs:
        repo_id = lmms_eval_specific_kwargs.get("dataset_repo") or lmms_eval_specific_kwargs.get("dataset_path") or ""
    if not repo_id:
        repo_id = "RadGenome/PMC-VQA"

    root = _resolve_image_root_for_repo(repo_id)
    if not root:
        raise RuntimeError(
            f"Images folder could not be resolved for '{repo_id}'. "
            "Ensure dataset assets are cached (images.zip/images2.zip)."
        )

    fig_key = next((k for k in ("Figure_path", "figure_path", "Figure_Path", "image_path") if k in doc), None)
    if not fig_key:
        raise RuntimeError("No figure path key found in document.")
    filename = os.path.basename(str(doc[fig_key]))

    path = _find_file(root, filename)
    if not path:
        raise RuntimeError(
            f"Image file '{filename}' not found under '{root}'. "
            "Verify that images.zip/images2.zip were downloaded and extracted."
        )

    try:
        with Image.open(path) as im:
            return [im.convert("RGB")]
    except Exception as e:
        eval_logger.warning(f"Failed to open image {path}: {e}")
        return []
    
def pmc_vqa_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    pred_raw = results[0] if results else ""
    pred = (pred_raw or "").strip()
    target = pmc_vqa_doc_to_target(doc)

    pred_letter = _extract_choice_letter(pred)
    score = 1.0 if (pred_letter and pred_letter == target) else 0.0

    qid = f"{doc.get('Figure_path','')}::{(doc.get('Question') or doc.get('question') or '').strip()}"[:256]

    return {"accuracy": {"question_id": qid, "score": score}}

def pmc_vqa_aggregate_results(results: List[Dict[str, Any]]) -> float:
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
    acc = (total / count) * 100.0 if count else 0.0
    eval_logger.info(f"PMC-VQA Accuracy: {acc:.2f}")
    return acc
"""PDF-first Q/A lookup for local Noryx RAG training PDFs.

It extracts simple Q/A pairs from configured PDFs and returns a direct answer
before the normal chat pipeline runs.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.runtime_paths import get_app_root

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
QA_RE = re.compile(
    r"(?:^|\s)(Q\d+\.\s+.*?\?)\s+(.*?)(?=\s+Q\d+\.\s+|\Z)",
    re.DOTALL,
)
QA_ANSWER_RE = re.compile(
    r"(?:^|\s)Q(?P<num>\d+)\s*(?:\([^)]*\))?\.\s*(?P<question>.*?\?)\s+"
    r"A(?P=num)\.\s*(?P<answer>.*?)(?=\s+Q\d+\s*(?:\([^)]*\))?\.\s+|\Z)",
    re.DOTALL,
)

DEFAULT_CONFIG = {
    "enabled": True,
    "document_path": "data/rag/documents/Qwen_RAG_Custom_QA_Training.pdf",
    "document_paths": [
        "data/rag/documents/Qwen_RAG_Custom_QA_Training.pdf",
        "data/rag/documents/Noryx_Supplemental_QA.pdf",
    ],
    "match_threshold": 0.80,
    "cache_seconds": 30,
}


@dataclass
class PdfQaMatch:
    answer: str
    question: str
    score: float
    source: str
    page: Optional[int] = None


_cache: dict[str, object] = {"loaded_at": 0.0, "signature": None, "items": [], "config": DEFAULT_CONFIG}


def _config_path() -> Path:
    return Path(get_app_root()) / "config" / "pdf_qa_rag.json"


def _load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    path = _config_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                cfg.update(raw)
        except Exception as exc:
            logger.warning("Failed to read PDF QA RAG config %s: %s", path, exc)
    return cfg


def _resolve_path(value: str) -> Path:
    path = Path(value or "")
    if not path.is_absolute():
        path = Path(get_app_root()) / path
    return path.resolve()


def _configured_paths(cfg: dict) -> list[Path]:
    raw_paths = cfg.get("document_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raw_paths = [cfg.get("document_path") or DEFAULT_CONFIG["document_path"]]

    paths: list[Path] = []
    seen: set[str] = set()
    for raw in raw_paths:
        if not raw:
            continue
        path = _resolve_path(str(raw))
        key = str(path).lower()
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def _token_similarity(a: str, b: str) -> float:
    a_tokens = set(TOKEN_RE.findall((a or "").lower()))
    b_tokens = set(TOKEN_RE.findall((b or "").lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))
    return pages


def _append_item(items: list[dict], path: Path, page_number: int, question: str, answer: str) -> None:
    question = re.sub(r"^Q\d+\.\s*", "", question or "").strip()
    answer = re.sub(r"\s+", " ", answer or "").strip()
    if question and answer:
        items.append({
            "question": question,
            "question_norm": _normalize(question),
            "answer": answer,
            "page": page_number,
            "source": path.name,
        })


def _extract_items(path: Path) -> list[dict]:
    items: list[dict] = []
    for page_number, text in _read_pdf_pages(path):
        matched_answer_style = False
        for match in QA_ANSWER_RE.finditer(text):
            matched_answer_style = True
            _append_item(items, path, page_number, match.group("question"), match.group("answer"))
        if matched_answer_style:
            continue
        for match in QA_RE.finditer(text):
            _append_item(items, path, page_number, match.group(1), match.group(2))
    return items


def _items_for_config(cfg: dict) -> tuple[list[dict], Path]:
    paths = _configured_paths(cfg)
    fallback_path = paths[0] if paths else _resolve_path(DEFAULT_CONFIG["document_path"])

    existing = [path for path in paths if path.exists()]
    for path in paths:
        if not path.exists():
            logger.warning("PDF QA RAG document not found: %s", path)
    if not existing:
        return [], fallback_path

    cache_seconds = max(1, int(cfg.get("cache_seconds") or DEFAULT_CONFIG["cache_seconds"]))
    now = time.monotonic()
    signature = tuple((str(path), path.stat().st_mtime) for path in existing)
    if (
        _cache.get("signature") == signature
        and now - float(_cache.get("loaded_at") or 0.0) < cache_seconds
        and isinstance(_cache.get("items"), list)
    ):
        return _cache["items"], fallback_path  # type: ignore[return-value]

    items: list[dict] = []
    for path in existing:
        try:
            items.extend(_extract_items(path))
        except Exception as exc:
            logger.warning("Failed to extract PDF QA pairs from %s: %s", path, exc)

    _cache.update({"loaded_at": now, "signature": signature, "items": items, "config": cfg})
    return items, fallback_path


def answer_from_pdf(question: str) -> Optional[PdfQaMatch]:
    """Return a direct PDF answer when the question matches a configured Q/A pair."""
    question = (question or "").strip()
    if not question:
        return None

    cfg = _load_config()
    if not cfg.get("enabled", True):
        return None

    items, fallback_path = _items_for_config(cfg)
    if not items:
        return None

    normalized = _normalize(question)
    best: Optional[dict] = None
    best_score = 0.0
    for item in items:
        if item.get("question_norm") == normalized:
            best = item
            best_score = 1.0
            break
        score = _token_similarity(question, str(item.get("question") or ""))
        if score > best_score:
            best = item
            best_score = score

    threshold = float(cfg.get("match_threshold") or DEFAULT_CONFIG["match_threshold"])
    if not best or best_score < threshold:
        return None

    return PdfQaMatch(
        answer=str(best.get("answer") or "").strip(),
        question=str(best.get("question") or "").strip(),
        score=best_score,
        source=str(best.get("source") or fallback_path.name),
        page=best.get("page") if isinstance(best.get("page"), int) else None,
    )


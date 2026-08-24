#!/usr/bin/env python3
"""Small offline PDF RAG pipeline.

Indexes PDFs into a local JSON vector store using a deterministic hashing
embedding. At query time it retrieves relevant chunks and optionally sends the
context to a local llama.cpp model through llama-cli.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import urllib.error
import urllib.request
import sys
from pathlib import Path
from typing import Iterable, Any

LOCAL_SITE_PACKAGES = Path(__file__).resolve().parent / ".venv" / "lib" / "python3.14" / "site-packages"
if LOCAL_SITE_PACKAGES.exists():
    sys.path.append(str(LOCAL_SITE_PACKAGES))

try:
    from pypdf import PdfReader
except ModuleNotFoundError as exc:
    raise SystemExit("pypdf is required. Install it with: python -m pip install pypdf") from exc


DEFAULT_INDEX = "index.json"
DEFAULT_EMBED_DIM = 384
DEFAULT_SERVER_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
PAGE_RE = re.compile(r"\[page\s+(\d+)\]", re.IGNORECASE)
QA_RE = re.compile(
    r"(?:^|\s)(Q\d+\.\s+.*?\?)\s+(.*?)(?=\s+Q\d+\.\s+|\Z)",
    re.DOTALL,
)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n[page {page_number}]\n{text}")
    return "\n".join(pages).strip()


def iter_documents(documents_dir: Path) -> Iterable[tuple[Path, str]]:
    for path in sorted(documents_dir.rglob("*.pdf")):
        try:
            text = read_pdf(path)
        except Exception as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        if text:
            yield path, text
        else:
            print(f"skipping {path}: no extractable text", file=sys.stderr)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_text(text: str, dim: int = DEFAULT_EMBED_DIM) -> list[float]:
    vec = [0.0] * dim
    tokens = TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize_question(text: str) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    return " ".join(tokens)


def sentence_similarity(a: str, b: str) -> float:
    a_tokens = set(TOKEN_RE.findall(a.lower()))
    b_tokens = set(TOKEN_RE.findall(b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def extract_qa_answer(question: str, matches: list[tuple[float, dict[str, Any], str]]) -> str | None:
    normalized_question = normalize_question(question)
    best_score = 0.0
    best_answer: str | None = None

    for _score, _meta, doc in matches:
        for qa_match in QA_RE.finditer(doc):
            candidate_question = re.sub(r"^Q\d+\.\s*", "", qa_match.group(1)).strip()
            answer = qa_match.group(2).strip()
            if normalize_question(candidate_question) == normalized_question:
                score = 1.0
            else:
                score = sentence_similarity(question, candidate_question)
            if score > best_score:
                best_score = score
                best_answer = answer

    if best_score >= 0.80:
        return best_answer
    return None


def first_page_number(text: str) -> int | None:
    match = PAGE_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def source_page(meta: dict[str, Any], doc: str) -> int | None:
    page = meta.get("page")
    if isinstance(page, int):
        return page
    if isinstance(page, str) and page.isdigit():
        return int(page)
    return first_page_number(doc)


def index_path(persist_dir: Path) -> Path:
    return persist_dir / DEFAULT_INDEX


def load_index(persist_dir: Path) -> dict[str, Any]:
    path = index_path(persist_dir)
    if not path.exists():
        raise SystemExit(f"index not found: {path}. Run the index command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_index(persist_dir: Path, payload: dict[str, Any]) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    index_path(persist_dir).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def build_index(args: argparse.Namespace) -> int:
    documents_dir = Path(args.documents).resolve()
    if not documents_dir.exists():
        print(f"documents directory not found: {documents_dir}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    for pdf_path, text in iter_documents(documents_dir):
        chunks = split_text(text, args.chunk_size, args.overlap)
        rel = pdf_path.relative_to(documents_dir).as_posix()
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{rel}:{chunk_index}",
                    "document": chunk,
                    "metadata": {"source": rel, "chunk": chunk_index, "page": first_page_number(chunk)},
                    "embedding": embed_text(chunk, args.embed_dim),
                }
            )

    if not records:
        print("no PDF text was extracted; index was not updated", file=sys.stderr)
        return 1

    payload = {"embed_dim": args.embed_dim, "records": records}
    save_index(Path(args.persist), payload)
    print(f"indexed {len(records)} chunks from {documents_dir}")
    return 0


def retrieve(args: argparse.Namespace) -> list[tuple[float, dict[str, Any], str]]:
    payload = load_index(Path(args.persist))
    query_embedding = embed_text(args.question, int(payload.get("embed_dim", args.embed_dim)))
    scored = []
    for record in payload.get("records", []):
        score = cosine(query_embedding, record["embedding"])
        if score >= args.threshold:
            scored.append((score, record["metadata"], record["document"]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: args.top_k]


def format_context(matches: list[tuple[float, dict[str, Any], str]]) -> str:
    context_parts = []
    for score, meta, doc in matches:
        page = source_page(meta, doc)
        page_label = page if page is not None else "unknown"
        context_parts.append(
            f"[source: {meta.get('source')}, page: {page_label}, chunk: {meta.get('chunk')}, score: {score:.4f}]\n{doc}"
        )
    return "\n\n".join(context_parts)


def build_document_prompt(question: str, matches: list[tuple[float, dict[str, Any], str]]) -> str:
    return (
        "You are a document assistant.\n\n"
        "Answer the question using ONLY the provided context.\n\n"
        f"Context:\n{format_context(matches)}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def build_fallback_prompt(question: str) -> str:
    return (
        "You are a helpful AI assistant.\n\n"
        "Answer the user's question normally using your knowledge.\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def print_document_sources(matches: list[tuple[float, dict[str, Any], str]]) -> None:
    print("\nSource:")
    seen = set()
    for score, meta, doc in matches:
        source = meta.get("source")
        page = source_page(meta, doc)
        key = (source, page, round(score, 4))
        if key in seen:
            continue
        seen.add(key)
        page_label = page if page is not None else "unknown"
        print(f"- PDF filename: {source}")
        print(f"  Page: {page_label}")
        print(f"  Similarity score: {score:.4f}")


def print_general_source() -> None:
    print("\nSource:")
    print("- General LLM knowledge")


def run_llama_server(args: argparse.Namespace, prompt: str) -> str:
    url = args.server_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.n_predict,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"failed to call llama.cpp server at {url}: {exc}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise SystemExit(f"llama.cpp server returned no choices: {body}")
    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("text") or ""
    return content.strip()


def run_llama_cli(args: argparse.Namespace, prompt: str) -> int:
    command = [args.llama_cli, "-m", args.model, "-p", prompt, "-n", str(args.n_predict), "--temp", str(args.temperature)]
    completed = subprocess.run(command, check=False, text=True)
    return completed.returncode


def ask(args: argparse.Namespace) -> int:
    matches = retrieve(args)
    is_document_answer = bool(matches and matches[0][0] >= args.threshold)

    if is_document_answer:
        prompt = build_document_prompt(args.question, matches)
        answer = run_llama_server(args, prompt)
        print("Answer:")
        print(answer)
        print_document_sources(matches)
        return 0

    prompt = build_fallback_prompt(args.question)
    answer = run_llama_server(args, prompt)
    print("Answer:")
    print(answer)
    print_general_source()
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline PDF RAG with optional llama.cpp generation")
    parser.add_argument("--persist", default=".rag-index", help="index persistence directory")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help="hashing embedding dimension")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="index PDFs")
    index_parser.add_argument("--documents", default="documents", help="directory containing PDF files")
    index_parser.add_argument("--chunk-size", type=int, default=220, help="chunk size in words")
    index_parser.add_argument("--overlap", type=int, default=40, help="chunk overlap in words")
    index_parser.set_defaults(func=build_index)

    ask_parser = subparsers.add_parser("ask", help="ask a question")
    ask_parser.add_argument("question", help="question to answer")
    ask_parser.add_argument("--top-k", type=int, default=4, help="number of chunks to retrieve")
    ask_parser.add_argument("--threshold", type=float, default=0.10, help="minimum cosine similarity")
    ask_parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="llama.cpp OpenAI-compatible server URL")
    ask_parser.add_argument("--llama-cli", help="path to llama-cli for generation")
    ask_parser.add_argument("--model", default=DEFAULT_MODEL, help="llama.cpp server model name")
    ask_parser.add_argument("--n-predict", type=int, default=256, help="maximum generated tokens")
    ask_parser.add_argument("--temperature", type=float, default=0.2, help="generation temperature")
    ask_parser.add_argument("--timeout", type=float, default=120.0, help="llama.cpp server request timeout in seconds")
    ask_parser.set_defaults(func=ask)

    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if getattr(args, "overlap", 0) >= getattr(args, "chunk_size", 1):
        parser.error("--overlap must be smaller than --chunk-size")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())










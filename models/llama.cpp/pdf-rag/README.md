# PDF RAG

Local PDF retrieval with hybrid answering through a llama.cpp OpenAI-compatible server.

## Index PDFs

```bash
python rag.py index --documents documents
```

## Ask a question

Start your llama.cpp server on `http://localhost:8080/v1`, then run:

```bash
python rag.py ask "What is Project Aurora?" --top-k 4
```

Default server/model settings:

```text
--server-url http://localhost:8080/v1
--model Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M
```

## Hybrid behavior

If the top PDF similarity score is at or above `--threshold`, the answer is generated only from retrieved PDF context and includes PDF filename, page, and score.

If no retrieved chunk meets `--threshold`, the original question is sent to the LLM normally and the source is marked as general LLM knowledge.

The script keeps the existing PDF extraction, chunking, hashing embeddings, JSON vector index, retrieval, and CLI commands.

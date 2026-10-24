"""Persistent memory on ChromaDB.

Two kinds of records live in one collection:
  - "email": snippets/bodies we saw via MCP tool results (upserted by message id)
  - "turn":  past Q&A pairs
Retrieved memories are injected into the prompt as *hints*, not ground truth.
"""
import json
import time
import uuid

import chromadb


class Memory:
    def __init__(self, path, embedding_function=None):
        self.client = chromadb.PersistentClient(path=str(path))
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        self.col = self.client.get_or_create_collection(
            "memory", metadata={"hnsw:space": "cosine"}, **kwargs
        )

    def count(self) -> int:
        return self.col.count()

    def add(self, text: str, kind: str, doc_id: str | None = None, **meta) -> None:
        text = text.strip()
        if not text:
            return
        meta = {k: v for k, v in meta.items() if v is not None}
        self.col.upsert(
            ids=[doc_id or f"{kind}:{uuid.uuid4().hex}"],
            documents=[text],
            metadatas=[{"kind": kind, "ts": int(time.time()), **meta}],
        )

    def search(self, query: str, k: int = 4, max_distance: float = 0.7) -> list[dict]:
        if self.count() == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, self.count()))
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            if dist <= max_distance:
                out.append({"text": doc, "meta": meta, "distance": round(dist, 3)})
        return out

    def add_turn(self, question: str, answer: str) -> None:
        self.add(f"Q: {question}\nA: {answer}", kind="turn")

    def ingest_tool_result(self, tool: str, raw: str) -> int:
        """Index Gmail MCP results (JSON list of messages, or one message dict).
        Coupled to our gmail_server output shape on purpose; generalize later."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return 0
        items = data if isinstance(data, list) else [data]
        n = 0
        for it in items:
            if not isinstance(it, dict) or "id" not in it:
                continue
            # don't let a snippet-only search hit clobber a stored full body
            if "body" not in it and self.col.get(ids=[f"email:{it['id']}"])["ids"]:
                continue
            body = it.get("body") or it.get("snippet") or ""
            text = (
                f"From: {it.get('from', '')} | Date: {it.get('date', '')} | "
                f"Subject: {it.get('subject', '')}\n{body}"
            )
            # a full body (get_email) overwrites an earlier snippet-only record
            self.add(text[:3000], kind="email", doc_id=f"email:{it['id']}", source=tool)
            n += 1
        return n

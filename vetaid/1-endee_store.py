"""Endee-backed vector store for the VetAid RAG pipeline.

Drop-in replacement for the previous ChromaDB store. It implements the
LangChain ``VectorStore`` interface so the rest of the pipeline
(``as_retriever``, ``similarity_search_with_score``, BM25 corpus rebuild)
works unchanged, while all vectors live in an Endee server
(https://endee.io) instead of a local Chroma directory.

Layout of state:
- Vectors + metadata: Endee collection (one per dataset signature).
- Local manifest JSON (per dataset signature): chunk ids, texts and
  metadata. Used to rebuild the BM25 corpus and for document counts,
  mirroring what ``collection.get()`` provided with ChromaDB.

Required environment:
    ENDEE_TOKEN        auth token from the Endee Serverless dashboard (app.endee.io)
Optional environment:
    ENDEE_BASE_URL     Enterprise/self-hosted only - default http://localhost:8080/api/v2
    ENDEE_ROOT_TOKEN   Enterprise/self-hosted root token (alternative to ENDEE_TOKEN)
    ENDEE_DB_NAME      Enterprise/self-hosted only - default "vetaid"

Exactly one of ENDEE_TOKEN (serverless) or ENDEE_ROOT_TOKEN (self-hosted)
is required.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from endee import Endee

logger = logging.getLogger(__name__)

ENDEE_TOKEN = os.environ.get("ENDEE_TOKEN", "")
ENDEE_BASE_URL = os.environ.get("ENDEE_BASE_URL", "http://localhost:8080/api/v2")
ENDEE_ROOT_TOKEN = os.environ.get("ENDEE_ROOT_TOKEN", "")
ENDEE_DB_NAME = os.environ.get("ENDEE_DB_NAME", "vetaid")

VECTOR_FIELD = "embedding"
UPSERT_BATCH = 256
STATE_ROOT = Path("endee_state")


def _root_client() -> Endee:
    if not ENDEE_ROOT_TOKEN:
        raise RuntimeError(
            "No Endee credentials configured. Either set ENDEE_TOKEN from the "
            "Endee Serverless dashboard (app.endee.io), or self-host Endee "
            "(https://endee.io/install.sh) and set ENDEE_ROOT_TOKEN."
        )
    admin = Endee(token=ENDEE_ROOT_TOKEN)
    admin.set_base_url(ENDEE_BASE_URL)
    return admin


def get_or_create_data_client(db_name: str = ENDEE_DB_NAME) -> Endee:
    """Return an Endee client for data operations.

    Serverless (ENDEE_TOKEN set): the dashboard-issued token encodes the
    region, so the client connects directly with no base URL or bootstrap.
    Self-hosted (ENDEE_ROOT_TOKEN set): create the database once, mint/reuse
    a db token stored locally, and connect scoped to it.
    """
    if ENDEE_TOKEN:
        return Endee(token=ENDEE_TOKEN)

    STATE_ROOT.mkdir(exist_ok=True)
    token_file = STATE_ROOT / f"{db_name}.token"

    admin = _root_client()
    existing = {db.get("name") for db in (admin.list_databases() or [])}

    if token_file.exists():
        db_token = token_file.read_text().strip()
    elif db_name in existing:
        # Database exists but no local token: mint a fresh db token.
        db_token = admin.create_token(db_name, name="vetaid-rw", token_type="rw")
        token_file.write_text(db_token)
    else:
        db_token = admin.create_database(db_name, db_type="enterprise")
        token_file.write_text(db_token)

    client = Endee(token=db_token)
    client.set_base_url(ENDEE_BASE_URL)
    return client


def _chroma_filter_to_endee(filter_dict: Optional[dict]) -> Optional[list[dict]]:
    """Translate the pipeline's Chroma-style filter into Endee filter syntax.

    Chroma:  {"species": {"$in": ["dog", "general"]}}  or  {"species": "dog"}
    Endee:   [{"species": {"$in": ["dog", "general"]}}] or [{"species": {"$eq": "dog"}}]
    """
    if not filter_dict:
        return None
    clauses = []
    for field, condition in filter_dict.items():
        if isinstance(condition, dict):
            clauses.append({field: condition})
        else:
            clauses.append({field: {"$eq": condition}})
    return clauses


class _ManifestCollectionAdapter:
    """Mimics the small slice of Chroma's collection API the app uses:
    ``get()`` (documents/metadatas/ids) and ``count()``."""

    def __init__(self, store: "EndeeVectorStore"):
        self._store = store

    def get(self, **_: Any) -> dict:
        items = self._store._manifest_items()
        return {
            "documents": [it["text"] for it in items],
            "metadatas": [it["metadata"] for it in items],
            "ids": [it["id"] for it in items],
        }

    def count(self) -> int:
        return len(self._store._manifest_items())


class EndeeVectorStore(VectorStore):
    """LangChain-compatible vector store backed by an Endee collection."""

    def __init__(
        self,
        client: Endee,
        collection_name: str,
        embedding: Embeddings,
        manifest_path: Path,
    ):
        self._client = client
        self._collection_name = collection_name
        self._embedding = embedding
        self._manifest_path = Path(manifest_path)
        self._endee_coll = None  # Endee collection handle, opened lazily
        self._adapter = _ManifestCollectionAdapter(self)

    # ---- LangChain interface -------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def _collection(self):
        """Chroma-compat shim: app code uses vectorstore._collection.get()/count()."""
        return self._adapter

    def _get_collection(self):
        if self._endee_coll is None:
            self._endee_coll = self._client.get_collection(self._collection_name)
        return self._endee_coll

    # ---- manifest ------------------------------------------------------------

    def _manifest_items(self) -> list[dict]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Manifest unreadable at %s", self._manifest_path)
        return []

    def _write_manifest(self, items: list[dict]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(items))

    # ---- writes ----------------------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        names = set()
        try:
            names = {c.get("name") for c in (self._client.list_collections() or [])}
        except Exception as e:
            logger.warning("Could not list Endee collections: %s", e)
        if self._collection_name not in names:
            self._client.create_collection(
                name=self._collection_name,
                fields=[{
                    "name": VECTOR_FIELD,
                    "type": "vector",
                    "params": {"dimension": dimension, "space_type": "cosine", "precision": "float32"},
                }],
            )
            logger.info("Created Endee collection %s (dim=%d)", self._collection_name, dimension)

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as e:
            logger.info("Delete before rebuild skipped: %s", e)
        self._endee_coll = None
        if self._manifest_path.exists():
            self._manifest_path.unlink()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        vectors = self._embedding.embed_documents(texts)
        if vectors:
            self.ensure_collection(dimension=len(vectors[0]))
        collection = self._get_collection()

        for start in range(0, len(texts), UPSERT_BATCH):
            batch = []
            for text, meta, oid, vec in zip(
                texts[start:start + UPSERT_BATCH],
                metadatas[start:start + UPSERT_BATCH],
                ids[start:start + UPSERT_BATCH],
                vectors[start:start + UPSERT_BATCH],
            ):
                batch.append({
                    "id": oid,
                    "meta": {"text": text, **(meta or {})},
                    "filter": {"species": (meta or {}).get("species", "general")},
                    "fields": {VECTOR_FIELD: [float(x) for x in vec]},
                })
            collection.upsert(batch)

        items = self._manifest_items()
        known = {it["id"] for it in items}
        for text, meta, oid in zip(texts, metadatas, ids):
            if oid not in known:
                items.append({"id": oid, "text": text, "metadata": meta or {}})
        self._write_manifest(items)
        return ids

    # ---- reads -----------------------------------------------------------------

    def _query(self, embedding: list[float], k: int, filter: Optional[dict]) -> list[tuple[Document, float]]:
        collection = self._get_collection()
        res = collection.search(
            fields={VECTOR_FIELD: {"query": [float(x) for x in embedding], "limit": k}},
            filter=_chroma_filter_to_endee(filter),
        )
        hits = res.get("results", {}).get(VECTOR_FIELD, [])
        out = []
        for hit in hits:
            meta = dict(hit.get("meta") or {})
            text = meta.pop("text", "")
            # Endee returns cosine similarity (higher = closer). Convert to a
            # distance-like score so downstream relevance math stays valid.
            distance = 1.0 - float(hit.get("similarity", 0.0))
            out.append((Document(page_content=text, metadata=meta), distance))
        return out

    def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)]

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any) -> list[tuple[Document, float]]:
        vec = self._embedding.embed_query(query)
        return self._query(vec, k, filter)

    def similarity_search_by_vector(self, embedding: list[float], k: int = 4, filter: Optional[dict] = None, **kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self._query(embedding, k, filter)]

    # ---- construction -----------------------------------------------------------

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Embeddings, metadatas: Optional[list[dict]] = None, **kwargs: Any) -> "EndeeVectorStore":
        client = kwargs.pop("client")
        collection_name = kwargs.pop("collection_name")
        manifest_path = kwargs.pop("manifest_path")
        store = cls(client=client, collection_name=collection_name, embedding=embedding, manifest_path=manifest_path)
        store.add_texts(texts, metadatas=metadatas)
        return store

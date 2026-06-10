"""
Hlokk - Vector Store (ChromaDB + OpenAI-compatible Embedding API)
Handles embedding generation, storage, and retrieval.
"""
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from openai import OpenAI

from text_chunker import Chunk


def _get_embedding_client(config: dict) -> OpenAI:
    """Create an OpenAI-compatible client for embedding API."""
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


def create_embeddings(
    texts: list[str], config: dict, batch_size: int = 32
) -> list[list[float]]:
    """
    Generate embeddings via OpenAI-compatible API.

    Args:
        texts: list of text strings to embed
        config: embedding API config (api_key, base_url, model)
        batch_size: number of texts per API call
    """
    client = _get_embedding_client(config)
    model = config["model"]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeds = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeds)

    return all_embeddings


def _collection_name(prefix: str, paper_id: str) -> str:
    """Generate a valid ChromaDB collection name."""
    safe_id = hashlib.md5(paper_id.encode()).hexdigest()[:12]
    name = f"{prefix}_{safe_id}"
    # ChromaDB requires 3-63 chars, start/end with alphanum
    return name[:63]


def init_store(db_path: str) -> chromadb.ClientAPI:
    """Initialize a persistent ChromaDB client."""
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def store_chunks(
    chunks: list[Chunk],
    embed_config: dict,
    db_client: chromadb.ClientAPI,
    collection_prefix: str = "hlokk",
    paper_id: str = "default",
) -> str:
    """
    Embed and store chunks in ChromaDB.
    Returns the collection name.
    """
    col_name = _collection_name(collection_prefix, paper_id)
    collection = db_client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # skip if already populated
    if collection.count() > 0:
        return col_name

    texts = [c.content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]

    # embed
    embeddings = create_embeddings(texts, embed_config)

    # upsert
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return col_name


def retrieve_relevant(
    queries: list[str],
    embed_config: dict,
    db_client: chromadb.ClientAPI,
    collection_name: str,
    top_k: int = 15,
) -> list[dict]:
    """
    Retrieve top-K relevant chunks for a set of queries.

    Returns deduplicated list of:
        {"content": str, "metadata": dict, "relevance_score": float, "matched_query": str}
    """
    collection = db_client.get_collection(name=collection_name)
    query_embeddings = create_embeddings(queries, embed_config)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # deduplicate and merge results across queries
    seen_ids = set()
    retrieved = []

    for q_idx, query_text in enumerate(queries):
        if not results["ids"][q_idx]:
            continue
        for i, doc_id in enumerate(results["ids"][q_idx]):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            distance = results["distances"][q_idx][i]
            score = 1.0 - distance  # cosine distance → similarity
            retrieved.append({
                "content": results["documents"][q_idx][i],
                "metadata": results["metadatas"][q_idx][i],
                "relevance_score": round(score, 4),
                "matched_query": query_text,
            })

    # sort by relevance
    retrieved.sort(key=lambda x: x["relevance_score"], reverse=True)
    return retrieved


def build_queries_from_profile(profile: dict) -> list[str]:
    """
    Build retrieval queries from user profile.
    Uses pain points + research directions as primary queries.
    """
    queries = []

    # pain points are the highest priority queries
    for pain in profile.get("pain_points", []):
        queries.append(pain)

    # research directions
    for direction in profile.get("research_directions", []):
        queries.append(direction)

    # combine pain points with tech stack for more specific queries
    tech_tools = profile.get("tech_stack", {}).get("frameworks", [])
    for pain in profile.get("pain_points", [])[:3]:
        for tool in tech_tools[:2]:
            queries.append(f"{pain} {tool}")

    # output preference focus
    focus = profile.get("output_preference", {}).get("focus", "")
    if focus == "methods":
        queries.extend([
            "methods parameters threshold",
            "software tool version pipeline",
            "statistical test model",
        ])
    elif focus == "reproduction":
        queries.extend([
            "step by step procedure protocol",
            "code repository github",
            "data preprocessing pipeline",
        ])

    return queries

# Embedding with fastembed — runs entirely locally, no API calls, no PyTorch.
#
# How fastembed works:
# - Uses ONNX Runtime instead of PyTorch (much smaller, no GPU needed)
# - Downloads the model once (~45MB) on first use, caches it permanently
# - After that, all embedding is a pure local CPU operation
# - Model: BAAI/bge-small-en-v1.5 — 384 dimensions, excellent quality
#   for semantic search, widely used in production RAG systems
#
# Everything else (Qdrant interactions, search logic) is identical to before.

import os
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

# Load the model once at module level — takes a few seconds on first run
# while it downloads the ONNX model file (~45MB). Every call after that
# is instant since the model stays in memory.
print(f"Loading fastembed model '{EMBEDDING_MODEL_NAME}'...")
_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
print("Embedding model ready.")


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection_exists():
    """Creates the Qdrant collection if it doesn't exist. Called at startup."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: '{QDRANT_COLLECTION}'")
    else:
        print(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")


def embed_text(text: str) -> list[float]:
    """
    Converts text to a vector using the local fastembed model.
    No network call — runs entirely on CPU inside the container.
    fastembed.embed() returns a generator, so we wrap it in list()
    and take the first (only) result.
    """
    embeddings = list(_model.embed([text]))
    return embeddings[0].tolist()


def embed_paper(paper_id: int, external_id: str, title: str, abstract: str) -> bool:
    """Embeds a paper and stores its vector in Qdrant."""
    try:
        text_to_embed = f"{title}\n\n{abstract}"
        vector = embed_text(text_to_embed)

        qdrant = get_qdrant_client()
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=paper_id,
                    vector=vector,
                    payload={
                        "external_id": external_id,
                        "title": title,
                        "abstract": abstract[:500],
                    },
                )
            ],
        )
        return True

    except Exception as e:
        print(f"Failed to embed paper {paper_id}: {e}")
        return False


def search_similar_papers(query: str, limit: int = 10) -> list[dict]:
    """
    Embeds the query locally and searches Qdrant for the
    most semantically similar stored paper vectors.
    """
    query_vector = embed_text(query)
    qdrant = get_qdrant_client()

    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "paper_id": result.id,
            "score": result.score,
            "title": result.payload.get("title", ""),
        }
        for result in results
    ]

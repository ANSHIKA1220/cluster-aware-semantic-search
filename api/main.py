import logging
import time
import numpy as np
import pickle
import faiss
import joblib

from fastapi import FastAPI
from api.models import QueryRequest
from utils.embedding_utils import EmbeddingModel
from cache.semantic_cache import SemanticCache


# -----------------------------
# Logging Setup
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="Semantic Search Cache API")


# -----------------------------
# Load resources at startup
# -----------------------------

logger.info("Loading embedding model...")
embedder = EmbeddingModel()

logger.info("Loading FAISS index...")
index = faiss.read_index("vector_store/faiss_index.bin")

logger.info("Loading documents...")
with open("data/newsgroups_raw.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]

logger.info("Loading clustering model...")
gmm = joblib.load("clustering/gmm_model.pkl")

logger.info("Initializing semantic cache...")
cache = SemanticCache(gmm_model=gmm)


# -----------------------------
# Helper Function
# -----------------------------

def search_documents(query_embedding, k=5):

    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = [documents[i] for i in indices[0]]

    return "\n\n".join(results)


# -----------------------------
# POST /query
# -----------------------------

@app.post("/query")
def query_system(request: QueryRequest):

    query = request.query
    start_time = time.time()

    query_embedding = embedder.encode_query(query).astype("float32")

    # Check semantic cache
    cache_result = cache.lookup(query_embedding)

    # CACHE HIT
    if cache_result is not None:

        latency = time.time() - start_time

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": float(cache_result["similarity_score"]),
            "result": cache_result["result"],
            "dominant_cluster": int(cache_result["cluster"]),
            "latency_ms": round(latency * 1000, 2)
        }

    # CACHE MISS → Perform FAISS search
    result = search_documents(query_embedding)

    cluster = cache.store(query, query_embedding, result)

    latency = time.time() - start_time

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": int(cluster),
        "latency_ms": round(latency * 1000, 2)
    }


# -----------------------------
# GET /cache/stats
# -----------------------------

@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# -----------------------------
# DELETE /cache
# -----------------------------

@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}
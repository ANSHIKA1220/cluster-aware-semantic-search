import pickle
import numpy as np
import faiss
from utils.embedding_utils import EmbeddingModel


def main():

    with open("data/newsgroups_raw.pkl", "rb") as f:
        data = pickle.load(f)

    documents = data["documents"]

    # Embedding Model Choice: sentence-transformers/all-MiniLM-L6-v2
    # Justification:
    # - it produces 384-dimensional embeddings
    # - it is lightweight and fast
    # - optimized for semantic similarity tasks
    # - good tradeoff between performance and accuracy for ~20k documents
    embedder = EmbeddingModel()

    embeddings = embedder.encode_documents(documents)

    embeddings = np.array(embeddings).astype("float32")

    np.save("embeddings/document_embeddings.npy", embeddings)

    dimension = embeddings.shape[1]

    # Vector Database Choice: FAISS IndexFlatL2
    # Justification:
    # - exact nearest neighbor search
    # - suitable for datasets under ~100k vectors
    # - avoids complexity of approximate indices such as IVF or HNSW
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "vector_store/faiss_index.bin")

    print("Embeddings and FAISS index created")


if __name__ == "__main__":
    main()
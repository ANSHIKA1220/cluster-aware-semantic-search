from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode_documents(self, docs):
        return self.model.encode(
            docs,
            show_progress_bar=True,
            batch_size=32
        )

    def encode_query(self, query):
        return self.model.encode([query])[0]
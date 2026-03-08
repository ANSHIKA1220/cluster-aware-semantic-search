import numpy as np
import json
from collections import defaultdict
from utils.similarity import cosine_similarity


class SemanticCache:

    def __init__(self, gmm_model, threshold=0.90, cache_file="cache/cache_state.json"):

        self.gmm = gmm_model
        self.threshold = threshold
        self.cache_file = cache_file

        # cluster → list of cache entries
        self.cache = defaultdict(list)

        # statistics
        self.hit_count = 0
        self.miss_count = 0

        # load persisted cache
        self.load_cache()


    # -----------------------------
    # Predict top clusters
    # -----------------------------
    def get_clusters(self, embedding):

        probs = self.gmm.predict_proba([embedding])[0]

        # get top 2 clusters
        top_clusters = np.argsort(probs)[-2:]

        return top_clusters


    # -----------------------------
    # Cache lookup
    # -----------------------------
    def lookup(self, query_embedding):

        clusters = self.get_clusters(query_embedding)

        best_match = None
        best_score = 0

        for cluster in clusters:

            for entry in self.cache[cluster]:

                stored_embedding = np.array(entry["embedding"])

                score = cosine_similarity(
                    query_embedding,
                    stored_embedding
                )

                if score > best_score:
                    best_score = score
                    best_match = entry

        if best_match and best_score >= self.threshold:

            self.hit_count += 1

            return {
                "matched_query": best_match["query"],
                "similarity_score": float(best_score),
                "result": best_match["result"],
                "cluster": int(best_match["cluster"])
            }

        self.miss_count += 1

        return None


    # -----------------------------
    # Store new query in cache
    # -----------------------------
    def store(self, query, embedding, result):

        clusters = self.get_clusters(embedding)

        dominant_cluster = int(clusters[-1])

        entry = {
            "query": query,
            "embedding": embedding.tolist(),
            "result": result,
            "cluster": dominant_cluster
        }

        self.cache[dominant_cluster].append(entry)

        self.save_cache()

        return dominant_cluster


    # -----------------------------
    # Cache statistics
    # -----------------------------
    def stats(self):

        total_entries = sum(len(v) for v in self.cache.values())

        total_queries = self.hit_count + self.miss_count

        hit_rate = (
            self.hit_count / total_queries
            if total_queries > 0 else 0
        )

        cluster_sizes = {
            int(k): len(v)
            for k, v in self.cache.items()
        }

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 3),
            "clusters": cluster_sizes
        }


    # -----------------------------
    # Clear cache
    # -----------------------------
    def clear(self):

        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

        self.save_cache()


    # -----------------------------
    # Save cache to disk
    # -----------------------------
    def save_cache(self):

        serializable_cache = {
            str(cluster): entries
            for cluster, entries in self.cache.items()
        }

        data = {
            "cache": serializable_cache,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count
        }

        with open(self.cache_file, "w") as f:

            json.dump(data, f)


    # -----------------------------
    # Load cache from disk
    # -----------------------------
    def load_cache(self):

        try:

            with open(self.cache_file, "r") as f:

                data = json.load(f)

                self.hit_count = data["hit_count"]
                self.miss_count = data["miss_count"]

                for cluster, entries in data["cache"].items():

                    self.cache[int(cluster)] = entries

        except FileNotFoundError:

            pass
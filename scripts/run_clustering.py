import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import joblib
from sklearn.mixture import GaussianMixture


def select_cluster_count(embeddings):

    bic_scores = []
    cluster_range = range(8, 21)

    for k in cluster_range:
        print(f"Testing clusters: {k}")

        # We use Gaussian Mixture Models (GMM) instead of KMeans because:
        # - GMM produces probabilistic cluster memberships instead of hard labels.
        # - This allows documents to belong to multiple topics simultaneously.
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="tied",
            max_iter=100,
            n_init=2,
            random_state=42
        )

        gmm.fit(embeddings)

        bic_scores.append(gmm.bic(embeddings))

    best_k = cluster_range[np.argmin(bic_scores)]

    print("Best cluster count:", best_k)

    return best_k


def main():

    embeddings = np.load("embeddings/document_embeddings.npy")
    embeddings = embeddings.astype("float32")

    best_k = select_cluster_count(embeddings)

    # We use Gaussian Mixture Models (GMM) instead of KMeans.
    # GMM produces probabilistic cluster memberships which allow documents 
    # to belong to multiple topics simultaneously, perfectly capturing 
    # semantic overlap in the text corpus.
    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="tied",
        random_state=42
    )

    gmm.fit(embeddings)

    joblib.dump(gmm, "clustering/gmm_model.pkl")

    print("Clustering model saved")


if __name__ == "__main__":
    main()
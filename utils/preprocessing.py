from sklearn.datasets import fetch_20newsgroups


def load_dataset():

    # We remove headers, footers, and quotes when loading the 20 Newsgroups dataset because:
    # - Headers contain email metadata, addresses, and routing information.
    # - Footers contain signatures.
    # - Quotes contain repeated quoted messages.
    # Removing these components is critical because they introduce non-semantic noise 
    # which would distort embeddings and clustering.
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = dataset.data
    targets = dataset.target
    target_names = dataset.target_names

    return documents, targets, target_names
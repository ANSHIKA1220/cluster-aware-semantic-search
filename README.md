# cluster-aware-semantic-search
Cluster-aware semantic search system using embeddings, FAISS vector search, fuzzy clustering, and a semantic cache served via FastAPI.


## Semantic Cache Threshold Exploration

The semantic cache determines whether to reuse a previously computed result based on the cosine similarity between the incoming query embedding and the cached embeddings in the relevant clusters. There is an essential tunable decision here: the similarity threshold.

By adjusting the threshold, we control the precision vs reuse tradeoff of the cache. Here is the behavior of several key thresholds:

- **`0.95` (very strict)**: Cache hits will only occur for exact or nearly identical queries. This heavily reduces incorrect matches but drastically cuts down the hit rate.
- **`0.90` (balanced)**: Captures true semantic similarity without bleeding into unrelated topics. This allows natural variations of phrasing to be cached effectively.
- **`0.80` (aggressive)**: Leads to a very high hit rate (aggressive reuse), but risks returning unrelated results, as moderately related questions are grouped together even if their specific intents differ.

### Example Experiment

Consider the following two queries:
- `"best graphics card for gaming"`
- `"which GPU is best for gaming"`

These naturally phrased queries produce a cosine similarity around **0.92–0.95**. 

By using a **0.90** threshold, the system successfully recognizes these queries as semantically equivalent. This demonstrates optimal semantic cache reuse: the second query securely results in a cache hit from the first without the caching system confusing it for loosely-related hardware questions.

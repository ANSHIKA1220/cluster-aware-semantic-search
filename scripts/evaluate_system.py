import time
import requests

API_URL = "http://127.0.0.1:8000/query"

# Groups of semantically related queries
test_queries = [
    "best graphics card for gaming",
    "which gpu is best for gaming",
    "top gaming graphics cards",

    "best cpu for gaming",
    "good processor for gaming pc",
    "which processor is best for gaming",

    "latest nasa space mission",
    "recent nasa mission",
    "space shuttle launch news"
]


def run_query(query):
    start = time.time()

    response = requests.post(API_URL, json={"query": query})

    latency = time.time() - start

    return response.json(), latency


def main():

    hit_count = 0
    miss_count = 0

    print("\nRunning evaluation...\n")

    for q in test_queries:

        result, latency = run_query(q)

        cache_hit = result.get("cache_hit", False)
        similarity = result.get("similarity_score", None)

        if cache_hit:
            hit_count += 1
        else:
            miss_count += 1

        print(f"Query: {q}")
        print(f"Cache hit: {cache_hit}")

        # Print similarity score if available
        if similarity is not None:
            print(f"Similarity score: {round(similarity, 3)}")
        else:
            print("Similarity score: None")

        print(f"Latency: {round(latency * 1000, 2)} ms")
        print("-" * 40)

    total = hit_count + miss_count

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Total queries: {total}")
    print(f"Cache hits: {hit_count}")
    print(f"Cache misses: {miss_count}")

    if total > 0:
        print(f"Hit rate: {round(hit_count/total, 2)}")


if __name__ == "__main__":
    main()
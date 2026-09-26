import time
import torch

from vec_search.index import FlatVectorIndex


def micro_bench(num_vectors=10000, dim=128, runs=10):
    vectors = torch.randn((num_vectors, dim), dtype=torch.float32)
    index = FlatVectorIndex(vectors)

    query = torch.randn((dim,), dtype=torch.float32)

    # warmup
    index.search(query, k=10)

    times = []
    for _ in range(runs):
        t0 = time.time()
        indices, scores = index.search(query, k=10)
        t1 = time.time()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    print(f"num_vectors={num_vectors}, dim={dim}, avg_latency={avg*1000:.3f} ms")


if __name__ == "__main__":
    micro_bench()

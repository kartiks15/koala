import torch
import pytest

from vec_search.index import FlatVectorIndex


def test_l2_search_returns_smallest_distances():
    vectors = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = FlatVectorIndex(vectors)

    query = torch.tensor([1.1, 0.0], dtype=torch.float32)
    indices, scores = index.search(query, k=2, metric="l2")

    assert indices.tolist() == [1, 2]
    # distances: |1.1-1|=0.1 -> sq=0.01, |2-1.1|=0.9 -> sq=0.81
    assert pytest.approx(scores.tolist(), rel=1e-6) == [0.01, 0.81]

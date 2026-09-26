import torch
import pytest

from vec_search.index import FlatVectorIndex


def test_search_returns_top_k_by_dot_product():
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    index = FlatVectorIndex(vectors)

    query = torch.tensor([1.0, 0.0], dtype=torch.float32)
    indices, scores = index.search(query, k=3)

    assert indices.tolist() == [3, 2, 0]
    assert scores.tolist() == [2.0, 1.0, 1.0]


def test_search_handles_batch_query_as_single_vector():
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    index = FlatVectorIndex(vectors)

    query = torch.tensor([0.0, 1.0], dtype=torch.float32)
    indices, scores = index.search(query, k=2)

    assert indices.tolist() == [1, 0]
    assert scores.tolist() == [1.0, 0.0]

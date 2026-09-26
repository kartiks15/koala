from __future__ import annotations

import torch

from .kernels import dot_product_kernel, l2_distance_kernel


def _has_active_triton_cuda_driver() -> bool:
    try:
        import triton
        from triton.runtime import driver

        _ = triton.runtime.driver.active.get_current_device()
        return True
    except Exception:
        return False


class FlatVectorIndex:
    def __init__(self, vectors: torch.Tensor):
        if not isinstance(vectors, torch.Tensor):
            vectors = torch.tensor(vectors, dtype=torch.float32)

        self.vectors = vectors.to(dtype=torch.float32).contiguous()
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D tensor of shape [num_vectors, dim]")

        self.device = self.vectors.device

    def search(self, query: torch.Tensor, k: int = 10, metric: str = "dot"):
        query_tensor = torch.as_tensor(query, device=self.device, dtype=self.vectors.dtype)
        if query_tensor.ndim == 2:
            if query_tensor.shape[0] != 1:
                raise ValueError("query must be a single vector or a [1, dim] tensor")
            query_tensor = query_tensor.squeeze(0)

        if query_tensor.numel() != self.vectors.shape[1]:
            raise ValueError(
                f"query length {query_tensor.numel()} does not match vector dimension {self.vectors.shape[1]}"
            )

        metric = metric.lower()
        if metric not in ("dot", "l2"):
            raise ValueError("metric must be 'dot' or 'l2'")

        if metric == "dot":
            if self.device.type == "cuda" and _has_active_triton_cuda_driver():
                scores = torch.empty(self.vectors.shape[0], device=self.device, dtype=self.vectors.dtype)
                grid = lambda meta: (self.vectors.shape[0],)
                dot_product_kernel[grid](
                    query_tensor,
                    self.vectors,
                    scores,
                    self.vectors.shape[0],
                    self.vectors.shape[1],
                    BLOCK_SIZE=128,
                )
            else:
                scores = self.vectors @ query_tensor
        else:  # l2
            if self.device.type == "cuda" and _has_active_triton_cuda_driver():
                scores = torch.empty(self.vectors.shape[0], device=self.device, dtype=self.vectors.dtype)
                grid = lambda meta: (self.vectors.shape[0],)
                l2_distance_kernel[grid](
                    query_tensor,
                    self.vectors,
                    scores,
                    self.vectors.shape[0],
                    self.vectors.shape[1],
                    BLOCK_SIZE=128,
                )
            else:
                diff = self.vectors - query_tensor
                scores = torch.sum(diff * diff, dim=1)

        k = min(max(k, 1), self.vectors.shape[0])
        # For dot we want largest, for L2 distance we want smallest
        if metric == "dot":
            topk_scores, topk_indices = torch.topk(scores, k=k, largest=True, sorted=True)
        else:
            topk_scores, topk_indices = torch.topk(scores, k=k, largest=False, sorted=True)
        return topk_indices.cpu(), topk_scores.cpu()

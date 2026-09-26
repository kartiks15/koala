import triton
import triton.language as tl


@triton.jit
def dot_product_kernel(
    query_ptr,
    matrix_ptr,
    output_ptr,
    rows,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)

    if row_idx >= rows:
        return

    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim

        q = tl.load(query_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(matrix_ptr + row_idx * dim + offsets, mask=mask, other=0.0)
        acc += tl.sum(q * v)

    tl.store(output_ptr + row_idx, acc)


@triton.jit
def l2_distance_kernel(
    query_ptr,
    matrix_ptr,
    output_ptr,
    rows,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    if row_idx >= rows:
        return

    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim

        q = tl.load(query_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(matrix_ptr + row_idx * dim + offsets, mask=mask, other=0.0)
        diff = q - v
        acc += tl.sum(diff * diff)

    tl.store(output_ptr + row_idx, acc)

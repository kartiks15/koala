# vec_search

A small learning project that implements a flat in-memory vector search using Triton for GPU kernels and PyTorch for fallbacks and orchestration.

Usage
-----

Install editable (recommended):

```bash
.venv/bin/python -m pip install -e .
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Benchmark (CPU fallback shown):

```bash
PYTHONPATH=. .venv/bin/python bench/bench_flat_index.py
```

Notes
-----
- Triton kernels will only run when a CUDA-backed driver is present. The library falls back to PyTorch CPU/GPU math otherwise.
- `bench/` contains a small micro-benchmark harness.

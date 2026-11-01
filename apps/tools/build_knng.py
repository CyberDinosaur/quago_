#!/usr/bin/env python3
"""
build_knng.py
=============
Brute-force k-NN helper built on FAISS, used to produce the auxiliary inputs QUAGO needs.

Two modes:

  1. Self-query (default) -> KNNG in the ids-only format IndexQuago::LoadKnng reads:
         [k:u32] then, for each point i: [self_id:u32, neighbor_0:u32, ..., neighbor_{k-1}:u32]
     The point itself is excluded from its own neighbor list.

  2. External query (--query) -> combined ground-truth ibin (for search evaluation):
         [n:u32, k:u32, idx:n*k u32, dist:n*k f32]
     which convert_gt_to_knng.py can also turn into a KNNG.

fbin format: [n:u32, dim:u32, n*dim floats].

Requires: numpy, faiss (faiss-cpu or faiss-gpu).

Examples:
    # KNNG over the base (self-query), inner-product metric, GPU if available
    python apps/tools/build_knng.py --base data/base.fbin --k 200 --dist ip --out data/knng.graph

    # Ground truth for a query set (L2)
    python apps/tools/build_knng.py --base data/base.fbin --query data/query.fbin \
        --k 100 --dist l2 --out data/gt.ibin
"""
import argparse
import os
import sys

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("This tool requires FAISS. Install with: pip install faiss-cpu  (or faiss-gpu)")


def read_fbin(path):
    """Load a fbin file ([n:u32, dim:u32, n*dim floats]) as a float32 array."""
    with open(path, "rb") as f:
        n = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=int(n) * int(dim)).reshape(int(n), int(dim))
    print(f"[load] {path}: n={n} dim={dim}")
    return np.ascontiguousarray(data)


def build_index(base, dist, use_gpu):
    dim = base.shape[1]
    if dist in ("ip", "cosine"):
        # Cosine == inner product on L2-normalized vectors.
        faiss.normalize_L2(base)
        index = faiss.IndexFlatIP(dim)
    elif dist == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        sys.exit(f"Unknown --dist '{dist}' (expected l2/ip/cosine)")

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[faiss] using GPU")
        except Exception as e:  # noqa: BLE001 - fall back to CPU on any GPU init failure
            print(f"[faiss] GPU unavailable ({e}); using CPU")
    index.add(base)
    return index


def write_knng(path, ids, k):
    n = ids.shape[0]
    with open(path, "wb") as f:
        np.array([k], dtype=np.uint32).tofile(f)
        for i in range(n):
            np.array([i], dtype=np.uint32).tofile(f)          # self id (skipped by the loader)
            ids[i].astype(np.uint32).tofile(f)                 # k neighbors
    print(f"[write] {path}: {os.path.getsize(path)} bytes (expected {4 + n * (k + 1) * 4})")


def write_gt(path, ids, dists):
    n, k = ids.shape
    with open(path, "wb") as f:
        np.array([n, k], dtype=np.uint32).tofile(f)
        ids.astype(np.uint32).tofile(f)
        dists.astype(np.float32).tofile(f)
    print(f"[write] {path}: {os.path.getsize(path)} bytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Base vectors, fbin format")
    ap.add_argument("--query", default=None, help="Query fbin; if omitted, run self-query to build a KNNG")
    ap.add_argument("--k", type=int, default=200, help="Number of neighbors (default 200)")
    ap.add_argument("--dist", default="ip", choices=["l2", "ip", "cosine"], help="Distance metric")
    ap.add_argument("--out", required=True, help="Output file (KNNG for self-query, GT ibin for --query)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
    args = ap.parse_args()

    base = read_fbin(args.base)
    index = build_index(base, args.dist, use_gpu=not args.cpu)

    if args.query:
        query = read_fbin(args.query)
        if args.dist in ("ip", "cosine"):
            faiss.normalize_L2(query)
        dists, ids = index.search(np.ascontiguousarray(query), args.k)
        write_gt(args.out, ids, dists)
    else:
        # Self-query: fetch k+1 so the point itself can be dropped, keeping k true neighbors.
        dists, ids = index.search(base, args.k + 1)
        n = base.shape[0]
        out = np.empty((n, args.k), dtype=np.int64)
        for i in range(n):
            row = [j for j in ids[i] if j != i][:args.k]
            if len(row) < args.k:  # pad if duplicates left the row short
                row += [row[-1]] * (args.k - len(row)) if row else [0] * args.k
            out[i] = row
        write_knng(args.out, out, args.k)

    print("Done.")


if __name__ == "__main__":
    main()

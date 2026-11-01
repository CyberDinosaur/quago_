#!/usr/bin/env python3
"""
convert_gt_to_knng.py
=====================
Convert a combined ground-truth ibin file into the ids-only KNNG format that
quago_build expects (IndexQuago::LoadKnng).

Input (combined GT ibin):
  [n:u32, k:u32, idx:n*k u32, dist:n*k f32]

Output (KNNG):
  [k:u32]
  then, for each point i in [0, n):
    [self_id:u32, neighbor_0:u32, ..., neighbor_{k-1}:u32]

Usage:
    python apps/tools/convert_gt_to_knng.py knn.ibin knng.graph
"""
import argparse
import os

import numpy as np


def read_combined_gt(path):
    """Read combined GT ibin: [n:u32, k:u32, idx:n*k u32, dist:n*k f32]."""
    with open(path, 'rb') as f:
        n = np.fromfile(f, dtype=np.uint32, count=1)[0]
        k = np.fromfile(f, dtype=np.uint32, count=1)[0]
        idx = np.fromfile(f, dtype=np.uint32, count=n * k).reshape(n, k)
        dist = np.fromfile(f, dtype=np.float32, count=n * k).reshape(n, k)
    return idx, dist


def write_knng(path, idx):
    """Write KNNG format: [k:u32, then for each point: [self:u32, k neighbors:u32]]."""
    n, k = idx.shape
    with open(path, 'wb') as f:
        np.array([k], dtype=np.uint32).tofile(f)
        for i in range(n):
            np.array([i], dtype=np.uint32).tofile(f)  # self id
            idx[i].astype(np.uint32).tofile(f)         # k neighbors
    size = os.path.getsize(path)
    expected = 4 + n * (k + 1) * 4
    print(f"Written {path}: {size} bytes (expected {expected})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input combined GT ibin file")
    ap.add_argument("output", help="Output KNNG file")
    args = ap.parse_args()

    print(f"Reading {args.input} ...")
    idx, _ = read_combined_gt(args.input)
    n, k = idx.shape
    print(f"Shape: {n} x {k}")

    print(f"Writing {args.output} ...")
    write_knng(args.output, idx)
    print("Done.")


if __name__ == "__main__":
    main()


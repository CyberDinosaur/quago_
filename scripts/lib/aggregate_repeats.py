#!/usr/bin/env python3
import argparse
import sys

import pandas as pd

NUMERIC = ["QPS", "Avg_hops", "Avg_cmps", "Mean_latency(ms)", "Recall@10"]
TIMING = {"QPS", "Mean_latency(ms)"}

def main():
    ap = argparse.ArgumentParser(
        description="Collapse repeated L-sweep passes into one curve, keeping the fastest pass per L.")
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--drop-warmup", type=int, default=0,
                    help="drop this many leading rows (cache warmup passes)")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    if args.drop_warmup:
        df = df.iloc[args.drop_warmup:].reset_index(drop=True)
    if df.empty:
        sys.exit(f"no rows left after dropping {args.drop_warmup} warmup rows in {args.raw}")

    passes = df.groupby("L_queue").size()
    print(f"[agg] {args.raw}: {len(df)} rows, {passes.max()} pass(es) per L")

    for col in NUMERIC:
        if col in TIMING or col not in df.columns:
            continue
        spread = df.groupby("L_queue")[col].agg(lambda s: s.max() - s.min())
        worst = spread.max()
        rel = worst / df[col].abs().max() if df[col].abs().max() else 0.0
        flag = "OK" if rel < 1e-6 else "VARIES"
        print(f"[agg]   {col:<18} max spread {worst:.4g} ({flag})")

    idx = df.groupby("L_queue")["QPS"].idxmax()
    best = df.loc[idx].sort_values("L_queue").reset_index(drop=True)

    def _gaps(v):
        v = sorted(v, reverse=True)
        return None if len(v) < 2 else ((v[0] - v[1]) / v[0] * 100, (v[0] - v[-1]) / v[0] * 100)

    gaps = df.groupby("L_queue")["QPS"].apply(_gaps).dropna()
    top2 = pd.Series([g[0] for g in gaps])
    full = pd.Series([g[1] for g in gaps])
    verdict = "OK" if top2.median() < 8.0 else "UNRELIABLE (no clean pass; raise QUAGO_REPEATS)"
    print(f"[agg]   QPS top2 gap: median {top2.median():.1f}%, max {top2.max():.1f}%  -> {verdict}")
    print(f"[agg]   QPS max-min : median {full.median():.1f}%, max {full.max():.1f}%  (one slow pass is fine)")
    mono = (best["QPS"].diff().dropna() <= 0).all()
    print(f"[agg]   QPS monotonic in L after aggregation: {'yes' if mono else 'NO'}")

    best.to_csv(args.out, index=False)
    print(f"[agg] wrote {args.out} ({len(best)} rows)")

if __name__ == "__main__":
    main()

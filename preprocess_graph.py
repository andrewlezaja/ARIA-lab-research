"""
preprocess_dataset.py
=====================
Steps:
  1. Load and back up datasets/<ds>/processed/graph.edgelist
  2. Randomly sample C_SIZE edges as candidate set C
  3. Remove C from graph -> refined graph G'
  4. Select K seed nodes on G':
       - highdeg: greedy MIA-based influence maximization (paper method)
       - random:  uniform random sampling
  5. Save candidate files and seed files

Usage:
    python preprocess_dataset.py --dataset datasets/google_web \\
        --c-size 2000 --Ks 50 100 150 200 --trials 3 --rng-seed 123
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import networkx as nx

# -----------------------------------------------------------------------------
# MIA influence
# -----------------------------------------------------------------------------
try:
    from influence import mia_influence
except ImportError:
    _HERE = Path(__file__).resolve().parent
    for _candidate in [_HERE / "src", _HERE]:
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
    try:
        from influence import mia_influence  # type: ignore
    except ImportError:
        raise ImportError(
            "Could not import mia_influence from influence.py. "
            "Make sure influence.py is in the repo root or src/."
        )


# -----------------------------------------------------------------------------
# Graph loading
# -----------------------------------------------------------------------------

def load_graph(path: str) -> nx.DiGraph:
    """Load a plain 'u v' edgelist into a DiGraph, skipping comments/self-loops."""
    G = nx.DiGraph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                G.add_edge(u, v)
    return G


# -----------------------------------------------------------------------------
# Candidate edge sampling
# -----------------------------------------------------------------------------

def sample_candidates(G: nx.DiGraph, c_size: int, rng: random.Random) -> list:
    """
    Randomly sample c_size edges from G.
    Per paper Section 6.1.3: purely random sampling from existing edges.
    """
    edges = list(G.edges())
    if len(edges) < c_size:
        raise ValueError(
            f"Graph has only {len(edges)} edges but c_size={c_size}. "
            "Reduce --c-size or use a larger graph."
        )
    return rng.sample(edges, c_size)


# -----------------------------------------------------------------------------
# Seed selection
# -----------------------------------------------------------------------------

def _assign_wc_weights(G: nx.DiGraph) -> nx.DiGraph:
    """
    Return a copy of G with WC weights: omega(u, v) = 1 / in_degree(v).
    If all edges already have weights, returns G as-is.
    """
    if all("weight" in G[u][v] for u, v in G.edges()):
        return G
    G_w = G.copy()
    in_deg = dict(G_w.in_degree())
    for u, v in G_w.edges():
        G_w[u][v]["weight"] = 1.0 / max(in_deg[v], 1)
    return G_w


def select_seeds_mia(
    G: nx.DiGraph,
    K: int,
    theta: float = 0.001,
) -> list:
    """
    Greedily selects K seed nodes, each time picking the node with
    the highest marginal influence gain under the MIA model:

        s* = argmax_{v not in S} [ mia_influence(G, S + {v}) - mia_influence(G, S) ]

    Parameters:
        G: refined DiGraph (WC weights assigned internally)
        K: number of seeds to select
        theta: MIA influence threshold (paper default: 0.001)

    Returns:
        List of K seed node IDs in selection order.
    """
    G_w = _assign_wc_weights(G)
    valid_nodes = [n for n in G_w.nodes() if G_w.degree(n) > 0]

    if len(valid_nodes) < K:
        raise ValueError(
            f"Graph has only {len(valid_nodes)} non-isolated nodes but K={K}."
        )

    S: list = []
    remaining = set(valid_nodes)

    print(f"    MIA greedy selection: K={K}, |V_valid|={len(valid_nodes)}")

    for step in range(K):
        best_node = None
        best_gain = -1.0
        current_spread = mia_influence(G_w, S, theta=theta) if S else 0.0

        for v in remaining:
            spread = mia_influence(G_w, S + [v], theta=theta)
            gain = spread - current_spread
            if gain > best_gain:
                best_gain = gain
                best_node = v

        S.append(best_node)
        remaining.remove(best_node)

        if (step + 1) % 10 == 0 or (step + 1) == K:
            print(f"    Step {step+1}/{K}: node={best_node}  gain={best_gain:.4f}")

    return S


def select_seeds_random(
    G: nx.DiGraph,
    K: int,
    rng: random.Random,
    trial: int,
) -> list:
    """
    Randomly select K seed nodes from non-isolated nodes in the refined graph.
    Each trial uses a deterministic but distinct RNG state.
    """
    rng_trial = random.Random(rng.randint(0, 2**32) + trial)
    nodes = [n for n in G.nodes() if G.degree(n) > 0]
    if len(nodes) < K:
        raise ValueError(
            f"Graph has only {len(nodes)} non-isolated nodes but K={K}."
        )
    return rng_trial.sample(nodes, K)

def select_seeds_highdeg(
    G: nx.DiGraph,
    K: int,
) -> list:
    """
    Fast seed selection: pick K nodes with highest out-degree.
    Used when --fast flag is set.
    """
    nodes_by_degree = sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)
    non_isolated = [n for n in nodes_by_degree if G.degree(n) > 0]
    if len(non_isolated) < K:
        raise ValueError(
            f"Graph has only {len(non_isolated)} non-isolated nodes but K={K}."
        )
    return non_isolated[:K]

# -----------------------------------------------------------------------------
# File writers
# -----------------------------------------------------------------------------

def write_edgelist(G: nx.DiGraph, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    print(f"  Saved graph:      {path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def write_candidates(candidates: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  Skipping candidates (exists): {path}")
        return
    with open(path, "w") as f:
        for u, v in candidates:
            f.write(f"{u} {v}\n")
    print(f"  Saved candidates: {path}  ({len(candidates)} edges)")


def write_seeds(seeds: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print(f"  Skipping seeds    (exists): {path}")
        return
    with open(path, "w") as f:
        for s in seeds:
            f.write(f"{s}\n")
    print(f"  Saved seeds:      {path}  ({len(seeds)} nodes)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a BIM dataset: sample candidates, remove from graph, select seeds."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset root (e.g. datasets/google_web). "
             "Reads from <dataset>/processed/graph.edgelist."
    )
    parser.add_argument("--c-size",   type=int, default=2000,
                        help="Candidate set size (default: 2000)")
    parser.add_argument("--Ks",       type=int, nargs="+", default=[50, 100, 150, 200],
                        help="Seed set sizes (default: 50 100 150 200)")
    parser.add_argument("--trials",   type=int, default=3,
                        help="Number of trials per K (default: 3)")
    parser.add_argument("--rng-seed", type=int, default=42,
                        help="Master RNG seed (default: 42)")
    parser.add_argument("--theta",    type=float, default=0.001,
                        help="MIA influence threshold (default: 0.001)")
    parser.add_argument("--fast", action="store_true",
                        help="Use high-degree heuristic for seed selection instead of MIA greedy (much faster)")
    args = parser.parse_args()

    dataset_root  = Path(args.dataset)
    processed_dir = dataset_root / "processed"
    graph_path    = processed_dir / "graph.edgelist"
    backup_path   = processed_dir / "graph.edgelist.backup"

    if not graph_path.exists():
        raise FileNotFoundError(f"Could not find graph at {graph_path}")

    print(f"\n{'='*60}")
    print(f"  BIM Preprocessing")
    print(f"  Dataset:  {dataset_root}")
    print(f"  C size:   {args.c_size}")
    print(f"  Ks:       {args.Ks}")
    print(f"  Trials:   {args.trials}")
    print(f"  RNG seed: {args.rng_seed}")
    print(f"  Theta:    {args.theta}")
    print(f"{'='*60}\n")

    # Step 1: Load and back up
    print("Step 1: Loading and backing up processed graph...")
    G_raw = load_graph(str(graph_path))
    print(f"  Loaded: {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    if not backup_path.exists():
        shutil.copy2(str(graph_path), str(backup_path))
        print(f"  Backup: {backup_path}")
    else:
        print(f"  Backup already exists, skipping: {backup_path}")

    # Step 2: Sample candidates
    print("\nStep 2: Sampling candidate edges...")
    rng = random.Random(args.rng_seed)
    candidates = sample_candidates(G_raw, args.c_size, rng)
    print(f"  Sampled {len(candidates)} candidate edges")

    # Step 3: Build refined graph
    print("\nStep 3: Building refined graph (removing candidates)...")
    G_refined = G_raw.copy()
    G_refined.remove_edges_from(candidates)
    print(f"Refined: {G_refined.number_of_nodes()} nodes, {G_refined.number_of_edges()} edges")

    write_edgelist(G_refined, str(graph_path))

    print("\nSaving candidate files...")
    cand_dir = processed_dir / "candidates"
    for K in args.Ks:
        for t in range(args.trials):
            for seed_type in ["highdeg", "random"]:
                write_candidates(
                    candidates,
                    str(cand_dir / f"C_{args.c_size}_seed{seed_type}_K{K}_t{t}.txt")
                )

    # Step 4: Select seeds on refined graph
    print("\nStep 4: Selecting seed nodes on refined graph...")
    seeds_dir = processed_dir / "seeds"

    # MIA highdeg seeds — greedy, deterministic, same across trials
    print("\n  [highdeg] Seed selection...")
    max_K = max(args.Ks)
    if args.fast:
        print(f"  Using fast high-degree heuristic (--fast flag set)...")
        seeds_hd_full = select_seeds_highdeg(G_refined, max_K)
    else:
        print(f"  Running MIA for max K={max_K} then slicing for smaller Ks...")
        seeds_hd_full = select_seeds_mia(G_refined, max_K, theta=args.theta)
    
    for K in args.Ks:
        seeds_hd = seeds_hd_full[:K]
        for t in range(args.trials):
            write_seeds(seeds_hd, str(seeds_dir / f"highdeg_K{K}_t{t}.txt"))

    # Random seeds — different per trial
    print("\n[random] Random seed selection...")
    for K in args.Ks:
        for t in range(args.trials):
            seeds_rd = select_seeds_random(G_refined, K, rng, trial=t)
            write_seeds(seeds_rd, str(seeds_dir / f"random_K{K}_t{t}.txt"))

    print(f"\n{'='*60}")
    print(f"Preprocessing complete.")
    print(f"Output: {processed_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

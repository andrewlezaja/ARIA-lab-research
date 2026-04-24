import json
import os
import sys
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------------------
# torch and DRL Graph are imported lazily so that graph_loader can be used
# in environments without torch installed (e.g. the baseline Docker image).
# They are only required when nx_to_drl_graph() or load_dataset() with
# load_drl=True is called.
# ---------------------------------------------------------------------------
_torch = None
_Graph = None

def _require_torch():
    """Import torch and the DRL Graph class on first use."""
    global _torch, _Graph
    if _torch is None:
        try:
            import torch as _t
            _torch = _t
        except ImportError:
            raise ImportError(
                "torch is required to build DRL Graph objects. "
                "Install it with: pip install torch\n"
                "Or use load_dataset(..., load_drl=False) to skip DRL graph creation."
            )
    if _Graph is None:
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        _DRL_SRC   = _REPO_ROOT / "drl_code" / "src"
        if str(_DRL_SRC) not in sys.path:
            sys.path.insert(0, str(_DRL_SRC))
        from utils.graph_utils import Graph
        _Graph = Graph
    return _torch, _Graph


# ===========================================================================
# 1.  Low-level file parsers  (extend src/loaders.py primitives)
# ===========================================================================

def _parse_edgelist(path: str) -> list[tuple[int, int]]:
    """
    Read a plain 'u v' edgelist file, skipping blank lines and #-comments.
    Returns a list of (src, dst) int tuples.
    Self-loops are silently dropped.
    """
    edges: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges.append((u, v))
    return edges


def _parse_int_list(path: str) -> list[int]:
    """Read one integer per line; skip blanks and #-comments."""
    out: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line))
    return out


# ===========================================================================
# 2.  NetworkX graph builder
# ===========================================================================

def build_nx_graph(edge_pairs: list[tuple[int, int]]) -> nx.DiGraph:

    G = nx.DiGraph()
    G.add_edges_from(edge_pairs)

    # Assign WC weights
    in_deg = dict(G.in_degree())
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 / in_deg[v] if in_deg[v] > 0 else 0.0

    return G


# ===========================================================================
# 3.  DRL Graph object builder
# ===========================================================================

def nx_to_drl_graph(G: nx.DiGraph, device=None):

    torch, Graph = _require_torch()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes = set(G.nodes())

    edges: dict[tuple[int, int], float] = {}
    children: dict[int, set] = {n: set() for n in nodes}
    parents:  dict[int, set] = {n: set() for n in nodes}

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0.0)
        edges[(u, v)] = w
        children[u].add(v)
        parents[v].add(u)

    return Graph(nodes, edges, children, parents, device=device)


# ===========================================================================
# 4.  Seed / candidate directory loaders
# ===========================================================================

def load_seeds_dir(seeds_dir: str) -> dict[str, list[int]]:
    """
    Load every seed file in a directory.

    Returns a dict keyed by stem filename, e.g.:
        { "highdeg_K10_t0": [3, 17, 42, ...], ... }
    """
    seeds: dict[str, list[int]] = {}
    p = Path(seeds_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Seeds directory not found: {seeds_dir}")
    for f in sorted(p.glob("*.txt")):
        seeds[f.stem] = _parse_int_list(str(f))
    return seeds


def load_candidates_dir(candidates_dir: str) -> dict[str, list[tuple[int, int]]]:
    """
    Load every candidate-edge file in a directory.

    Returns a dict keyed by stem filename, e.g.:
        { "C_2000_seedhighdeg_K10_t0": [(u1,v1), (u2,v2), ...], ... }
    """
    candidates: dict[str, list[tuple[int, int]]] = {}
    p = Path(candidates_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Candidates directory not found: {candidates_dir}")
    for f in sorted(p.glob("*.txt")):
        candidates[f.stem] = _parse_edgelist(str(f))
    return candidates


# ===========================================================================
# 5.  Unified entry point
# ===========================================================================

def load_dataset(
    dataset_root: str,
    device=None,
    load_seeds: bool = True,
    load_candidates: bool = True,
    load_drl: bool = True,
) -> dict:

    root = Path(dataset_root)

    # -- graph -----------------------------------------------------------------
    graph_path = root / "graph.edgelist"
    if not graph_path.exists():
        raise FileNotFoundError(f"graph.edgelist not found in {root}")

    edge_pairs = _parse_edgelist(str(graph_path))
    nx_graph   = build_nx_graph(edge_pairs)
    drl_graph  = nx_to_drl_graph(nx_graph, device=device) if load_drl else None

    # -- seeds -----------------------------------------------------------------
    seeds: dict[str, list[int]] = {}
    if load_seeds:
        seeds_dir = root / "seeds"
        if seeds_dir.is_dir():
            seeds = load_seeds_dir(str(seeds_dir))

    # -- candidates ------------------------------------------------------------
    candidates: dict[str, list[tuple[int, int]]] = {}
    if load_candidates:
        cand_dir = root / "candidates"
        if cand_dir.is_dir():
            candidates = load_candidates_dir(str(cand_dir))

    # -- stats -----------------------------------------------------------------
    stats: dict = {}
    stats_path = root / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)

    return {
        "nx_graph":   nx_graph,
        "drl_graph":  drl_graph,
        "seeds":      seeds,
        "candidates": candidates,
        "stats":      stats,
    }


# ===========================================================================
# 6.  Convenience: match a seed key to its candidate key
# ===========================================================================

def get_matching_candidates(
    seed_key: str,
    candidates: dict[str, list[tuple[int, int]]],
    num_candidates: int = 2000,
) -> list[tuple[int, int]] | None:
    """
    Given a seed key like "highdeg_K10_t0", find the matching candidate list
    "C_2000_seedhighdeg_K10_t0" (or whatever num_candidates is).

    Returns the candidate list, or None if no match is found.
    """
    target = f"C_{num_candidates}_seed{seed_key}"
    return candidates.get(target, None)

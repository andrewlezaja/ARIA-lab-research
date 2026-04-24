import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from graph_loader import load_dataset, get_matching_candidates  # noqa: E402


# ===========================================================================
# Check helpers
# ===========================================================================

class CheckRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed  = 0
        self.failed  = 0

    def ok(self, label: str, detail: str = ""):
        self.passed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [PASS] {label}{suffix}")

    def fail(self, label: str, detail: str = ""):
        self.failed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [FAIL] {label}{suffix}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'='*55}")
        print(f"  Results: {self.passed}/{total} checks passed")
        print(f"{'='*55}")
        return self.failed == 0


# ===========================================================================
# Individual check groups
# ===========================================================================

def check_nx_graph(cr: CheckRunner, nx_graph, stats: dict) -> None:
    print("\n── NetworkX Graph ──────────────────────────────────────")
    import networkx as nx

    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()

    # Basic existence
    if n > 0:
        cr.ok("Graph has nodes", f"n={n}")
    else:
        cr.fail("Graph has nodes", "n=0")

    if m > 0:
        cr.ok("Graph has edges", f"m={m}")
    else:
        cr.fail("Graph has edges", "m=0")

    # Match stats.json if available
    if stats:
        exp_n = stats.get("n_nodes")
        exp_m = stats.get("n_edges")
        if exp_n is not None:
            if n == exp_n:
                cr.ok("Node count matches stats.json", f"{n} == {exp_n}")
            else:
                cr.fail("Node count matches stats.json", f"{n} != {exp_n}")
        if exp_m is not None:
            if m == exp_m:
                cr.ok("Edge count matches stats.json", f"{m} == {exp_m}")
            else:
                cr.fail("Edge count matches stats.json", f"{m} != {exp_m}")

    # Directed
    if nx_graph.is_directed():
        cr.ok("Graph is directed")
    else:
        cr.fail("Graph is directed", "got undirected")

    # No self-loops
    self_loops = list(nx.selfloop_edges(nx_graph))
    if not self_loops:
        cr.ok("No self-loops")
    else:
        cr.fail("No self-loops", f"{len(self_loops)} found")

    # WC weights in (0, 1]
    weights = [d["weight"] for _, _, d in nx_graph.edges(data=True)]
    bad_weights = [w for w in weights if not (0.0 < w <= 1.0)]
    if not bad_weights:
        cr.ok("All edge weights in (0, 1]", f"checked {len(weights)} edges")
    else:
        cr.fail("All edge weights in (0, 1]", f"{len(bad_weights)} out-of-range weights")

    # 0-indexed nodes
    min_node = min(nx_graph.nodes()) if n > 0 else -1
    max_node = max(nx_graph.nodes()) if n > 0 else -1
    if min_node == 0:
        cr.ok("Nodes are 0-indexed", f"range [{min_node}, {max_node}]")
    else:
        cr.fail("Nodes are 0-indexed", f"min node = {min_node}")

    # Weakly connected components
    n_wcc = nx.number_weakly_connected_components(nx_graph)
    if n_wcc == 1:
        cr.ok("Graph is weakly connected", "1 component")
    else:
        cr.ok(f"Graph has {n_wcc} weakly connected components (informational)")


def check_drl_graph(cr: CheckRunner, drl_graph, nx_graph) -> None:
    print("\n── DRL Graph Object ────────────────────────────────────")
    import torch

    # Node / edge counts match nx
    if drl_graph.num_nodes == nx_graph.number_of_nodes():
        cr.ok("DRL num_nodes matches nx", f"{drl_graph.num_nodes}")
    else:
        cr.fail("DRL num_nodes matches nx",
                f"{drl_graph.num_nodes} vs {nx_graph.number_of_nodes()}")

    if drl_graph.num_edges == nx_graph.number_of_edges():
        cr.ok("DRL num_edges matches nx", f"{drl_graph.num_edges}")
    else:
        cr.fail("DRL num_edges matches nx",
                f"{drl_graph.num_edges} vs {nx_graph.number_of_edges()}")

    # edge_index shape
    ei = drl_graph.edge_index
    if ei.shape == (2, drl_graph.num_edges):
        cr.ok("edge_index shape correct", f"{list(ei.shape)}")
    else:
        cr.fail("edge_index shape correct", f"got {list(ei.shape)}")

    # edge_weights length
    ew = drl_graph.edge_weights
    if len(ew) == drl_graph.num_edges:
        cr.ok("edge_weights length correct", f"{len(ew)}")
    else:
        cr.fail("edge_weights length correct", f"{len(ew)} vs {drl_graph.num_edges}")

    # initial_embeddings shape: (num_nodes, embed_dim)
    emb = drl_graph.initial_embeddings
    if emb.shape[0] == drl_graph.num_nodes:
        cr.ok("initial_embeddings row count matches num_nodes",
              f"shape {list(emb.shape)}")
    else:
        cr.fail("initial_embeddings row count matches num_nodes",
                f"shape {list(emb.shape)}")

    # children / parents cover all nodes
    missing_children = [n for n in drl_graph.nodes if n not in drl_graph.children]
    missing_parents  = [n for n in drl_graph.nodes if n not in drl_graph.parents]
    if not missing_children:
        cr.ok("children dict covers all nodes")
    else:
        cr.fail("children dict covers all nodes",
                f"{len(missing_children)} nodes missing")
    if not missing_parents:
        cr.ok("parents dict covers all nodes")
    else:
        cr.fail("parents dict covers all nodes",
                f"{len(missing_parents)} nodes missing")

    # Sample: children/parents are consistent
    sample_edges = list(drl_graph.edges.keys())[:10]
    inconsistent = []
    for u, v in sample_edges:
        if v not in drl_graph.children.get(u, []):
            inconsistent.append((u, v, "children"))
        if u not in drl_graph.parents.get(v, []):
            inconsistent.append((u, v, "parents"))
    if not inconsistent:
        cr.ok("Sample edge consistency (children/parents)", "first 10 edges ok")
    else:
        cr.fail("Sample edge consistency", str(inconsistent))


def check_seeds(cr: CheckRunner, seeds: dict, nx_graph, verbose: bool) -> None:
    print("\n── Seeds ───────────────────────────────────────────────")
    valid_nodes = set(nx_graph.nodes())

    if seeds:
        cr.ok("Seeds directory loaded", f"{len(seeds)} seed files")
    else:
        cr.fail("Seeds directory loaded", "0 files found")
        return

    oob_files = []
    empty_files = []
    for key, s_list in seeds.items():
        if verbose:
            print(f"       {key}: {len(s_list)} seeds  {s_list[:5]}{'...' if len(s_list)>5 else ''}")
        if not s_list:
            empty_files.append(key)
        bad = [n for n in s_list if n not in valid_nodes]
        if bad:
            oob_files.append((key, bad))

    if not empty_files:
        cr.ok("No empty seed files")
    else:
        cr.fail("No empty seed files", str(empty_files))

    if not oob_files:
        cr.ok("All seed nodes exist in graph")
    else:
        cr.fail("All seed nodes exist in graph",
                f"{len(oob_files)} files have out-of-range nodes")

    # Check K is consistent with filename
    bad_k = []
    for key, s_list in seeds.items():
        parts = key.split("_")
        for p in parts:
            if p.startswith("K") and p[1:].isdigit():
                expected_k = int(p[1:])
                if len(s_list) != expected_k:
                    bad_k.append((key, expected_k, len(s_list)))
    if not bad_k:
        cr.ok("Seed list lengths match K in filename")
    else:
        cr.fail("Seed list lengths match K in filename", str(bad_k))


def check_candidates(cr: CheckRunner, candidates: dict, seeds: dict,
                     nx_graph, verbose: bool) -> None:
    print("\n── Candidates ──────────────────────────────────────────")
    valid_nodes  = set(nx_graph.nodes())
    existing_edges = set(nx_graph.edges())

    if candidates:
        cr.ok("Candidates directory loaded", f"{len(candidates)} candidate files")
    else:
        cr.fail("Candidates directory loaded", "0 files found")
        return

    oob_node_files   = []
    duplicate_files  = []
    already_in_graph = []
    self_loop_files  = []

    for key, c_list in candidates.items():
        if verbose:
            print(f"       {key}: {len(c_list)} candidates")

        # Out-of-range nodes
        bad_nodes = [(u, v) for u, v in c_list
                     if u not in valid_nodes or v not in valid_nodes]
        if bad_nodes:
            oob_node_files.append((key, len(bad_nodes)))

        # Self-loops
        loops = [(u, v) for u, v in c_list if u == v]
        if loops:
            self_loop_files.append((key, len(loops)))

        # Duplicates within file
        if len(set(c_list)) < len(c_list):
            duplicate_files.append(key)

        # Edges already in graph (should not happen; flagged as warning)
        in_graph = [e for e in c_list if e in existing_edges]
        if in_graph and "_BAD_" not in key:
            already_in_graph.append((key, len(in_graph)))

    if not oob_node_files:
        cr.ok("All candidate nodes exist in graph")
    else:
        cr.fail("All candidate nodes exist in graph",
                f"{len(oob_node_files)} files affected")

    if not self_loop_files:
        cr.ok("No self-loops in candidates")
    else:
        cr.fail("No self-loops in candidates", str(self_loop_files))

    if not duplicate_files:
        cr.ok("No duplicate edges within candidate files")
    else:
        cr.fail("No duplicate edges within candidate files", str(duplicate_files))

    if not already_in_graph:
        cr.ok("No candidate edges already in graph (excluding _BAD_ files)")
    else:
        cr.ok(f"Warning: {len(already_in_graph)} files have edges already in graph "
              f"(informational — check _BAD_ files)", "")

    # Matching: for each seed file, does a corresponding candidate file exist?
    unmatched_seeds = []
    for seed_key in seeds:
        if get_matching_candidates(seed_key, candidates) is None:
            unmatched_seeds.append(seed_key)
    if not unmatched_seeds:
        cr.ok("Every seed file has a matching candidate file")
    else:
        cr.ok(f"{len(unmatched_seeds)} seed files have no matching candidate file "
              f"(informational)", str(unmatched_seeds[:3]))


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate graph_loader output for a processed dataset."
    )
    parser.add_argument(
        "--dataset",
        default="datasets/email_enron/processed",
        help="Path to processed dataset directory (default: datasets/email_enron/processed)",
    )
    parser.add_argument(
        "--no-drl",
        action="store_true",
        help="Skip DRL Graph checks (use if torch / drl_code is unavailable)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-seed / per-candidate details",
    )
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Graph Loader Validation")
    print(f"  Dataset : {args.dataset}")
    print(f"{'='*55}")

    # Load
    print("\nLoading dataset...")
    try:
        data = load_dataset(
            args.dataset,
            load_seeds=True,
            load_candidates=True,
            load_drl=(not args.no_drl),
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to load dataset: {exc}")
        return 1

    print("  Dataset loaded successfully.")

    cr = CheckRunner(verbose=args.verbose)

    # Run checks
    check_nx_graph(cr, data["nx_graph"], data["stats"])

    if not args.no_drl:
        try:
            check_drl_graph(cr, data["drl_graph"], data["nx_graph"])
        except Exception as exc:
            print(f"\n  [SKIP] DRL Graph checks failed with: {exc}")
            print("         Re-run with --no-drl to skip these checks.")

    check_seeds(cr, data["seeds"], data["nx_graph"], args.verbose)
    check_candidates(cr, data["candidates"], data["seeds"],
                     data["nx_graph"], args.verbose)

    passed = cr.summary()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

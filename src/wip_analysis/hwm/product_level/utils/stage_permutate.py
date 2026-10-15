import re
import threading
from collections.abc import Mapping, Sequence
from heapq import heappop, heappush
from typing import Optional

import numpy as np
import numpy.typing as npt
from loguru import logger
from ortools.sat.python import cp_model

from wip_analysis.utils.env_num import env_num
from wip_analysis.utils.logging import format_sequence

_STAGE_KEY_SPLIT_RE = re.compile(r"(\d+)")


def _cp_sat_log_callback(msg: str) -> None:
    if not msg or not msg.strip():
        return

    def format_cp_sat_lines() -> str:
        lines = (part.rstrip() for part in msg.split("\n"))
        return "\n".join(f"CP-SAT | {line}" for line in lines if line)

    logger.opt(lazy=True).trace("{}", format_cp_sat_lines)


def _natural_stage_key(stage: str) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (0, int(part), part) if part.isdigit() else (1, 0, part)
        for part in _STAGE_KEY_SPLIT_RE.split(stage)
        if part
    )


def get_primary_path(
    seqs: Mapping[str, Sequence[str | None]],
    sol_timeout: float | None = None,
) -> list[str]:
    """
    Get the primary recipe path from the given sequences.
    Scope: recipe sequences of **one single cluster** of a `product_id`.

    Args:
        seqs: A mapping from lot IDs to their recipe sequences.
        sol_timeout (float, optional): The timeout for the solver in seconds. Defaults to 180.

    Returns:
        List[str]: The primary recipe path.
    """
    n_lots = len(seqs)
    if n_lots == 0:
        logger.warning("Input no seqs!")
        return []

    sanitized_seqs: dict[str, list[str]] = {}
    for lot_id, seq in seqs.items():
        filtered_seq = [stage for stage in seq if stage is not None and stage != ""]
        sanitized_seqs[lot_id] = filtered_seq

    unique_stages = sorted(
        {stage for seq in sanitized_seqs.values() for stage in seq},
        key=_natural_stage_key,
    )
    stoi = {s: i for i, s in enumerate(unique_stages)}
    mat_size = len(unique_stages)
    logger.debug("Num of lots: {}, Num of unique stages: {}", n_lots, mat_size)

    pairorder_cnt_mat = np.zeros((mat_size, mat_size), dtype=int)
    triu_idx_cache: dict[int, tuple[npt.NDArray, npt.NDArray]] = {}

    for seq in sanitized_seqs.values():
        if len(seq) < 2:
            continue

        indexed_seq = [stoi[stage] for stage in seq]
        if len(indexed_seq) == 2:
            idx1, idx2 = indexed_seq
            if idx1 != idx2:
                pairorder_cnt_mat[idx2, idx1] += 1
            continue

        if len(set(indexed_seq)) == len(indexed_seq):
            idx_arr = np.array(indexed_seq, dtype=int)
            tri_i, tri_j = triu_idx_cache.setdefault(
                len(indexed_seq),
                np.triu_indices(len(indexed_seq), k=1),
            )
            pairorder_cnt_mat[idx_arr[tri_j], idx_arr[tri_i]] += 1
            continue

        seen_pairs_in_seq: set[tuple[int, int]] = set()
        for i, idx1 in enumerate(indexed_seq[:-1]):
            for idx2 in indexed_seq[i + 1 :]:
                if idx1 != idx2:
                    seen_pairs_in_seq.add((idx1, idx2))

        for idx1, idx2 in seen_pairs_in_seq:
            pairorder_cnt_mat[idx2, idx1] += 1

    obj, permutation = permutation_opt(pairorder_cnt_mat, timeout=sol_timeout)

    primary_sequence: list[str] = []
    if permutation is not None:
        if obj is not None and obj > 0:
            logger.warning(f"obj > 0 : {obj} violations out of {n_lots} lots")
        primary_sequence = [unique_stages[i] for i in permutation]
        logger.opt(lazy=True).debug(
            "Solved primary path: {}",
            lambda: format_sequence(primary_sequence),
        )
        return primary_sequence
    else:
        logger.error("Failed to find feasible solution!")
        return primary_sequence


max_parallel_solvers = threading.Semaphore(env_num("WIP_ANALYSIS_MAX_PARALLEL_SOLVERS", 6))


def _zero_objective_permutation_if_acyclic(
    reference_order: npt.NDArray,
    rows: npt.NDArray,
    cols: npt.NDArray,
) -> list[int] | None:
    """Return a zero-cost topological order when all positive constraints form a DAG.

    `reference_order` is only used to break ties among items that are currently
    unconstrained by observed pair-order evidence.
    """
    n = len(reference_order)
    reference_rank = {int(item_idx): rank for rank, item_idx in enumerate(reference_order)}
    adjacency: list[list[int]] = [[] for _ in range(n)]
    indegree = [0] * n

    for i, j in zip(rows, cols, strict=True):
        parent = int(j)
        child = int(i)
        if parent == child:
            continue
        # Positive p[i, j] is avoided by ordering j before i.
        adjacency[parent].append(child)
        indegree[child] += 1

    queue: list[tuple[int, int]] = []
    for item_idx, degree in enumerate(indegree):
        if degree == 0:
            heappush(queue, (reference_rank[item_idx], item_idx))

    order: list[int] = []
    while queue:
        _, item_idx = heappop(queue)
        order.append(item_idx)
        for child in adjacency[item_idx]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(queue, (reference_rank[child], child))

    if len(order) != n:
        return None
    return order


def _ensure_violation_var(
    model: cp_model.CpModel,
    x: list[cp_model.IntVar],
    b: list[list[cp_model.BoolVarT | None]],
    int_matrix: npt.NDArray,
    objective_terms: list[cp_model.LinearExpr],
    j: int,
    i: int,
    idx2val: dict[int, int],
) -> cp_model.BoolVarT:
    """Return ``b[j,i]`` (1 iff ``x[j] > x[i]``), creating a zero-cost aux var if needed."""
    existing = b[j][i]
    if existing is not None:
        return existing

    is_violation = model.NewBoolVar(f"b_{j}_{i}")
    b[j][i] = is_violation
    model.Add(x[j] > x[i]).OnlyEnforceIf(is_violation)
    model.Add(x[j] <= x[i]).OnlyEnforceIf(is_violation.Not())

    weight = int(int_matrix[i, j])
    if weight > 0:
        objective_terms.append(weight * is_violation)

    if idx2val[j] > idx2val[i]:
        model.AddHint(is_violation, 1)
    else:
        model.AddHint(is_violation, 0)

    return is_violation


def _build_preference_succ(int_matrix: npt.NDArray, n: int) -> list[list[int]]:
    """``succ[u]`` = targets ``v`` for preference edges ``u -> v`` (``int_matrix[v, u] > 0``)."""
    succ: list[list[int]] = [[] for _ in range(n)]
    rows, cols = np.nonzero(int_matrix)
    for i, j in zip(rows, cols, strict=True):
        if i == j:
            continue
        # Edge j -> i when int_matrix[i, j] > 0; store as succ[j].append(i).
        succ[int(j)].append(int(i))
    return succ


def _add_valid_inequalities(
    model: cp_model.CpModel,
    x: list[cp_model.IntVar],
    b: list[list[cp_model.BoolVarT | None]],
    int_matrix: npt.NDArray,
    objective_terms: list[cp_model.LinearExpr],
    idx2val: dict[int, int],
    *,
    max_cycle_len: int,
) -> dict[str, int]:
    """Add sparse valid inequalities that preserve the stage-1 optimum."""
    n = int_matrix.shape[0]
    succ = _build_preference_succ(int_matrix, n)
    stats = {
        "two_cycle_cuts": 0,
        "three_cycle_cuts": 0,
        "four_cycle_cuts": 0,
        "transitivity_cuts": 0,
        "auxiliary_b_vars": 0,
    }
    transitivity_seen: set[tuple[int, int, int]] = set()

    def ensure(j: int, i: int) -> cp_model.BoolVarT:
        before = b[j][i]
        var = _ensure_violation_var(model, x, b, int_matrix, objective_terms, j, i, idx2val)
        if before is None:
            stats["auxiliary_b_vars"] += 1
        return var

    def add_transitivity(k: int, j: int, i: int) -> None:
        key = (k, j, i)
        if key in transitivity_seen:
            return
        transitivity_seen.add(key)
        model.Add(ensure(k, i) <= ensure(k, j) + ensure(j, i))
        stats["transitivity_cuts"] += 1

    rows, cols = np.nonzero(int_matrix)
    seen_mutual: set[tuple[int, int]] = set()
    for i, j in zip(rows, cols, strict=True):
        if i == j:
            continue
        pair = (min(int(i), int(j)), max(int(i), int(j)))
        if pair in seen_mutual:
            continue
        if int_matrix[j, i] > 0:
            seen_mutual.add(pair)
            model.Add(ensure(int(j), int(i)) + ensure(int(i), int(j)) >= 1)
            stats["two_cycle_cuts"] += 1

    if max_cycle_len >= 3:
        seen_triples: set[tuple[int, int, int]] = set()
        for a in range(n):
            for b_node in succ[a]:
                for c in succ[b_node]:
                    if c == a:
                        continue
                    if int_matrix[a, c] <= 0:
                        continue
                    cycle = tuple(sorted((a, b_node, c)))
                    if cycle in seen_triples:
                        continue
                    seen_triples.add(cycle)
                    model.Add(ensure(b_node, a) + ensure(c, b_node) + ensure(a, c) >= 1)
                    stats["three_cycle_cuts"] += 1
                    add_transitivity(c, b_node, a)

    if max_cycle_len >= 4:
        seen_quads: set[tuple[int, int, int, int]] = set()
        for a in range(n):
            for b_node in succ[a]:
                if b_node == a:
                    continue
                for c in succ[b_node]:
                    if c in (a, b_node):
                        continue
                    for d in succ[c]:
                        if d in (a, b_node, c):
                            continue
                        if int_matrix[a, d] <= 0:
                            continue
                        cycle = tuple(sorted((a, b_node, c, d)))
                        if cycle in seen_quads:
                            continue
                        seen_quads.add(cycle)
                        model.Add(
                            ensure(b_node, a) + ensure(c, b_node) + ensure(d, c) + ensure(a, d) >= 1
                        )
                        stats["four_cycle_cuts"] += 1
                        add_transitivity(d, c, a)

    return stats


def _default_permutation_scc(
    int_matrix: npt.NDArray,
    reference_order: npt.NDArray,
    rows: npt.NDArray,
    cols: npt.NDArray,
) -> list[int]:
    n = len(reference_order)
    reference_rank = {int(item_idx): rank for rank, item_idx in enumerate(reference_order)}

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(rows, cols, strict=True):
        parent = int(j)
        child = int(i)
        if parent != child:
            adjacency[parent].append(child)

    index = 0
    stack: list[int] = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlinks = [0] * n
    components: list[list[int]] = []

    def strongconnect(node: int) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack[node] = True

        for child in adjacency[node]:
            if indices[child] == -1:
                strongconnect(child)
                lowlinks[node] = min(lowlinks[node], lowlinks[child])
            elif on_stack[child]:
                lowlinks[node] = min(lowlinks[node], indices[child])

        if lowlinks[node] != indices[node]:
            return

        component: list[int] = []
        while True:
            item = stack.pop()
            on_stack[item] = False
            component.append(item)
            if item == node:
                break
        components.append(component)

    for item_idx in reference_order:
        item = int(item_idx)
        if indices[item] == -1:
            strongconnect(item)

    component_id = [-1] * n
    for cid, component in enumerate(components):
        for item in component:
            component_id[item] = cid

    component_edges: list[set[int]] = [set() for _ in components]
    component_indegree = [0] * len(components)
    for i, j in zip(rows, cols, strict=True):
        parent_cid = component_id[int(j)]
        child_cid = component_id[int(i)]
        if parent_cid == child_cid or child_cid in component_edges[parent_cid]:
            continue
        component_edges[parent_cid].add(child_cid)
        component_indegree[child_cid] += 1

    component_min_rank = [
        min(reference_rank[item] for item in component) for component in components
    ]
    queue: list[tuple[int, int]] = []
    for cid, degree in enumerate(component_indegree):
        if degree == 0:
            heappush(queue, (component_min_rank[cid], cid))

    component_order: list[int] = []
    while queue:
        _, cid = heappop(queue)
        component_order.append(cid)
        for child_cid in sorted(component_edges[cid], key=lambda idx: component_min_rank[idx]):
            component_indegree[child_cid] -= 1
            if component_indegree[child_cid] == 0:
                heappush(queue, (component_min_rank[child_cid], child_cid))

    if len(component_order) != len(components):
        # The SCC condensation graph should be acyclic by construction. Keep a
        # deterministic fallback if duplicate or malformed edges ever break that.
        component_order = sorted(range(len(components)), key=lambda idx: component_min_rank[idx])

    permutation: list[int] = []
    for cid in component_order:
        component = components[cid]
        if len(component) == 1:
            permutation.append(component[0])
            continue

        component_arr = np.array(sorted(component, key=lambda item: reference_rank[item]))
        local_incoming = int_matrix[np.ix_(component_arr, component_arr)].sum(axis=1)
        local_outgoing = int_matrix[np.ix_(component_arr, component_arr)].sum(axis=0)
        local_score = local_incoming - local_outgoing
        seed_order = component_arr[np.lexsort((component_arr, local_score))]

        component_permutation: list[int] = []
        for item_idx_raw in seed_order:
            item_idx = int(item_idx_raw)
            if not component_permutation:
                component_permutation.append(item_idx)
                continue

            current_cost = int(int_matrix[item_idx, component_permutation].sum())
            best_cost = current_cost
            best_pos = 0

            for pos, placed_idx in enumerate(component_permutation, start=1):
                current_cost += int(int_matrix[placed_idx, item_idx])
                current_cost -= int(int_matrix[item_idx, placed_idx])
                if current_cost <= best_cost:
                    best_cost = current_cost
                    best_pos = pos

            component_permutation.insert(best_pos, item_idx)

        permutation.extend(component_permutation)

    return permutation


def permutation_opt(
    p_matrix: npt.NDArray,
    timeout: float | None = None,
    break_symmetry: bool = False,
    *,
    _enable_valid_inequalities: bool = True,
) -> tuple[Optional[float], Optional[list[int]]]:
    """
    Given a matrix `p` where `p[i][j]` is the cost incurred if item `j` is
    placed after item `i` in a permutation, find the permutation that minimizes
    the total cost. A positive `p[i][j]` therefore represents evidence or
    weight for ordering `j` before `i`.

    The model is formulated as:
    Let `x[i]` be the new position of original item `i`.
    Objective: min ΣΣ p[i][j] * (x[j] > x[i])

    Args:
      p_matrix: An `n x n` NumPy array where p_matrix[i, j] represents the
           integer weight or count penalizing item `j` coming after
           item `i`.
      timeout: The maximum time in **seconds** allowed for the solver.
      break_symmetry: whether to solve a two-stage model instead of a single model. Note: this model must achieve optimality so not time limit will be set.

    Returns:
      A tuple containing:
      - The objective value after converting weights to integers, or None if
       no solution is found.
      - The optimal permutation as a list where `permutation[new_position] = original_item`,
       or None if no solution is found.
    """
    timeout = timeout or env_num("WIP_ANALYSIS_SOLVER_TIMEOUT", 180.0)
    assert p_matrix.ndim == 2, "Input matrix must be 2D"
    assert p_matrix.shape[0] == p_matrix.shape[1], "Input matrix must be square"

    n = p_matrix.shape[0]
    int_matrix = p_matrix.astype(int)
    np.fill_diagonal(int_matrix, 0)

    reference_order = np.arange(n, dtype=int)

    if not np.any(int_matrix):
        return 0.0, list(range(n))

    rows, cols = np.nonzero(int_matrix)
    acyclic_permutation = _zero_objective_permutation_if_acyclic(
        reference_order,
        rows,
        cols,
    )
    if acyclic_permutation is not None:
        logger.info(
            "Primary-path pair constraints are acyclic; "
            "skipped CP-SAT solver with exact zero-cost reference-order topo sort"
        )
        return 0.0, acyclic_permutation

    # Cyclic/conflicting evidence: keep the acyclic structure between SCCs
    # fixed by topological order, and only use greedy insertion inside each
    # cyclic SCC. This preserves observed inter-SCC order and leaves free
    # singleton nodes on reference/natural order.
    heuristic_order = _default_permutation_scc(
        int_matrix,
        reference_order,
        rows,
        cols,
    )
    hint_positions = np.empty(n, dtype=int)
    hint_positions[heuristic_order] = np.arange(n, dtype=int)
    violated_by_hint = hint_positions[cols] > hint_positions[rows]
    heuristic_obj = int(int_matrix[rows[violated_by_hint], cols[violated_by_hint]].sum())

    model = cp_model.CpModel()

    x = [model.NewIntVar(0, n - 1, f"x_{i}") for i in range(n)]
    b = [[None] * n for _ in range(n)]

    # Constr: items must be in different positions
    model.AddAllDifferent(x)

    objective_terms = []
    violation_vars = []
    for i, j in zip(rows, cols, strict=True):
        weight = int_matrix[i, j]
        # is_violation = 1 iff x[j] > x[i]
        is_violation = model.NewBoolVar(f"b_{j}_{i}")
        violation_vars.append(is_violation)
        b[j][i] = is_violation

        model.Add(x[j] > x[i]).OnlyEnforceIf(is_violation)
        model.Add(x[j] <= x[i]).OnlyEnforceIf(is_violation.Not())

        # add objective term
        objective_terms.append(weight * is_violation)

    idx2val = {item_idx: rank for rank, item_idx in enumerate(hint_positions)}

    for item_idx, rank in enumerate(hint_positions):
        model.AddHint(x[item_idx], int(rank))

    for i, j, is_violation in zip(rows, cols, violation_vars, strict=True):
        if hint_positions[int(j)] > hint_positions[int(i)]:
            model.AddHint(is_violation, 1)
        else:
            model.AddHint(is_violation, 0)

    if _enable_valid_inequalities:
        max_cycle_len = env_num("WIP_ANALYSIS_MAX_CYCLE_LEN", 3)
        cut_stats = _add_valid_inequalities(
            model,
            x,
            b,
            int_matrix,
            objective_terms,
            idx2val,
            max_cycle_len=max_cycle_len,
        )
        logger.debug("Valid inequalities: {}", cut_stats)

    # Objective
    model.Minimize(sum(objective_terms))

    # Solver
    logger.info(f"Starting CP-SAT solver with timeout={timeout}s for permutation optimization")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = env_num("WIP_ANALYSIS_SOLVER_WORKERS", 8)
    solver.parameters.random_seed = 42
    solver.parameters.max_time_in_seconds = timeout
    # solver.parameters.max_deterministic_time = timeout

    solver.parameters.log_search_progress = True
    solver.parameters.log_to_stdout = False
    solver.log_callback = _cp_sat_log_callback

    with max_parallel_solvers:
        status = solver.Solve(model)
    status_name = solver.StatusName(status)
    logger.opt(lazy=True).debug(
        f"Solver status: {status_name} (code={status}, UNKNOWN={cp_model.UNKNOWN})"
    )
    logger.info(
        f"CP-SAT time: wall={solver.WallTime():.3f}s user={solver.UserTime():.3f}s timeout={timeout}s status={status_name}"
    )
    logger.opt(lazy=True).debug("CP-SAT ResponseStats:\n{}", lambda: solver.ResponseStats())

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        positions = [solver.Value(x_i) for x_i in x]

        if break_symmetry:
            current_obj = int(solver.ObjectiveValue())
            # record solution of stage 1 for warm start
            model.Add(sum(objective_terms) == current_obj)

            # define new obj to break the symmetry
            objective_2_terms = []
            for i in range(n):
                weight_p2 = n - i
                objective_2_terms.append(weight_p2 * x[i])

            # Stage-1 heuristic hints must be cleared first; duplicate hints on the
            # same variable make the model MODEL_INVALID on the second Solve().
            model.ClearHints()
            for i in range(n):
                model.AddHint(x[i], solver.Value(expression=x[i]))
            for is_violation in violation_vars:
                model.AddHint(is_violation, solver.Value(is_violation))

            model.Minimize(sum(objective_2_terms))
            stage2_solver = cp_model.CpSolver()
            stage2_solver.parameters.num_search_workers = env_num("WIP_ANALYSIS_SOLVER_WORKERS", 8)
            stage2_solver.parameters.random_seed = 42
            # stage2_solver.parameters.log_search_progress = True
            # stage2_solver.parameters.log_to_stdout = True

            with max_parallel_solvers:
                stage2_status = stage2_solver.Solve(model)
            stage2_status_name = stage2_solver.StatusName(stage2_status)
            logger.info(
                "Second-stage CP-SAT: wall={:.3f}s status={}",
                stage2_solver.WallTime(),
                stage2_status_name,
            )

            if stage2_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                positions = [stage2_solver.Value(x_i) for x_i in x]
            else:
                logger.warning(
                    "Second-stage CP-SAT returned {}; keeping stage-1 permutation",
                    stage2_status_name,
                )

        permutation = [-1] * n
        for orig_idx, new_pos in enumerate(positions):
            permutation[new_pos] = orig_idx

        return solver.ObjectiveValue(), permutation
    elif status == cp_model.UNKNOWN:  # fallback logic
        positions = hint_positions.tolist()

        permutation = [-1] * n
        for orig_idx, new_pos in enumerate(positions):
            permutation[new_pos] = orig_idx

        if break_symmetry:
            extra_message = "the second-stage model will not be solved at all; "
        else:
            extra_message = ""

        logger.warning(
            "CP-SAT did not return a feasible/optimal solution before timeout={}s; {}"
            "returning heuristic fallback with objective={}",
            timeout,
            extra_message,
            heuristic_obj,
        )
        return heuristic_obj, permutation

    return None, None

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
import subprocess
from pathlib import Path
from mpmath import mp, mpf, nint, log10, ceil, matrix, lu_solve, fsum
from collections import Counter, deque
from joblib import Parallel, delayed
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import heapq
import sys
from fractions import Fraction
from sympy import Matrix, Rational
import gmpy2
from scipy.sparse import lil_matrix
import os
from collections import Counter, defaultdict
from itertools import combinations
import copy
from collections import Counter, defaultdict
from itertools import combinations
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from numbers import Integral

_DTYPE = torch.float64

sys.set_int_max_str_digits(100000)
gmpy2.get_context().precision = 1024*3
# Replace with correct path to mpsolve executable
solver_path = Path(r"C:\Users\sshvydun\Downloads\mpsolve-3.2.1-windows\MPSolve-3.2.1-windows\bin\mpsolve.exe")


def recover_markov_chain(prob, joint_prob, precision=400, tol=1e-14, label='', max_group=None, top_k=2,
                         is_parallel=True, loops_allowed=True):
    print("Estimating steady-state vector... ", end="")
    ss_vector = recover_roots(prob, precision, tol, label + "_ss_vector")
    print(f"Done. Number of states is {len(ss_vector)}")
    print("Estimating elements of W matrix... ", end="")
    w_elements = recover_roots(joint_prob, precision, tol, label + "_w_elements")
    print(f"Done. Number of elements in W is {len(w_elements)}")
    vals, counts = unique_with_counts(w_elements)

    if max_group is None:
        max_group = len(ss_vector)

    if is_parallel:
        results = Parallel(n_jobs=-1, batch_size=1)(
            delayed(generate_row_coalitions)(vals, counts, r_sum, max_group)
            for r_sum in tqdm(ss_vector, desc="Allocating elements to rows of matrix W...")
        )
    else:
        results = []
        for row_idx, r_sum in enumerate(tqdm(ss_vector, desc="Allocating elements to rows of matrix W...")):
            results.append(generate_row_coalitions(vals, counts, r_sum, max_group))
    print("Allocating elements to rows of matrix W... finished.")
    results = filter_best_coalitions(results, vals, ss_vector, top_k=top_k)
    print("Allocating elements to W... ", end="")
    markov_matrices = generate_allocations(vals, counts, results)
    markov_matrices = [[[x / ss_vector[i] if x is not None else None for x in row]
                        for i, row in enumerate(M)] for M in markov_matrices]
    print(f"Done. Found {len(markov_matrices)} solutions.")
    if not loops_allowed:
        markov_matrices = [matrix for matrix in markov_matrices if all(matrix[i][i] == 0 for i in range(len(matrix)))]
        print(f"Done. Filtered {len(markov_matrices)} solutions (no loops).")
    return {"ss_vector": ss_vector, "w_elements": w_elements, "markov chains": markov_matrices}


def recover_roots(init_prob, precision=400, tol=1e-14, label='', max_order=None):
    num_prob = len(init_prob)

    if max_order is not None:
        num_prob = max_order

    prob = init_prob      
    script_folder = Path.cwd()  # same folder as Python script
    example_path = script_folder / f"coefficients_{label}.txt"
    output_path = script_folder / f"roots_{label}.txt"

    print("Preparing coefficients...")
    coefficients = list(reversed(newton_from_power_sums_batched_mpq(prob, num_prob)))

    # Write the file
    with example_path.open("w", buffering=1024 * 1024) as f:
        f.write("Monomial;\n")
        f.write(f"Degree={len(coefficients) - 1};\n")
        f.write("Rational;\n")
        f.write("Real;\n\n")

        for coef in coefficients:
            f.write(str(coef))
            f.write("\n")

    # Solve polynomial
    subprocess.run(
        [str(solver_path),  "-Ga", f"-o{precision}", str(example_path)],    
        cwd=example_path.parent,  # run in same folder as Python script
        stdout=output_path.open("w"),  # redirect output to file
        check=True
    )

    complex_roots = []
    real_parts = []
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Keep original complex string
            complex_roots.append(line)

            # Extract real part
            if line.startswith("(") and "," in line:
                real_str = line.strip("()").split(",")[0].strip()
            else:
                real_str = line

                # Convert string to exact Fraction
            r = Fraction(real_str)
            real_parts.append(r)

    eps = Fraction(str(tol))
    real_parts_normalized = sorted([r for r in real_parts if r > eps])

    return real_parts_normalized


def unique_with_counts(elements):
    elements_mpf = [Fraction(e) for e in elements]
    counter = Counter(elements_mpf)
    # sort directly as mpf (no float conversion)
    unique_values = sorted(counter.keys())

    counts = [counter[val] for val in unique_values]
    return unique_values, counts


def generate_row_coalitions(values, counts, target, max_group, eps_error=None):
    """
    Optimized integer-based coalition generator.
    Finds all combinations of elements (with multiplicities) of size <= max_group
    whose sum is within eps of target.
    """

    # ---- Convert to mpf ----
    values = [mpf(str(v)) for v in values]
    target = mpf(str(target))

    # ---- Determine scaling factor ----
    sorted_values = sorted(set(values))
    if len(sorted_values) == 1:
        min_diff = sorted_values[0]
    else:
        min_diff = min(sorted_values[i + 1] - sorted_values[i]
                       for i in range(len(sorted_values) - 1))
    extra_digits = max(10, int(ceil(-log10(min_diff) + 10)))
    scale = mpf(str(10)) ** extra_digits / min_diff
    if eps_error is None:
        eps = min(min(values) / mpf(str(2)), mpf(str(0.00001 * target)))  # eps = 0.001% of the target value
    else:
        eps = mpf(str(eps_error))

    # ---- Convert to integers ----
    int_values = [int(nint(v * scale)) for v in values]
    int_target = int(nint(target * scale))
    int_eps = max(int(nint(eps * scale)), 1)

    # ---- Filter values <= target+eps ----
    filtered = [(v, c, idx) for idx, (v, c) in enumerate(zip(int_values, counts))
                if v <= int_target + int_eps]

    if not filtered:
        return []

    # ---- Sort descending for pruning (keep original index) ----
    filtered.sort(key=lambda x: -x[0])
    int_values, counts, orig_indices = zip(*filtered)
    int_values, counts, orig_indices = list(int_values), list(counts), list(orig_indices)
    size = len(int_values)

    # ---- Precompute suffix count for pruning ----
    suffix_count = [0] * (size + 1)
    for i in range(size - 1, -1, -1):
        suffix_count[i] = suffix_count[i + 1] + counts[i]

    solutions = []
    remaining_counts = counts.copy()

    # ---- Max-sum helper for pruning ----
    def max_possible_sum(start, slots):
        s = 0
        r = slots
        for i in range(start, size):
            take = min(r, remaining_counts[i])
            s += int_values[i] * take
            r -= take
            if r == 0:
                break
        return s

    # ---- Recursive DFS ----
    def backtrack(start, current_size, current_sum, coalition):

        # record valid solution
        if 1 <= current_size <= max_group and abs(current_sum - int_target) <= int_eps:
            # map to original indices
            solutions.append([orig_indices[i] for i in coalition])

        # stop if max size reached
        if current_size == max_group:
            return

        # prune overshoot
        if current_sum > int_target + int_eps:
            return

        remaining_slots = max_group - current_size

        # prune if max possible sum can't reach target
        max_add = max_possible_sum(start, remaining_slots)
        if current_sum + max_add < int_target - int_eps:
            return

        # prune if not enough elements left
        if suffix_count[start] < 1:  # at least 1 element needed to proceed
            return

        # recurse
        for i in range(start, size):
            if remaining_counts[i] == 0:
                continue
            v = int_values[i]
            if current_sum + v > int_target + int_eps:
                continue

            coalition.append(i)
            remaining_counts[i] -= 1

            backtrack(i, current_size + 1, current_sum + v, coalition)

            remaining_counts[i] += 1
            coalition.pop()

    # ---- Start DFS ----
    backtrack(0, 0, 0, [])
    return solutions


def filter_best_coalitions(all_rows_coalitions, vals, ss, top_k=4):
    """
    Extract top_k coalitions closest to target ss[i] for each row,
    keeping all arithmetic in mpf.

    Parameters:
    - all_rows_coalitions: list of lists of index lists
    - vals: list of mpf numbers
    - ss: list of mpf targets
    - top_k: number of best coalitions to keep

    Returns:
    - filtered: list of lists of top_k coalitions
    """
    filtered = []

    for i, coalitions in enumerate(tqdm(all_rows_coalitions, desc="Processing rows")):
        target = ss[i]

        if isinstance(top_k, int):
            if len(coalitions) <= top_k:
                filtered.append(coalitions)
                continue

            best = [sub for _, sub in heapq.nsmallest(
                top_k,
                ((abs(sum(vals[idx] for idx in sub) - target), sub) for sub in coalitions),
                key=lambda x: x[0]
            )]

        else:
            # Treat top_k as tolerance
            diffs = [
                (abs(sum(vals[idx] for idx in sub) - target), sub)
                for sub in coalitions
            ]

            # Keep those strictly within tolerance
            within_tol = [(d, sub) for d, sub in diffs if d < top_k]

            if within_tol:
                # Sort by closeness
                within_tol.sort(key=lambda x: x[0])
                best = [sub for _, sub in within_tol]
            else:
                # Fallback: take the single closest
                best = [min(diffs, key=lambda x: x[0])[1]]

        filtered.append(best)

    return filtered


def generate_allocations(vals, cnts, all_rows_coalitions):
    N = len(all_rows_coalitions)
    remaining_counts = cnts[:]

    # Convert row and column coalitions to Counters
    row_domains = [[Counter(coal) for coal in row] for row in all_rows_coalitions]
    col_domains = [[Counter(coal) for coal in col] for col in all_rows_coalitions]

    # Element -> columns / rows mapping
    elem_to_cols = defaultdict(set)
    elem_to_rows = defaultdict(set)
    for r in range(N):
        for coal in row_domains[r]:
            for e in coal:
                elem_to_rows[e].add(r)
        for coal in col_domains[r]:
            for e in coal:
                elem_to_cols[e].add(r)

    solutions = []

    def recursive_assign(assignment):
        if len(assignment) == N:
            # Build NxN matrix
            matrix = [[None]*N for _ in range(N)]
            for r, (_, alloc) in assignment.items():
                for e, c in alloc:
                    matrix[r][c] = vals[e]
            solutions.append(matrix)
            return

        # Choose unassigned row with smallest domain
        unassigned = [r for r in range(N) if r not in assignment]

        def row_priority(r):
            num_coals = len(row_domains[r])
            total_size = 0
            for coal in row_domains[r]:
                for val in coal.values():
                    total_size += val
            return num_coals, -total_size

        row = min(unassigned, key=row_priority)

        for coal_counter in row_domains[row]:
            elements = list(coal_counter.keys())

            # Candidate columns for each element
            candidate_columns = {}
            for e in elements:
                candidate_columns[e] = [
                    c for c in elem_to_cols[e]
                    if any(col[e] > 0 for col in col_domains[c])
                ]
                # Fail-fast: not enough columns to assign this element
                if len(candidate_columns[e]) < coal_counter[e]:
                    break
            else:  # Only proceed if all elements have enough candidate columns
                # Recursive assignment of elements to columns
                def assign_elements(idx, used_cols, current_alloc):
                    if idx == len(elements):
                        yield current_alloc
                        return
                    e = elements[idx]
                    cnt = coal_counter[e]
                    for cols_selected in combinations(candidate_columns[e], cnt):
                        if any(c in used_cols for c in cols_selected):
                            continue
                        yield from assign_elements(
                            idx+1,
                            used_cols | set(cols_selected),
                            current_alloc + [(e, c) for c in cols_selected]
                        )

                for alloc in assign_elements(0, set(), []):
                    # ----- COLUMN UPDATE -----
                    col_backup = {}
                    col_usage = defaultdict(Counter)
                    for e, c in alloc:
                        col_usage[c][e] += 1

                    for c, usage in col_usage.items():
                        old_coals = col_domains[c]
                        # Keep only coalitions that can satisfy usage
                        new_coals = [
                            Counter(coal) for coal in old_coals
                            if all(coal[e] >= cnt for e, cnt in usage.items())
                        ]
                        col_backup[c] = old_coals
                        col_domains[c] = new_coals

                    # ----- UPDATE GLOBAL COUNTS -----
                    for e, cnt in coal_counter.items():
                        remaining_counts[e] -= cnt

                    # ----- PRUNE ROWS -----
                    row_backup = {}
                    affected_rows = set()
                    for e in coal_counter:
                        affected_rows |= elem_to_rows[e]

                    valid = True
                    for r in affected_rows:
                        if r == row or r in assignment:
                            continue
                        old = row_domains[r]
                        new = [
                            Counter(coal) for coal in old
                            if all(remaining_counts[e] >= cnt for e, cnt in coal.items())
                        ]
                        row_backup[r] = old
                        row_domains[r] = new
                        if not new:
                            valid = False
                            break

                    if not valid:
                        # rollback everything
                        for e, cnt in coal_counter.items():
                            remaining_counts[e] += cnt
                        for c, old_coals in col_backup.items():
                            col_domains[c] = old_coals
                        for r, old in row_backup.items():
                            row_domains[r] = old
                        continue

                    # ----- RECURSE -----
                    assignment[row] = (coal_counter, alloc)
                    recursive_assign(assignment)
                    del assignment[row]

                    # ----- ROLLBACK -----
                    for e, cnt in coal_counter.items():
                        remaining_counts[e] += cnt
                    for c, old_coals in col_backup.items():
                        col_domains[c] = old_coals
                    for r, old in row_backup.items():
                        row_domains[r] = old

    recursive_assign({})
    return solutions


def newton_from_power_sums_batched_mpq(power_sums, max_order=None):
    mpq = gmpy2.mpq
    mpz = gmpy2.mpz
    numer = gmpy2.numer
    denom = gmpy2.denom
    lcm = gmpy2.lcm

    n = max(power_sums.keys())
    if max_order is not None:
        n = min(n, max_order)

    zero = mpq(0)

    s = [zero] + [
        power_sums.get(k, zero)
        for k in range(1, n + 1)
    ]

    s_num = [mpz(0)] + [
        mpz(numer(s[k]))
        for k in range(1, n + 1)
    ]

    s_den = [mpz(1)] + [
        mpz(denom(s[k]))
        for k in range(1, n + 1)
    ]

    a_num = [mpz(0)] * (n + 1)
    a_den = [mpz(1)] * (n + 1)

    a_num[0] = mpz(1)
    a_den[0] = mpz(1)

    for k in tqdm(range(1, n + 1), desc="Computing coefficients from power sums…"):
        L = s_den[k]

        for j in range(1, k):
            L = lcm(L, a_den[j] * s_den[k - j])

        total_num = s_num[k] * (L // s_den[k])

        for j in range(1, k):
            term_den = a_den[j] * s_den[k - j]
            total_num += a_num[j] * s_num[k - j] * (L // term_den)

        q = mpq(-total_num, k * L)

        a_num[k] = mpz(numer(q))
        a_den[k] = mpz(denom(q))

    return [
        mpq(a_num[k], a_den[k])
        for k in range(n + 1)
    ]


def _node_index(nodes):
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    return {node: i for i, node in enumerate(nodes)}


def _to_logits(x, shape, dim):
    x = torch.as_tensor(x, dtype=_DTYPE)
    if x.shape != shape:
        raise ValueError(f"init_solution must have shape {shape}")
    if not torch.isfinite(x).all() or torch.any(x < 0):
        raise ValueError("init_solution must contain finite nonnegative values")
    sums = x.sum(dim=dim, keepdim=True)
    if torch.any(sums <= 0):
        raise ValueError("each probability row must have positive sum")
    x = x / sums
    return torch.log(x.clamp_min(torch.finfo(_DTYPE).tiny))


def _combination(comb):
    return (comb,) if isinstance(comb, Integral) else tuple(comb)


def _generator(seed, restart):
    return None if seed is None else torch.Generator().manual_seed(int(seed) + restart)


class SteadyStateSolver(nn.Module):
    def __init__(self, n_states, prob, nodes, min_size=2, max_size=4, init_solution=None, generator=None):
        super().__init__()
        self.n_nodes = len(nodes)
        node_index = _node_index(nodes)
        shape = (self.n_nodes, n_states)
        logits = torch.randn(shape, dtype=_DTYPE,
                             generator=generator) if init_solution is None else _to_logits(init_solution, shape, 1)
        self.logits = nn.Parameter(logits)
        self.terms = []
        counter = 0
        for size, values in prob.items():
            size = int(size)
            if not (min_size <= size <= max_size) or not values:
                continue
            indices, targets = [], []
            for comb, value in values.items():
                comb = _combination(comb)
                if len(comb) != size:
                    raise ValueError(f"combination {comb} has size {len(comb)}, expected {size}")
                indices.append([node_index[node] for node in comb])
                targets.append(float(value))
            idx_name, target_name = f"indices_{counter}", f"targets_{counter}"
            self.register_buffer(idx_name, torch.tensor(indices, dtype=torch.long))
            self.register_buffer(target_name, torch.tensor(targets, dtype=_DTYPE))
            self.terms.append((idx_name, target_name))
            counter += 1
        if not self.terms:
            raise ValueError("No steady-state equations found")

    def solution(self):
        return torch.softmax(self.logits, dim=1)

    def forward(self):
        s = self.solution()
        return torch.cat([s[getattr(self,
                                    idx)].prod(dim=1).sum(dim=1) - getattr(self, target) for idx, target in self.terms])


class MarkovMatrixSolver(nn.Module):
    def __init__(self, ss_vectors, prob, nodes, min_size=2, max_size=3, init_solution=None, generator=None):
        super().__init__()
        s = torch.as_tensor(ss_vectors, dtype=_DTYPE)
        if s.ndim != 2 or s.shape[0] != len(nodes):
            raise ValueError("ss_vectors must have shape (len(nodes), n_states)")
        if not torch.isfinite(s).all() or torch.any(s < 0) or torch.any(s.sum(dim=1) <= 0):
            raise ValueError("ss_vectors must contain finite nonnegative rows with positive sums")
        s = s / s.sum(dim=1, keepdim=True)
        self.n_nodes, self.n_states = s.shape
        self.register_buffer("ss_vectors", s)
        shape = (self.n_nodes, self.n_states, self.n_states)
        logits = torch.randn(shape, dtype=_DTYPE,
                             generator=generator) if init_solution is None else _to_logits(init_solution, shape, 2)
        self.logits = nn.Parameter(logits)
        node_index = _node_index(nodes)
        self.terms = []
        counter = 0
        for kappa, by_size in prob.items():
            kappa = int(kappa)
            if kappa < 0:
                raise ValueError("kappa must be nonnegative")
            for size, values in by_size.items():
                size = int(size)
                if not (min_size <= size <= max_size) or not values:
                    continue
                indices, targets = [], []
                for comb, value in values.items():
                    comb = _combination(comb)
                    if len(comb) != size:
                        raise ValueError(f"combination {comb} has size {len(comb)}, expected {size}")
                    indices.append([node_index[node] for node in comb])
                    targets.append(float(value))
                idx_name, target_name = f"indices_{counter}", f"targets_{counter}"
                self.register_buffer(idx_name, torch.tensor(indices, dtype=torch.long))
                self.register_buffer(target_name, torch.tensor(targets, dtype=_DTYPE))
                self.terms.append((kappa, idx_name, target_name))
                counter += 1
        if not self.terms:
            raise ValueError("No transition-matrix equations found")

    def solution(self):
        return torch.softmax(self.logits, dim=2)

    def forward(self):
        p, s = self.solution(), self.ss_vectors
        residuals = [(torch.bmm(s.unsqueeze(1), p).squeeze(1) - s).reshape(-1)]
        powers = {}
        for kappa, idx_name, target_name in self.terms:
            if kappa not in powers:
                powers[kappa] = torch.linalg.matrix_power(p, kappa)
            indices, targets = getattr(self, idx_name), getattr(self, target_name)
            matrices = powers[kappa][indices].prod(dim=1)
            row_sums = matrices.sum(dim=2)
            vectors = s[indices].prod(dim=1)
            sigma = (row_sums * vectors).sum(dim=1)
            residuals.append(sigma - targets)
        return torch.cat(residuals)


def _optimize(model, num_epochs=5000, lr=0.01, tolerance=1e-20, lbfgs_max_iter=200):
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss, best_logits = float("inf"), None
    for _ in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = model().square().sum()
        value = loss.item()
        if not torch.isfinite(loss):
            break
        if value < best_loss:
            best_loss, best_logits = value, model.logits.detach().clone()
        if value <= tolerance:
            break
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        value = model().square().sum().item()
        if value < best_loss:
            best_loss, best_logits = value, model.logits.detach().clone()
        if best_logits is None:
            raise FloatingPointError("Optimization produced no finite solution")
        model.logits.copy_(best_logits)
    if lbfgs_max_iter > 0 and best_loss > tolerance:
        optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter, tolerance_grad=1e-13,
                                tolerance_change=1e-15, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = model().square().sum()
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            value = model().square().sum().item()
            if value < best_loss:
                best_loss, best_logits = value, model.logits.detach().clone()
            model.logits.copy_(best_logits)
    return model.solution().detach().clone(), best_loss


def find_steady_states(prob, n_states, nodes, max_size, min_size=2, num_epochs=5000, init_solution=None, lr=0.03,
                       tolerance=1e-20, n_restarts=5, seed=None, lbfgs_max_iter=200):
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    best_solution, best_loss = None, float("inf")
    for restart in range(n_restarts):
        init = init_solution if restart == 0 else None
        model = SteadyStateSolver(n_states, prob, nodes, min_size, max_size, init, _generator(seed, restart))
        solution, loss = _optimize(model, num_epochs, lr, tolerance, lbfgs_max_iter)
        if loss < best_loss:
            best_solution, best_loss = solution, loss
    return best_solution, best_loss


def get_p_matrix(prob, ss_vectors, nodes, max_size=3, min_size=2, num_epochs=5000, init_solution=None, lr=0.01,
                 tolerance=1e-20, n_restarts=5, seed=None, lbfgs_max_iter=200):
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    best_solution, best_loss = None, float("inf")
    for restart in range(n_restarts):
        init = init_solution if restart == 0 else None
        model = MarkovMatrixSolver(ss_vectors, prob, nodes, min_size, max_size, init, _generator(seed, restart))
        solution, loss = _optimize(model, num_epochs, lr, tolerance, lbfgs_max_iter)
        if loss < best_loss:
            best_solution, best_loss = solution, loss
    return best_solution, best_loss


def recover_markov_chain_numerical(obj, n_states, nodes, max_size=None, min_size=2, ss_epochs=50000, p_epochs=50000,
                                   ss_lr=0.03, p_lr=0.01, tolerance=1e-20, ss_restarts=5, p_restarts=5, seed=None,
                                   lbfgs_max_iter=200, init_ss=None, init_p=None):
    if "prob" not in obj or "joint_prob" not in obj:
        raise KeyError("obj must contain 'prob' and 'joint_prob'")
    prob, joint_prob = obj["prob"], obj["joint_prob"]
    if max_size is None:
        max_size = max(int(k) for k in prob)
    ss_vectors, ss_error = find_steady_states(prob, n_states, nodes, max_size, min_size, ss_epochs, init_ss, ss_lr,
                                              tolerance, ss_restarts, seed, lbfgs_max_iter)
    p_matrix, p_error = get_p_matrix(joint_prob, ss_vectors, nodes, max_size, min_size, p_epochs, init_p, p_lr,
                                     tolerance, p_restarts, seed, lbfgs_max_iter)
    return {"ss_vectors": ss_vectors.cpu().numpy(), "p_matrix": p_matrix.cpu().numpy(),
            "ss_error": ss_error, "p_error": p_error}

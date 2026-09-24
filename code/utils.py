import pickle
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import random
from collections import defaultdict, Counter, deque
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path
from mpmath import mp, mpf, nint, log10, ceil, matrix, lu_solve, fsum
from joblib import Parallel, delayed
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import heapq
import sys
from sympy import Matrix, Rational
import gmpy2
from scipy.sparse import lil_matrix
import os
sys.set_int_max_str_digits(100000)
gmpy2.get_context().precision = 1024


def save_graph(filename, g):
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(g, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_graph(filename):
    with open(filename + '.pickle', 'rb') as handle:
        return pickle.load(handle)


def compute_ss_contact_probabilities(markov_graph, n_walkers=1, is_approximate=False):
    """
    Compute steady-state contact probabilities using arbitrary precision.

    Returns
    -------
    dict with:
        ss_vector : steady-state vector (mpf)
        prob : dict of power sums
        :param n_walkers: Maximum power-sum order
        :param markov_graph: Markov transition table
        :param is_approximate:  compute statistics approximately using floating point arithmetic (True)
         or exactly using fractions (False)
    """

    markov_chain = dict_to_markov_chain(markov_graph)
    if is_approximate:
        ss_vectors = steady_state_vector(markov_chain)
        ss_vectors = [gmpy2.mpq(str(x)) for x in ss_vectors]
    else:
        ss_vectors = steady_state_vector_exact(markov_chain)[0] 
    order = np.argsort(ss_vectors)

    if is_approximate:
        w_elements, w_k = weighted_ss_list(markov_chain, ss_vectors, order=1)
        w_elements = [gmpy2.mpq(str(x)) for x in w_elements]
        w_k_sorted = [[w_k[old_i, old_j] for old_j in order] for old_i in order]
    else:
        w_elements, w_k = weighted_ss_list_fraction(markov_chain, ss_vectors, order=1)
        w_k_sorted = [[w_k[old_i][old_j] for old_j in order] for old_i in order]

    w_elements = sorted(w_elements)
    if is_approximate:
        w_elements2 = sorted(weighted_ss_list(markov_chain, ss_vectors, order=2)[0])
        w_elements2 = [gmpy2.mpq(str(x)) for x in w_elements2]
    else:
        w_elements2 = sorted(weighted_ss_list_fraction(markov_chain, ss_vectors, order=2)[0])

    list_p_sums = power_sums_fraction(ss_vectors, n_walkers)  
    list_w_sums = power_sums_fraction(w_elements, n_walkers)  
    list_w_sums2 = power_sums_fraction(w_elements2, min(10, n_walkers), is_parallel=False)  
    prob, joint_prob, joint_prob2 = {}, {}, {}
    for i in range(n_walkers):
        prob[i + 1] = list_p_sums[i]
        joint_prob[i + 1] = list_w_sums[i]
        if i < len(list_w_sums2):
            joint_prob2[i + 1] = list_w_sums2[i]

    return {"ss_vector": ss_vectors, "prob": prob, "joint_prob": joint_prob, "joint_prob2": joint_prob2,
            "w_elements": w_elements, "w_matrix": w_k_sorted}


def dict_to_markov_chain(table):
    """
    Convert a dictionary-based transition table to a stochastic NetworkX DiGraph
    using mpmath.mpf for high precision.

    Parameters:
    - table: dict of {hist: (probs, neighbors)}
      probs: list of probabilities (can sum != 1)
      neighbors: list of next states
    - precision: number of decimal digits for mpf

    Returns:
    - g: networkx.DiGraph with 'weight' attributes as mpf, row-stochastic
    """
    g = nx.DiGraph()

    for hist, (probs, neighbors) in table.items():
        # Convert probs to mpf
        for i, neighbor in enumerate(neighbors):
            p = probs[i]
            if p > 0:
                # Build next_hist
                if len(hist) > 0:
                    next_hist = hist[1:] + (neighbor,)
                else:
                    next_hist = (neighbor,)
                g.add_edge(hist, next_hist, weight=p)
    return g


def steady_state_vector_exact(graph):
    """
    Compute the steady-state vector of a directed weighted graph
    with exact rational arithmetic (Fractions).

    Parameters:
    - graph: networkx.DiGraph, edge weights must sum to 1 per node

    Returns:
    - pi: list of Fractions stationary probabilities
    - nodes_order: list of nodes corresponding to pi
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    # Build dense Fraction transition matrix
    P = [[Fraction(0) for _ in range(n)] for _ in range(n)]

    for u in nodes:
        i = node_index[u]
        out_edges = list(graph.out_edges(u, data=True))
        if len(out_edges) == 0:
            # Dangling node to self-loop
            P[i][i] = Fraction('1')
        else:
            for _, v, d in out_edges:
                j = node_index[v]
                weight = d.get('weight', '1')
                if isinstance(weight, float):
                    weight = Fraction(str(weight))  # exact from float
                else:
                    weight = Fraction(weight)
                P[i][j] = weight

    # Solve (I - P^T) pi = 0 with sum(pi) = 1
    A = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    b = [Fraction(0) for _ in range(n)]

    # Build (I - P^T)
    for i in range(n):
        for j in range(n):
            A[i][j] = Fraction(int(i == j)) - P[j][i]  # note transpose

    # Replace last row to impose sum(pi) = 1
    for j in range(n):
        A[-1][j] = Fraction('1')
    b[-1] = Fraction('1')

    # A and b as lists of Fractions
    A_sym = Matrix([[Rational(f.numerator, f.denominator) for f in row] for row in A])
    b_sym = Matrix([Rational(f.numerator, f.denominator) for f in b])

    pi_sym = A_sym.LUsolve(b_sym)
    pi = [Fraction(int(x.p), int(x.q)) for x in pi_sym]
    return pi, nodes


def weighted_ss_list_fraction(graph, ss_vector, order=1):
    """
    Compute a flat list of s_i * (P^k)_{ij} values using exact fractions.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph with transition probabilities stored in 'weight'
    ss_vector : list
        Steady-state vector (Fractions)
    order : int
        Power of transition matrix (P^k)

    Returns
    -------
    result : list
        Flat list of s_i * (P^k)_{ij} values (Fractions)
    Pk : list of lists
        P^k matrix as list of list of Fractions
    """
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}

    N = len(ss_vector)
    P = [[Fraction(0) for _ in range(N)] for _ in range(N)]

    for u in nodes:
        i = index[u]
        for v in graph.successors(u):
            j = index[v]
            P[i][j] = Fraction(str(graph[u][v]['weight']))

    # Compute P^order using exact fraction matrix multiplication
    def mat_pow(M, k):
        n = N
        result = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]  # identity
        for _ in range(k):
            temp = [[Fraction(0) for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for l in range(n):
                    if result[i][l] == 0:
                        continue
                    for j in range(n):
                        if M[l][j] != 0:
                            temp[i][j] += result[i][l] * M[l][j]
            result = temp
        return result

    Pk = mat_pow(P, order)

    # Compute s_i * (P^k)_{ij} and flatten
    result = []
    for i in range(N):
        s_i = ss_vector[i]  # Fraction
        for j in range(N):
            if Pk[i][j] != 0:
                Pk[i][j] = s_i * Pk[i][j]
                result.append(Pk[i][j])

    return result, Pk


def power_sums_fraction(roots, r_max, is_parallel=True, n_jobs=-1):
    if not is_parallel:
        return _power_sums_chunk(roots, r_max)

    k = os.cpu_count() if n_jobs == -1 else n_jobs
    chunks = [roots[i::k] for i in range(k)]

    partials = Parallel(n_jobs=n_jobs)(
        delayed(_power_sums_chunk)(chunk, r_max)
        for chunk in tqdm(chunks, desc="Computing stationary contact probabilities")
        if chunk
    )
    return [sum(p[i] for p in partials) for i in range(r_max)]


def steady_state_vector(graph):
    """
    Compute the steady-state vector of a directed weighted graph.

    Parameters:
    - G: networkx.DiGraph, edge weights must sum to 1 per node

    Returns:
    - pi: numpy array of stationary probabilities
    - nodes_order: list of nodes corresponding to pi
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    rows, cols, data = [], [], []
    # Build sparse transition matrix
    for u in nodes:
        i = node_index[u]
        out_edges = list(graph.out_edges(u, data=True))
        if len(out_edges) == 0:
            # Dangling node to self-loop
            rows.append(i)
            cols.append(i)
            data.append(1.0)
        else:
            for _, v, d in out_edges:
                j = node_index[v]
                weight = float(d.get('weight', 1.0))
                rows.append(i)
                cols.append(j)
                data.append(weight)

    p_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)

    # Solve (I - P^T) pi = 0 with sum(pi)=1
    a_matrix = (sp.eye(n, format='csr') - p_matrix.T).tolil()
    a_matrix[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = spla.spsolve(a_matrix.tocsr(), b)
    return pi


def weighted_ss_list(graph, ss_vector, order=1):
    """
    Compute a flat list of s_i * (P^k)_{ij} values.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph with transition probabilities stored in 'weight'
    ss_vector : list
        Steady-state vector (mpf values)
    order : int
        Power of transition matrix (P^k)

    Returns
    -------
    list
        Flat list of s_i * (P^k)_{ij} values (mpf)
    """

    nodes = list(graph.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    P = lil_matrix((n, n), dtype=float)
    for u in nodes:
        i = index[u]
        for v in graph.successors(u):
            j = index[v]
            P[i, j] = graph[u][v]['weight']
    P = P.tocsr()
    # element-wise power (NOT matrix power)
    P.data **= order
    ss_vector = np.array(ss_vector, dtype=float)
    # row scaling
    P = P.multiply(ss_vector[:, None])
    if order > 1:
        Pk = None
    else:
        Pk = P.toarray()
    # extract nonzeros
    result = P.data.tolist()

    return result, Pk


def _power_sums_chunk(roots, r_max):
    roots = [gmpy2.mpq(r.numerator, r.denominator) for r in roots]
    powers = roots[:]
    sums = []

    for _ in range(r_max):
        sums.append(sum(powers, gmpy2.mpq(0)))
        for i in range(len(powers)):
            powers[i] *= roots[i]

    return sums


def sparse_probs(neighbors, sparsity=0.6, prob_precision=None):
    n = len(neighbors)
    keep = max(1, int(n * sparsity))

    active = np.random.choice(n, size=keep, replace=False)
    probs = np.zeros(n)

    if prob_precision is None:
        probs[active] = np.random.rand(keep)
        probs /= probs.sum()

    else:
        units = 10 ** prob_precision

        if keep > units:
            raise ValueError("Too many active neighbors for the requested precision.")

        if keep == 1:
            probs[active] = 1.0
        else:
            cuts = np.sort(
                np.random.choice(
                    np.arange(1, units),
                    size=keep - 1,
                    replace=False,
                )
            )
            weights = np.diff(
                np.concatenate(([0], cuts, [units]))
            )
            np.random.shuffle(weights)
            probs[active] = weights / units

    return probs


def build_kth_order_markov_graph(graph, k, sparsity=0.6, loops=True, prob_precision=None):
    """
    Build a k-th order Markov chain table from adjacency list.
    Returns: dict mapping history tuple -> (neighbor_probs, neighbor_list)
    """
    if loops:
        adj = {node: list(graph.neighbors(node)) + [node] for node in graph.nodes()}
    else:
        adj = {node: list(graph.neighbors(node)) for node in graph.nodes()}
    table = {}
    chk = True
    print(f"Building strongly connected {k+1}-order Markov graph with sparsity {sparsity}")
    while chk:
        chk = False
        if k == -1:
            # 0-th order: history is empty tuple
            for node in graph:
                neighbors = np.array(adj[node])
                if len(neighbors) == 0:
                    continue
                probs = sparse_probs(neighbors, sparsity, prob_precision=prob_precision)
                mask = probs > 0
                table[(node,)] = (probs[mask], neighbors[mask])
        else:
            # all possible histories of length k
            histories = generate_valid_histories(adj, k + 1)
            sparse_graph = sparsify_strongly_connected_graph(build_history_graph(histories, graph), sparsity)
            table = assign_sparse_weights(sparse_graph, prob_precision=prob_precision)
        if not is_markov_chain_strongly_connected(table):
            chk = True
            table = {}
    print(f"Strongly connected {k+1}-order Markov graph with sparsity {sparsity} is constructed")
    return table


def generate_valid_histories(adj, order):
    valid_histories = []

    def dfs(path):
        if len(path) == order:
            valid_histories.append(tuple(path))
            return
        last = path[-1]
        for neighbor in adj[last]:
            dfs(path + [neighbor])

    for node in adj:
        dfs([node])

    return valid_histories


def sparsify_strongly_connected_graph(g, sparsity=0, min_edges_per_node=2):
    """
    Remove edges randomly while preserving strong connectivity.

    Parameters
    ----------
    g : nx.DiGraph
        Strongly connected directed graph
    sparsity : float
        Fraction of edges to attempt to remove (0 = keep all, 1 = maximum removal)
    min_edges_per_node : int
        Minimum outgoing edges per node

    Returns
    -------
    G_sparse : nx.DiGraph
        Sparse, strongly connected graph
    """
    if not nx.is_strongly_connected(g):
        raise ValueError("Input graph must be strongly connected")

    g_sparse = g.copy()
    total_edges = g_sparse.number_of_edges()
    target_remove = int(total_edges * sparsity)

    # Only consider edges from nodes that have more than min_edges_per_node outgoing edges
    candidate_edges = [(u, v) for u, v in g_sparse.edges() if g_sparse.out_degree(u) > min_edges_per_node]
    random.shuffle(candidate_edges)

    removed = 0
    for u, v in candidate_edges:
        if removed >= target_remove:
            break

        g_sparse.remove_edge(u, v)

        # Only check connectivity if it might matter
        if nx.is_strongly_connected(g_sparse):
            removed += 1
        else:
            g_sparse.add_edge(u, v)  # restore if broken

    return g_sparse


def assign_sparse_weights(g_sparse, epsilon=1e-4, prob_precision=None):
    """
    Assign random weights to edges, ensuring each node's outgoing edges sum to 1.
    """
    table = {}
    for node in g_sparse.nodes():
        neighbors = list(g_sparse.successors(node))
        node_neighbors = [nodes[-1] for nodes in neighbors]
        if not neighbors:
            continue
        if prob_precision is None:
            probs = np.random.rand(len(neighbors))
            probs = [Fraction(p) + Fraction(epsilon) for p in probs]
            total = sum(probs)
            probs = [p / total for p in probs]
        else:
            units = 10 ** prob_precision
            n = len(neighbors)
            if n > units:
                raise ValueError(
                    "Too many outgoing edges for the requested precision."
                )
            if n == 1:
                probs = [Fraction(1, 1)]
            else:
                cuts = np.sort(
                    np.random.choice(
                        np.arange(1, units),
                        size=n - 1,
                        replace=False,
                    )
                )
                weights = np.diff(
                    np.concatenate(([0], cuts, [units]))
                )
                np.random.shuffle(weights)
                probs = [Fraction(int(w), units) for w in weights]
        table[node] = (probs, node_neighbors)

    return table


def is_markov_chain_strongly_connected(table):
    """
    Check if a k-th order Markov chain is strongly connected.

    Parameters
    ----------
    table : dict
        Mapping from history (tuple) -> (probs, neighbors)
        probs: list or array of probabilities corresponding to neighbours
        neighbors: list of next nodes

    Returns
    -------
    bool
        True if the Markov chain is strongly connected, False otherwise
    """
    g = nx.DiGraph()

    for hist, (probs, neighbors) in table.items():
        for i, neighbor in enumerate(neighbors):
            if probs[i] > 0:
                # Create next history for k-th order
                if len(hist) > 0:
                    next_hist = hist[1:] + (neighbor,)
                else:
                    next_hist = (neighbor,)
                g.add_edge(hist, next_hist)

    if len(g) == 0:
        return False  # no nodes, not connected
    return nx.is_strongly_connected(g)


def run_walkers(
    markov_graph,
    sequence_length=1000,
    n_walkers=5,
    disable_tqdm=False,
    rng=None,
    initial_state_ids=None,
    return_final_state_ids=False,
):
    """
    Simulate walkers on an order-k Markov model using history-states directly.

    Parameters
    ----------
    markov_graph : dict
        Mapping:
            history_tuple -> (probs, neighbors)

        where:
            - history_tuple has length k+1
            - probs are transition probabilities from that history
            - neighbors are the next visited physical locations

    sequence_length : int
        Number of visited locations to return per walker for this batch.

    n_walkers : int
        Number of walkers.

    disable_tqdm : bool
        Disable progress bar.

    rng : np.random.Generator or None
        Random generator. If None, a new default generator is created.

    initial_state_ids : np.ndarray or None
        Optional array of shape (n_walkers,) containing the Markov state ID
        of each walker at the start of the batch.

        Use this to continue a simulation from the previous batch.

    return_final_state_ids : bool
        If True, return:
            positions, final_state_ids

        If False, return:
            positions

    Returns
    -------
    out : np.ndarray
        Array of shape (sequence_length, n_walkers) containing visited locations.

    final_state_ids : np.ndarray, optional
        Final Markov state IDs after this batch.
    """

    if rng is None:
        rng = np.random.default_rng()

    # ---- Compile history states to integer ids ----
    states = list(markov_graph.keys())
    state_to_id = {state: i for i, state in enumerate(states)}
    n_states = len(states)

    if n_states == 0:
        out = np.empty((sequence_length, n_walkers))
        if return_final_state_ids:
            return out, np.empty(n_walkers, dtype=np.int32)
        return out

    markov_order = len(states[0]) - 1
    hist_len = markov_order + 1

    out_dtype = np.asarray(states[0]).dtype

    if sequence_length <= 0:
        out = np.empty((0, n_walkers), dtype=out_dtype)
        if return_final_state_ids:
            if initial_state_ids is None:
                final_state_ids = rng.integers(0, n_states, size=n_walkers, dtype=np.int32)
            else:
                final_state_ids = np.asarray(initial_state_ids, dtype=np.int32)
            return out, final_state_ids
        return out

    cdfs = [None] * n_states
    next_state_ids = [None] * n_states
    last_pos = np.empty(n_states, dtype=out_dtype)

    for sid, state in enumerate(states):
        probs, neighbors = markov_graph[state]

        probs = np.asarray(probs, dtype=np.float64)
        neighbors = np.asarray(neighbors)

        cdf = np.cumsum(probs)
        cdf[-1] = 1.0

        suffix = state[1:]

        next_ids = np.empty(len(neighbors), dtype=np.int32)

        for j, nxt in enumerate(neighbors):
            next_state = suffix + (nxt,)
            next_ids[j] = state_to_id[next_state]

        cdfs[sid] = cdf
        next_state_ids[sid] = next_ids
        last_pos[sid] = state[-1]

    # ------------------------------------------------------------
    # Initial states
    # ------------------------------------------------------------
    fresh_start = initial_state_ids is None

    if fresh_start:
        state_ids = rng.integers(0, n_states, size=n_walkers, dtype=np.int32)
    else:
        state_ids = np.asarray(initial_state_ids, dtype=np.int32)

        if len(state_ids) != n_walkers:
            raise ValueError(
                f"initial_state_ids has length {len(state_ids)}, "
                f"but n_walkers={n_walkers}"
            )

    # ------------------------------------------------------------
    # Output array
    # ------------------------------------------------------------
    out = np.empty((sequence_length, n_walkers), dtype=out_dtype)

    if fresh_start:
        # For the first batch, we can output the full initial histories.
        init_histories = np.array(
            [states[sid] for sid in state_ids],
            dtype=out_dtype,
        )

        n_init = min(sequence_length, hist_len)
        out[:n_init] = init_histories[:, :n_init].T

        start_t = n_init

        if sequence_length <= hist_len:
            if return_final_state_ids:
                return out, state_ids
            return out

    else:
        # For continuation batches, the current state already represents
        # the walkers' current histories. Output the current physical node.
        out[0] = last_pos[state_ids]
        start_t = 1

    # ------------------------------------------------------------
    # Advance walkers
    # ------------------------------------------------------------
    for t in tqdm(range(start_t, sequence_length), disable=disable_tqdm):
        new_state_ids = np.empty_like(state_ids)
        u = rng.random(n_walkers)

        for i, sid in enumerate(state_ids):
            idx = np.searchsorted(cdfs[sid], u[i])
            new_state_ids[i] = next_state_ids[sid][idx]

        state_ids = new_state_ids
        out[t] = last_pos[state_ids]

    if return_final_state_ids:
        return out, state_ids

    return out


def build_history_graph(histories, graph):
    """
    Build a directed graph from k+1-length histories.

    Parameters
    ----------
    histories : list of tuples
        Each tuple is a valid path of length k+1
    graph : initial graph

    Returns
    -------
    G : nx.DiGraph
        Directed graph where nodes = histories, edges = allowed transitions
    """
    g = nx.DiGraph()
    history_set = set(histories)  # for fast lookup
    for hist in histories:
        last_node = hist[-1]
        list_neighbors = list(graph.neighbors(last_node))+[last_node]
        for neighbor in list_neighbors:
            # New history after moving one step
            new_hist = hist[1:] + (neighbor,)
            # Only connect if new history is a valid history
            if new_hist in history_set:
                g.add_edge(hist, new_hist)
    return g


def generate_contact_data_from_graph(graph, n_walkers=5, markov_order=0, sequence_length=1000,
                                     sparsity=0.8, return_final_state_ids=False):
    markov_graph = build_kth_order_markov_graph(graph, markov_order, sparsity)
    positions = run_walkers(markov_graph, sequence_length, n_walkers, return_final_state_ids)
    contact_seq = generate_contact_edges(positions)
    return {"graph": graph, "markov_graph": markov_graph, "order": markov_order,
            "contacts": contact_seq, "positions": positions}


def average_metric(contact_seq, start_p=0, max_p=5):
    length_seq = len(contact_seq)
    edge_dict = build_edge_dict(contact_seq)
    all_values = []

    for pair in tqdm(edge_dict, desc="Computing ratio for all pairs"):
        vals = ratio_metric(edge_dict[pair], start_p=start_p, max_p=max_p, max_length=length_seq)
        all_values.append(vals)
    avg_vals = np.mean(np.array(all_values, dtype=float), axis=0)
    return avg_vals


def estimate_order(values, threshold=0.01, start_p=1):
    for idx, val in enumerate(values):
        if val < threshold:
            return idx  
    return start_p + len(values) - 2  # max p if never below threshold


def build_edge_dict(contact_seq):
    edge_dict = defaultdict(list)
    for t, edges in enumerate(contact_seq):
        for pair in edges:  # each pair is (i,j) with i<j
            edge_dict[pair].append(t)

    return dict(edge_dict)


def generate_contact_edges(position_seq):
    """
    Returns list of edge lists per timestep.
    Each element is list of (i, j) contacts.
    """
    T, N_w = position_seq.shape
    contact_edges = []

    for t in range(T):
        groups = defaultdict(list)
        # group walkers by position
        for i, pos in enumerate(position_seq[t]):
            groups[pos].append(i)
        edges_t = []

        # create edges within each group
        for walkers in groups.values():
            if len(walkers) > 1:
                for i in range(len(walkers)):
                    for j in range(i+1, len(walkers)):
                        edges_t.append((walkers[i], walkers[j]))
        contact_edges.append(edges_t)

    return contact_edges


def conditional_prob_optimized(y_indices, p, total_length):
    """
    Correct and optimized conditional probability for sparse Y (m << T).
    """
    y_indices = np.asarray(y_indices)
    y_set = set(y_indices)
    m = len(y_indices)

    if total_length <= 0:
        return 0.0
    if p == 0:
        return m / total_length
    if m < p:
        return 0.0

    count_total = 0
    count_cond = 0

    # Slide over sorted Y_indices
    for i in range(m - p + 1):
        window = y_indices[i:i + p]
        # Check if the p positions are consecutive in the full sequence
        if all(window[j] == window[0] + j for j in range(p)):
            t = window[-1] + 1
            if t >= total_length:
                continue
            count_total += 1
            if t in y_set:
                count_cond += 1
    return count_cond / count_total if count_total > 0 else 0.0


def ratio_metric(y_sequence, start_p=0, max_p=5, max_length=None, tol=1e-30):
    deviations = []
    for p in range(start_p, max_p + 1):
        p_curr = conditional_prob_optimized(y_sequence, p, max_length)
        p_prev = conditional_prob_optimized(y_sequence, p - 1, max_length)
        ratio_prob = (p_curr - p_prev)**2 / ((p_prev + tol) * (p_curr + tol))

        deviations.append(abs(ratio_prob))
    return deviations

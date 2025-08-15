import torch

def topological_sort_batch(PN, adj_matrix, N_max, device, topological_sort_single_fn):
    """
    Vectorized Kahn's algorithm across P genomes.
    Returns topo_order: [P, N_max] where unused positions are -1.
    Falls back to topological_sort_single_fn for incomplete orders.
    """
    P = PN.shape[0]
    N = N_max

    # Active nodes mask from PN marker
    active_nodes = (PN[:, :, 0] != 0)  # [P, N]

    # Compute in-degree (incoming edges from active sources)
    adj_active = adj_matrix & active_nodes.unsqueeze(1)  # require source active
    in_degree = adj_active.sum(dim=1).float()  # [P, N]
    in_degree = torch.where(active_nodes, in_degree, torch.full_like(in_degree, -1.0))

    topo_order = torch.full((P, N), -1, dtype=torch.long, device=device)
    pos = torch.zeros(P, dtype=torch.long, device=device)

    arange_nodes = torch.arange(N, device=device).unsqueeze(0).expand(P, N)  # [P, N]

    for step in range(N):
        ready = in_degree == 0.0
        if not ready.any():
            break

        masked = torch.where(ready, arange_nodes, torch.full_like(arange_nodes, N))
        node_idx = masked.min(dim=1).values
        has_ready = node_idx < N
        if not has_ready.any():
            break

        idxs = torch.nonzero(has_ready, as_tuple=True)[0]
        topo_order[idxs, pos[idxs]] = node_idx[idxs]
        pos[idxs] += 1

        in_degree[idxs, node_idx[idxs]] = -1.0
        children = adj_matrix[idxs, node_idx[idxs], :]  # [k, N]
        in_degree[idxs] = in_degree[idxs] - children.float()

    processed_counts = pos
    expected_counts = active_nodes.sum(dim=1).long()
    incomplete = (processed_counts < expected_counts)

    if incomplete.any():
        incomplete_idxs = torch.nonzero(incomplete, as_tuple=True)[0].cpu().numpy().tolist()
        for gi in incomplete_idxs:
            try:
                single_order = topological_sort_single_fn(int(gi))
            except ValueError as e:
                raise ValueError(f"Cycle detected in genome {gi}") from e
            k = len(single_order)
            if k > 0:
                topo_order[gi, :k] = torch.tensor(single_order, dtype=torch.long, device=device)

    return topo_order


def topological_sort_single(PN, PC, enabled, genome_idx, N_max):
    """
    Fall-back single-genome Kahn (keeps previous behavior).
    """
    active_nodes = (PN[genome_idx, :, 0] != 0)
    active_indices = active_nodes.nonzero(as_tuple=True)[0].cpu().numpy().tolist()

    adj_list = {int(node): [] for node in active_indices}
    in_degree = {int(node): 0 for node in active_indices}

    en = enabled[genome_idx]
    conns = PC[genome_idx, en]
    for conn in conns:
        src, dst = int(conn[0].item()), int(conn[1].item())
        if src in adj_list and dst in adj_list:
            adj_list[src].append(dst)
            in_degree[dst] += 1

    queue = [node for node in active_indices if in_degree[int(node)] == 0]
    queue.sort()
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for nb in adj_list[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)
                queue.sort()

    if len(topo_order) != len(active_indices):
        raise ValueError(f"Cycle detected in genome {genome_idx}")
    return topo_order

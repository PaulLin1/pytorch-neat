from topological_sort import topological_sort_batch, topological_sort_single
import torch

class NEAT:
    def __init__(self, P, N_max, C_max, num_inputs, num_outputs, device='cpu'):
        """
        Optimized, vectorized Genome for batched NEAT-like forward passes.
        Major optimizations:
         - explicit active node marker (PN[...,0] == 1 means active)
         - caches connection arrays (input_idx, output_idx, enabled, weight)
         - vectorized batch topological sort (no python loop over genomes)
         - vectorized forward: for each topo step gather inputs for all genomes at once
        """
        self.P = P
        self.N_max = N_max
        self.C_max = C_max
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.device = device

        # Node features: [marker, bias]
        # marker: 1.0 active node, 0.0 inactive
        self.PN = torch.zeros((P, N_max, 2), device=device)
        # Mark inputs and outputs active
        self.PN[:, :num_inputs, 0] = 1.0
        self.PN[:, -num_outputs:, 0] = 1.0

        # small random biases for inputs and outputs
        self.PN[:, :num_inputs, 1] = 0.01 * torch.randn((P, num_inputs), device=device)
        self.PN[:, -num_outputs:, 1] = 0.01 * torch.randn((P, num_outputs), device=device)

        # connection tensor: [input_idx, output_idx, enabled_flag, weight]
        self.PC = torch.zeros((P, C_max, 4), device=device)
        # node values placeholder
        self.node_vals = torch.zeros((P, N_max), device=device)  # not strictly necessary at init

        # initialize fully-connected input->output connections (as many as C_max allows)
        for i in range(P):
            conn_features = torch.zeros((C_max, 4), device=device)
            full_connections = []
            for inp in range(num_inputs):
                for out in range(N_max - num_outputs, N_max):
                    full_connections.append((inp, out))
            if len(full_connections) > 0:
                full_connections = torch.tensor(full_connections, device=device, dtype=torch.long)
                n_full = full_connections.shape[0]
                c_assign = min(C_max, n_full)
                conn_features[:c_assign, 0:2] = full_connections[:c_assign].float()
                conn_features[:c_assign, 2] = 1.0  # enabled
                conn_features[:c_assign, 3] = 0.1 * torch.randn(c_assign, device=device)
            self.PC[i] = conn_features

        # Cache connection arrays to avoid re-extracting them each call
        self._cache_connection_arrays()

    def _cache_connection_arrays(self):
        # Shapes: [P, C_max]
        self.enabled = (self.PC[:, :, 2] != 0)
        # ensure integer indices for gathers
        self.input_idx = self.PC[:, :, 0].long().clamp(0, self.N_max - 1)
        self.output_idx = self.PC[:, :, 1].long().clamp(0, self.N_max - 1)
        self.weights = self.PC[:, :, 3]  # float

        # For adjacency boolean matrix (used in topo sort)
        # adj_matrix[p, i, j] == True if connection i->j exists and enabled
        P = self.P; N = self.N_max; C = self.C_max
        adj = torch.zeros((P, N, N), dtype=torch.bool, device=self.device)
        # broadcasting trick: produce genome index shape [P, C]
        gidx = torch.arange(P, device=self.device).unsqueeze(1).expand(P, C)  # [P, C]
        # Use scatter_ to avoid fancy indexing assignment issues
        # We'll flatten indices and use scatter for a safe vectorized set
        flat_src = self.input_idx.reshape(-1)       # [P*C]
        flat_dst = self.output_idx.reshape(-1)      # [P*C]
        flat_enabled = self.enabled.reshape(-1)     # [P*C]
        flat_g = gidx.reshape(-1)                   # [P*C]

        # Filter only enabled connections to reduce work
        enabled_positions = flat_enabled.nonzero(as_tuple=True)[0]
        if enabled_positions.numel() > 0:
            gg = flat_g[enabled_positions]
            ss = flat_src[enabled_positions]
            dd = flat_dst[enabled_positions]
            adj[gg, ss, dd] = True
        self.adj_matrix = adj  # [P, N, N]

    def topological_sort_batch(self):
        return topological_sort_batch(
            self.PN,
            self.adj_matrix,
            self.N_max,
            self.device,
            lambda gi: self.topological_sort_single(gi)
        )

    def topological_sort_single(self, genome_idx: int):
        return topological_sort_single(
            self.PN,
            self.PC,
            self.enabled,
            genome_idx,
            self.N_max
        )


    def forward(self, X):
        """
        Vectorized forward pass.
        X: [B, num_inputs] (batch over data samples)
        Returns: outputs for each genome: [B, P, num_outputs] (matching original return)
        """
        device = self.device
        B = X.shape[0]
        P = self.P
        N = self.N_max
        C = self.C_max

        # nodes values: [B, P, N]
        node_vals = torch.zeros((B, P, N), device=device)
        # set input node values for each genome (assuming inputs are nodes [0:num_inputs])
        # Expand X across genomes
        node_vals[:, :, :self.num_inputs] = X.unsqueeze(1).expand(B, P, self.num_inputs)

        # get topo order for all genomes (includes inputs). -1 indicates padding/inactive
        topo_batch = self.topological_sort_batch()  # [P, N]

        # Permute node_vals for easier per-genome gather operations:
        # nv : [P, B, N]
        nv = node_vals.permute(1, 0, 2).contiguous()  # will be updated in-place (affects node_vals view)

        # pre-expand indices for gather: [P, B, C]
        input_idx_expanded = self.input_idx.unsqueeze(1).expand(P, B, C)  # long
        weights_expanded = self.weights.unsqueeze(1)  # [P, 1, C] -> broadcasts to [P, B, C]

        # gather input node values for all connections ONCE: nv is [P, B, N], index along last dim
        # result: [P, B, C]
        inputs_per_conn = torch.gather(nv, 2, input_idx_expanded)  # [P, B, C]

        # process topological positions (skip nodes that are inputs)
        for pos in range(N):
            node_ids = topo_batch[:, pos]  # [P], may contain -1 or input nodes
            # valid indicates we should compute this node (non-input and active)
            valid = (node_ids >= self.num_inputs) & (node_ids >= 0)
            if not valid.any():
                continue

            node_ids_clamped = torch.where(valid, node_ids, torch.zeros_like(node_ids))

            # build connection mask for these target nodes:
            # mask_conn[p, c] == True if connection c of genome p targets node node_ids_clamped[p]
            mask_conn = (self.output_idx == node_ids_clamped.unsqueeze(1)) & self.enabled  # [P, C] boolean

            # multiply inputs by weights and zero-out non-target-connections using mask_conn
            contribs = inputs_per_conn * weights_expanded  # [P, B, C]
            contribs = contribs * mask_conn.unsqueeze(1)  # [P, B, C]

            # sum contributions across connections -> weighted sum for the target node (per-genome, per-batch)
            weighted_sum = contribs.sum(dim=2)  # [P, B]

            # get biases for target nodes: PN[:, node_ids_clamped, 1] with advanced indexing
            p_idx = torch.arange(P, device=device)
            biases = self.PN[p_idx, node_ids_clamped, 1]  # [P]

            # compute activations: [B, P]
            activated = torch.sigmoid(weighted_sum.transpose(0, 1) + biases.unsqueeze(0))  # [B, P]

            # assign back into nv at the proper node slot for each genome
            valid_indices = torch.nonzero(valid, as_tuple=True)[0]  # [k]
            if valid_indices.numel() == 0:
                continue

            nodes_to_set = node_ids_clamped[valid_indices]  # [k]
            # advanced-index assignment: pairs each genome in valid_indices with its node in nodes_to_set
            nv[valid_indices, :, nodes_to_set] = activated[:, valid_indices].transpose(0, 1)

        # after processing, node_vals is updated in-place because nv is a permuted view
        node_vals = nv.permute(1, 0, 2).contiguous()  # [B, P, N]

        # return last num_outputs nodes (assuming outputs at the end of node index space)
        outputs = node_vals[:, :, -self.num_outputs:]  # [B, P, num_outputs]
        return outputs

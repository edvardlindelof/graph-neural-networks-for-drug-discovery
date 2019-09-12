import torch
from torch import nn


class EMN(nn.Module):

    def __init__(self, edge_features, edge_embedding_size, message_passes, out_features):
        super(EMN, self).__init__()
        self.edge_features = edge_features
        self.edge_embedding_size = edge_embedding_size
        self.message_passes = message_passes
        self.out_features = out_features

    def preprocess_edges(self, nodes, node_neighbours, edges):
        raise NotImplementedError

    # (total number of edges in batch, edge_features) and (total number of edges in batch, max_node_degree, edge_features)
    def propagate_edges(self, edges, ingoing_edge_memories, ingoing_edges_mask):
        raise NotImplementedError

    def readout(self, hidden_nodes, input_nodes, node_mask):
        raise NotImplementedError

    # adjacency (N, n_nodes, n_nodes); edges (N, n_nodes, n_nodes, edge_features)
    def forward(self, adjacency, nodes, edges):
        # indices for finding edges in batch
        edges_b_idx, edges_n_idx, edges_nhb_idx = adjacency.nonzero().unbind(-1)

        n_edges = edges_n_idx.shape[0]
        adj_of_edge_batch_indices = adjacency.clone().long()
        r = torch.arange(n_edges) + 1  # +1 to distinguish the index 0 from 'empty' elements, subtracted few lines down
        if next(self.parameters()).is_cuda:
            r = r.cuda()
        adj_of_edge_batch_indices[edges_b_idx, edges_n_idx, edges_nhb_idx] = r

        ingoing_edges_eb_idx = (torch.cat([
            row[row.nonzero()] for row in adj_of_edge_batch_indices[edges_b_idx, edges_nhb_idx, :]
        ]) - 1).squeeze()

        edge_degrees = adjacency[edges_b_idx, edges_nhb_idx, :].sum(-1).long()
        ingoing_edges_igeb_idx = torch.cat([i * torch.ones(d) for i, d in enumerate(edge_degrees)]).long()
        ingoing_edges_ige_idx = torch.cat([torch.arange(i) for i in edge_degrees]).long()

        batch_size = adjacency.shape[0]
        n_nodes = adjacency.shape[1]
        max_node_degree = adjacency.sum(-1).max().int()
        edge_memories = torch.zeros(n_edges, self.edge_embedding_size)
        ingoing_edge_memories = torch.zeros(n_edges, max_node_degree, self.edge_embedding_size)
        ingoing_edges_mask = torch.zeros(n_edges, max_node_degree)
        if next(self.parameters()).is_cuda:
            edge_memories = edge_memories.cuda()
            ingoing_edge_memories = ingoing_edge_memories.cuda()
            ingoing_edges_mask = ingoing_edges_mask.cuda()

        edge_batch_nodes = nodes[edges_b_idx, edges_n_idx, :]
        edge_batch_neighbours = nodes[edges_b_idx, edges_nhb_idx, :]
        edge_batch_edges = edges[edges_b_idx, edges_n_idx, edges_nhb_idx, :]
        edge_batch_edges = self.preprocess_edges(edge_batch_nodes, edge_batch_neighbours, edge_batch_edges)

        # remove h_ji:s influence on h_ij
        ingoing_edges_nhb_idx = edges_nhb_idx[ingoing_edges_eb_idx]
        ingoing_edges_receiving_edge_n_idx = edges_n_idx[ingoing_edges_igeb_idx]
        not_same_idx = (ingoing_edges_receiving_edge_n_idx != ingoing_edges_nhb_idx).nonzero()
        ingoing_edges_eb_idx = ingoing_edges_eb_idx[not_same_idx].squeeze()
        ingoing_edges_ige_idx = ingoing_edges_ige_idx[not_same_idx].squeeze()
        ingoing_edges_igeb_idx = ingoing_edges_igeb_idx[not_same_idx].squeeze()

        ingoing_edges_mask[ingoing_edges_igeb_idx, ingoing_edges_ige_idx] = 1

        for i in range(self.message_passes):
            ingoing_edge_memories[ingoing_edges_igeb_idx, ingoing_edges_ige_idx, :] = \
                edge_memories[ingoing_edges_eb_idx, :]
            edge_memories = self.propagate_edges(edge_batch_edges, ingoing_edge_memories.clone(), ingoing_edges_mask)

        node_mask = (adjacency.sum(-1) != 0)

        node_sets = torch.zeros(batch_size, n_nodes, max_node_degree, self.edge_embedding_size)
        if next(self.parameters()).is_cuda:
            node_sets = node_sets.cuda()

        edge_batch_edge_memory_indices = torch.cat(
            [torch.arange(row.sum()) for row in adjacency.view(-1, n_nodes)]
        ).long()

        node_sets[edges_b_idx, edges_n_idx, edge_batch_edge_memory_indices, :] = edge_memories
        graph_sets = node_sets.sum(2)
        output = self.readout(graph_sets, graph_sets, node_mask)

        return output

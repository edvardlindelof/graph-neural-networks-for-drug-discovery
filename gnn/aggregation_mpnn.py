import torch
from torch import nn


class AggregationMPNN(nn.Module):

    def __init__(self, node_features, edge_features, message_size, message_passes, out_features):
        super(AggregationMPNN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.message_size = message_size
        self.message_passes = message_passes
        self.out_features = out_features

    # nodes (total number of nodes in batch, number of features)
    # node_neighbours (total number of nodes in batch, max node degree, number of features)
    # node_neighbours (total number of nodes in batch, max node degree, number of edge features)
    # mask (total number of nodes in batch, max node degree) elements are 1 if corresponding neighbour exist
    def aggregate_message(self, nodes, node_neighbours, edges, mask):
        raise NotImplementedError

    # inputs are "batches" of shape (maximum number of nodes in batch, number of features)
    def update(self, nodes, messages):
        raise NotImplementedError

    # inputs are "batches" of same shape as the nodes passed to update
    # node_mask is same shape as inputs and is 1 if elements corresponding exists, otherwise 0
    def readout(self, hidden_nodes, input_nodes, node_mask):
        raise NotImplementedError

    def forward(self, adjacency, nodes, edges):
        edge_batch_batch_indices, edge_batch_node_indices, edge_batch_neighbour_indices = adjacency.nonzero().unbind(-1)

        node_batch_batch_indices, node_batch_node_indices = adjacency.sum(-1).nonzero().unbind(-1)
        node_batch_adj = adjacency[node_batch_batch_indices, node_batch_node_indices, :]

        node_batch_size = node_batch_batch_indices.shape[0]
        node_degrees = node_batch_adj.sum(-1).long()
        max_node_degree = node_degrees.max()
        node_batch_node_neighbours = torch.zeros(node_batch_size, max_node_degree, self.node_features)
        node_batch_edges = torch.zeros(node_batch_size, max_node_degree, self.edge_features)

        node_batch_neighbour_neighbour_indices = torch.cat([torch.arange(i) for i in node_degrees])

        edge_batch_node_batch_indices = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees)]
        ).long()

        node_batch_node_neighbour_mask = torch.zeros(node_batch_size, max_node_degree)

        if next(self.parameters()).is_cuda:
            node_batch_node_neighbours = node_batch_node_neighbours.cuda()
            node_batch_edges = node_batch_edges.cuda()
            node_batch_neighbour_neighbour_indices = node_batch_neighbour_neighbour_indices.cuda()
            edge_batch_node_batch_indices = edge_batch_node_batch_indices.cuda()
            node_batch_node_neighbour_mask = node_batch_node_neighbour_mask.cuda()

        node_batch_node_neighbour_mask[edge_batch_node_batch_indices, node_batch_neighbour_neighbour_indices] = 1

        node_batch_edges[edge_batch_node_batch_indices, node_batch_neighbour_neighbour_indices, :] = \
            edges[edge_batch_batch_indices, edge_batch_node_indices, edge_batch_neighbour_indices, :]

        hidden_nodes = nodes.clone()

        for i in range(self.message_passes):
            node_batch_nodes = hidden_nodes[node_batch_batch_indices, node_batch_node_indices, :]
            node_batch_node_neighbours[edge_batch_node_batch_indices, node_batch_neighbour_neighbour_indices, :] = \
                hidden_nodes[edge_batch_batch_indices, edge_batch_neighbour_indices, :]

            messages = self.aggregate_message(
                node_batch_nodes, node_batch_node_neighbours.clone(), node_batch_edges, node_batch_node_neighbour_mask
            )
            hidden_nodes[node_batch_batch_indices, node_batch_node_indices, :] = self.update(node_batch_nodes, messages)

        node_mask = (adjacency.sum(-1) != 0)#.unsqueeze(-1).expand_as(nodes)
        output = self.readout(hidden_nodes, nodes, node_mask)
        return output

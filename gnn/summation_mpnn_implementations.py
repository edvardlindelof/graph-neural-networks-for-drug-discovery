import torch
from torch import nn

from gnn.modules import GraphGather, FeedForwardNetwork, Set2Vec
from gnn.summation_mpnn import SummationMPNN


class ENNS2V(SummationMPNN):

    def __init__(self, node_features, edge_features, message_size, message_passes, out_features,
                 enn_depth=4, enn_hidden_dim=200, enn_dropout_p=0,
                 s2v_lstm_computations=12, s2v_memory_size=50,
                 out_depth=1, out_hidden_dim=200, out_dropout_p=0):
        super(ENNS2V, self).__init__(node_features, edge_features, message_size, message_passes, out_features)

        self.enn = FeedForwardNetwork(
            edge_features, [enn_hidden_dim] * enn_depth, node_features * message_size, dropout_p=enn_dropout_p
        )
        self.gru = nn.GRUCell(input_size=message_size, hidden_size=node_features, bias=False)
        self.s2v = Set2Vec(node_features, s2v_lstm_computations, s2v_memory_size)
        self.out_nn = FeedForwardNetwork(
            s2v_memory_size * 2, [out_hidden_dim] * out_depth, out_features, dropout_p=out_dropout_p, bias=False
        )

    def message_terms(self, nodes, node_neighbours, edges):
        enn_output = self.enn(edges)
        matrices = enn_output.view(-1, self.message_size, self.node_features)
        msg_terms = torch.matmul(matrices, node_neighbours.unsqueeze(-1)).squeeze(-1)
        return msg_terms

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.s2v(hidden_nodes, input_nodes, node_mask)
        return self.out_nn(graph_embeddings)


class GGNN(SummationMPNN):

    def __init__(self, node_features, edge_features, message_size, message_passes, out_features,
                 msg_depth=4, msg_hidden_dim=200, msg_dropout_p=0.0,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100, gather_att_dropout_p=0.0,
                 gather_emb_depth=3, gather_emb_hidden_dim=100, gather_emb_dropout_p=0.0,
                 out_depth=2, out_hidden_dim=100, out_dropout_p=0.0, out_layer_shrinkage=1.0):
        super(GGNN, self).__init__(node_features, edge_features, message_size, message_passes, out_features)

        self.msg_nns = nn.ModuleList()
        for _ in range(edge_features):
            self.msg_nns.append(
                FeedForwardNetwork(node_features, [msg_hidden_dim] * msg_depth, message_size, dropout_p=msg_dropout_p, bias=False)
            )
        self.gru = nn.GRUCell(input_size=message_size, hidden_size=node_features, bias=False)
        self.gather = GraphGather(
            node_features, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )
        out_layer_sizes = [  # example: depth 5, dim 50, shrinkage 0.5 => out_layer_sizes [50, 42, 35, 30, 25]
            round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width, out_layer_sizes, out_features, dropout_p=out_dropout_p)

    def message_terms(self, nodes, node_neighbours, edges):
        # terms_masked_per_edge contains (edge_batch_size, message_size)-shape tensors, that has 0s in all rows except
        # the ones corresponding to the edge type indicated by the list index
        # intuitive way of writing this involves a torch.stack along batch dimension and is immensely slow
        edges_v = edges.view(-1, self.edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(-1, 1, self.node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :]) for i in range(self.edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return self.out_nn(graph_embeddings)
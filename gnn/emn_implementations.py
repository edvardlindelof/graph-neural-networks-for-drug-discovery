import torch
from torch import nn

from gnn.emn import EMN
from gnn.modules import FeedForwardNetwork, GraphGather


class EMNImplementation(EMN):

    def __init__(self, node_features, edge_features, message_passes, out_features,
                 edge_embedding_size,
                 edge_emb_depth=3, edge_emb_hidden_dim=150, edge_emb_dropout_p=0.0,
                 att_depth=3, att_hidden_dim=80, att_dropout_p=0.0,
                 msg_depth=3, msg_hidden_dim=80, msg_dropout_p=0.0,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100, gather_att_dropout_p=0.0,
                 gather_emb_depth=3, gather_emb_hidden_dim=100, gather_emb_dropout_p=0.0,
                 out_depth=2, out_hidden_dim=100, out_dropout_p=0, out_layer_shrinkage=1.0):
        super(EMNImplementation, self).__init__(
            edge_features, edge_embedding_size, message_passes, out_features
        )
        self.embedding_nn = FeedForwardNetwork(
            node_features * 2 + edge_features, [edge_emb_hidden_dim] * edge_emb_depth, edge_embedding_size, dropout_p=edge_emb_dropout_p
        )

        self.emb_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [msg_hidden_dim] * msg_depth, edge_embedding_size, dropout_p=msg_dropout_p
        )
        self.att_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [att_hidden_dim] * att_depth, edge_embedding_size, dropout_p=att_dropout_p
        )

        #self.extra_gru_layer = nn.Linear(edge_embedding_size, edge_embedding_size, bias=False)
        self.gru = nn.GRUCell(edge_embedding_size, edge_embedding_size, bias=False)
        self.gather = GraphGather(
            edge_embedding_size, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )
        out_layer_sizes = [  # example: depth 5, dim 50, shrinkage 0.5 => out_layer_sizes [50, 42, 35, 30, 25]
            round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width, out_layer_sizes, out_features, dropout_p=out_dropout_p)

    def preprocess_edges(self, nodes, node_neighbours, edges):
        cat = torch.cat([nodes, node_neighbours, edges], dim=1)
        return torch.tanh(self.embedding_nn(cat))

    def propagate_edges(self, edges, ingoing_edge_memories, ingoing_edges_mask):
        BIG_NEGATIVE = -1e6
        energy_mask = ((1 - ingoing_edges_mask).float() * BIG_NEGATIVE).unsqueeze(-1)

        cat = torch.cat([edges.unsqueeze(1), ingoing_edge_memories], dim=1)
        embeddings = self.emb_msg_nn(cat)

        edge_energy = self.att_msg_nn(edges)
        ing_memory_energies = self.att_msg_nn(ingoing_edge_memories) + energy_mask
        energies = torch.cat([edge_energy.unsqueeze(1), ing_memory_energies], dim=1)
        attention = torch.softmax(energies, dim=1)

        # set aggregation of the set of the given edge feature and ingoing edge memories
        message = (attention * embeddings).sum(dim=1)
        return self.gru(message)  # returning hidden state but it is also set internally I think.. hm

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return self.out_nn(graph_embeddings)

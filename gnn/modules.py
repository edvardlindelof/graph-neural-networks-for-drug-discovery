import torch
from torch import nn

import math


class GraphGather(nn.Module):
    r"""The GGNN readout function
    """

    def __init__(self, node_features, out_features,
                 att_depth=2, att_hidden_dim=100, att_dropout_p=0.0,
                 emb_depth=2, emb_hidden_dim=100, emb_dropout_p=0.0):
        super(GraphGather, self).__init__()

        # denoted i and j in GGNN, MPNN and PotentialNet papers
        self.att_nn = FeedForwardNetwork(
            node_features * 2, [att_hidden_dim] * att_depth, out_features, dropout_p=att_dropout_p, bias=False
        )
        self.emb_nn = FeedForwardNetwork(
            node_features, [emb_hidden_dim] * emb_depth, out_features, dropout_p=emb_dropout_p, bias=False
        )

    def forward(self, hidden_nodes, input_nodes, node_mask):
        cat = torch.cat([hidden_nodes, input_nodes], dim=2)
        energy_mask = (node_mask == 0).float() * 1e6
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = torch.sigmoid(energies)
        #attention = torch.softmax(energies, dim=1)
        embedding = self.emb_nn(hidden_nodes)
        return torch.sum(attention * embedding, dim=1)


class Set2Vec(nn.Module):
    r"""The readout function of MPNN paper's best network
    """

    # used to set attention terms to 0 when passing energies to softmax
    # tf code uses same trick
    BIG_NEGATIVE = -1e6

    def __init__(self, node_features, lstm_computations, memory_size):
        super(Set2Vec, self).__init__()

        self.lstm_computations = lstm_computations
        self.memory_size = memory_size

        self.embedding_matrix = nn.Linear(node_features * 2, self.memory_size, bias=False)
        self.lstm = nn.LSTMCell(self.memory_size, self.memory_size, bias=False)

    def forward(self, hidden_output_nodes, input_nodes, node_mask):
        batch_size = input_nodes.shape[0]
        energy_mask = (1 - node_mask).float() * self.BIG_NEGATIVE

        lstm_input = torch.zeros(batch_size, self.memory_size)

        cat = torch.cat([hidden_output_nodes, input_nodes], dim=2)
        memory = self.embedding_matrix(cat)

        hidden_state = torch.zeros(batch_size, self.memory_size)
        cell_state = torch.zeros(batch_size, self.memory_size)

        if next(self.parameters()).is_cuda:
            lstm_input = lstm_input.cuda()
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        for i in range(self.lstm_computations):
            query, cell_state = self.lstm(lstm_input, (hidden_state, cell_state))
            # dot product query x memory
            energies = (query.view(batch_size, 1, self.memory_size) * memory).sum(dim=-1)
            attention = torch.softmax(energies + energy_mask, dim=1)
            read = (attention.unsqueeze(-1) * memory).sum(dim=1)

            hidden_state = query
            lstm_input = read

        cat = torch.cat([query, read], dim=1)
        return cat


class FeedForwardNetwork(nn.Module):
    r"""Convenience class to create network composed of linear layers with an activation function
    applied between them

    Args:
        in_features: size of each input sample
        hidden_layer_sizes: list of hidden layer sizes
        out_features: size of each output sample
        activation: 'SELU' or 'ReLU'
        bias: If set to False, the layers will not learn an additive bias.
            Default: ``False``
    """

    def __init__(self, in_features, hidden_layer_sizes, out_features, activation='SELU', bias=False, dropout_p=0.0):
        super(FeedForwardNetwork, self).__init__()

        if activation == 'SELU':
            Activation = nn.SELU
            Dropout = nn.AlphaDropout
            init_constant = 1.0
        elif activation == 'ReLU':
            Activation = nn.ReLU
            Dropout = nn.Dropout
            init_constant = 2.0

        layer_sizes = [in_features] + hidden_layer_sizes + [out_features]

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(Dropout(dropout_p))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias))
            layers.append(Activation())
        layers.append(Dropout(dropout_p))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias))

        self.seq = nn.Sequential(*layers)

        for i in range(1, len(layers), 3):
            # initialization recommended in SELU paper
            nn.init.normal_(layers[i].weight, std=math.sqrt(init_constant / layers[i].weight.size(1)))

    def forward(self, input):
        return self.seq(input)

    # I'm probably *supposed to* override extra_repr but then self.seq (unreadable) will be printed too
    def __repr__(self):
        ffnn = type(self).__name__
        in_features = self.seq[1].in_features
        hidden_layer_sizes = [linear.out_features for linear in self.seq[1:-1:3]]
        out_features = self.seq[-1].out_features
        if len(self.seq) > 2:
            activation = str(self.seq[2])
        else:
            activation = 'None'
        bias = self.seq[1].bias is not None
        dropout_p = self.seq[0].p
        return '{}(in_features={}, hidden_layer_sizes={}, out_features={}, activation={}, bias={}, dropout_p={})'.format(
            ffnn, in_features, hidden_layer_sizes, out_features, activation, bias, dropout_p
        )

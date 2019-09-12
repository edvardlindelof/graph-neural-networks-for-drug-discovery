import unittest

import numpy as np
import torch
from torch import optim, nn

from gnn.molgraph_data import molgraph_collate_fn, smile_to_graph
from gnn.summation_mpnn import SummationMPNN


# this script checks that the MPNN implementations can be initialized and trained for a few iterations
# without crashing, and that they fulfill the principles of invariance to node order, padding size
# and shuffled input order
#
# padding invariance is important because it indicates whether node vectors full of 0s (corresponding to
# non-existant nodes) affects the output
#
# node order invariance is important because it shold in principle not matter
#
# add a newly implemented MPNN by extending MPNNTestCase and overrding it's member net, see bottom of file

# some graphs generated that 1) are small enough so that the tensors are readable and
# 2) have different size adjacency matrices, to assure molgraph_collate_fn:s padding is being used


COMPOUNDS = ['OC(=O)[C@@H]1CCN1', 'BrC1=NC=CC=C1', 'ClCCOC=C']
BATCH_SIZE = len(COMPOUNDS)
DUMMY_ADJ, DUMMY_NODES, DUMMY_EDGES, DUMMY_TARGET = \
    molgraph_collate_fn(list(map(lambda smile: (smile_to_graph(smile), [1]), COMPOUNDS)))
NODE_FEATURES = 5
DUMMY_NODES = DUMMY_NODES[:, :, :NODE_FEATURES]  # dropping all node features except a few
EDGE_FEATURES = DUMMY_EDGES.shape[3]
OUT_FEATURES = DUMMY_TARGET.shape[1]


class DummyMPNN(SummationMPNN):

    def __init__(self, node_features, edge_features, message_size, message_passes, out_features):
        super(DummyMPNN, self).__init__(node_features, edge_features, message_size, message_passes, out_features)

        # breaks padding invariance, unless node_mask is used properly
        #self.readout_layer = nn.Linear(NODE_FEATURES, 1, bias=True)
        self.readout_layer = nn.Linear(NODE_FEATURES, 1, bias=False)

    def message_terms(self, nodes, node_neighbours, edges):
        message_terms = nodes + node_neighbours
        return message_terms

    def update(self, nodes, messages):
        return messages

    def readout(self, hidden_nodes, input_nodes, node_mask):
        output = self.readout_layer(hidden_nodes).sum(dim=1)
        return output


class GNNTestCase(unittest.TestCase):

    NODE_FEATURES = NODE_FEATURES
    EDGE_FEATURES = EDGE_FEATURES
    OUT_FEATURES = OUT_FEATURES

    # keep number of weights down to make this run fast
    MESSAGE_SIZE = 5
    MESSAGE_PASSES = 2

    net = DummyMPNN(NODE_FEATURES, EDGE_FEATURES, MESSAGE_SIZE, MESSAGE_PASSES, OUT_FEATURES)

    @classmethod
    def setUpClass(self):
        optimizer = optim.Adam(self.net.parameters(), lr=0.0005)
        criterion = nn.MSELoss()
        self.net.train()
        for i in range(10):
            self.net.zero_grad()
            output = self.net(DUMMY_ADJ, DUMMY_NODES, DUMMY_EDGES)
            loss = criterion(output, DUMMY_TARGET)
            loss.backward()
            optimizer.step()

    def test_padding_invariance(self):
        padded_dim_size = DUMMY_ADJ.shape[1] + 5
        padded_adj = torch.zeros(BATCH_SIZE, padded_dim_size, padded_dim_size)
        padded_adj[:, :DUMMY_ADJ.shape[1], :DUMMY_ADJ.shape[2]] = DUMMY_ADJ
        padded_nodes = torch.zeros(BATCH_SIZE, padded_dim_size, NODE_FEATURES)
        padded_nodes[:, :DUMMY_NODES.shape[1], :] = DUMMY_NODES
        padded_edges = torch.zeros(BATCH_SIZE, padded_dim_size, padded_dim_size, EDGE_FEATURES)
        padded_edges[:, :DUMMY_EDGES.shape[1], :DUMMY_EDGES.shape[2], :] = DUMMY_EDGES

        with torch.no_grad():
            self.net.eval()
            normal_output = self.net(DUMMY_ADJ, DUMMY_NODES, DUMMY_EDGES)
            extra_padding_output = self.net(padded_adj, padded_nodes, padded_edges)
            # consider outputs equal if difference is smaller than 0.001%
            # this is not always exact for whatever numerical reason
            self.assertTrue(np.allclose(normal_output, extra_padding_output, rtol=1e-5))

    def test_sample_order_invariance(self):
        permutation = [1, 2, 0]
        shuffled_adj = DUMMY_ADJ[permutation, :, :]
        shuffled_nodes = DUMMY_NODES[permutation, :, :]
        shuffled_edges = DUMMY_EDGES[permutation, :, :, :]

        with torch.no_grad():
            self.net.eval()
            output = self.net(DUMMY_ADJ, DUMMY_NODES, DUMMY_EDGES)
            shuffling_after_prop_output = output[permutation]
            shuffling_before_prop_output = self.net(shuffled_adj, shuffled_nodes, shuffled_edges)
            # consider outputs equal if difference is smaller than 0.001%
            # this is not always exact for whatever numerical reason
            self.assertTrue(np.allclose(shuffling_after_prop_output, shuffling_before_prop_output, rtol=1e-5))

    def test_node_order_invariance(self):
        shuffled_adj = torch.zeros_like(DUMMY_ADJ)
        shuffled_nodes = torch.zeros_like(DUMMY_NODES)
        shuffled_edges = torch.zeros_like(DUMMY_EDGES)
        for i in range(BATCH_SIZE):
            n_real_nodes = (DUMMY_ADJ[i, :, :].sum(dim=1) != 0).sum().item()
            perm = np.random.permutation(n_real_nodes).reshape(1, -1)
            perm_t = perm.transpose()
            shuffled_adj[i, :n_real_nodes, :n_real_nodes] = DUMMY_ADJ[i, perm, perm_t]
            shuffled_nodes[i, :n_real_nodes] = DUMMY_NODES[i, perm, :]
            shuffled_edges[i, :n_real_nodes, :n_real_nodes, :] = DUMMY_EDGES[i, perm, perm_t, :]

        with torch.no_grad():
            self.net.eval()
            normal_output = self.net(DUMMY_ADJ, DUMMY_NODES, DUMMY_EDGES)
            shuffling_output = self.net(shuffled_adj, shuffled_nodes, shuffled_edges)
            # consider outputs equal if difference is smaller than 0.001%
            # this is not always exact for whatever numerical reason
            self.assertTrue(np.allclose(normal_output, shuffling_output, rtol=1e-5))

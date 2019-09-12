import gnn.aggregation_mpnn_implementations
import gnn.emn_implementations
import gnn.summation_mpnn_implementations

from tests.gnn_test_case import GNNTestCase


NODE_FEATURES = GNNTestCase.NODE_FEATURES
EDGE_FEATURES = GNNTestCase.EDGE_FEATURES
MESSAGE_SIZE = GNNTestCase.MESSAGE_SIZE
MESSAGE_PASSES = GNNTestCase.MESSAGE_PASSES
OUT_FEATURES = GNNTestCase.OUT_FEATURES


class ENNS2VTestCase(GNNTestCase):
    net = gnn.summation_mpnn_implementations.ENNS2V(
        NODE_FEATURES, EDGE_FEATURES, MESSAGE_SIZE, MESSAGE_PASSES, OUT_FEATURES,
        enn_hidden_dim=4, out_hidden_dim=6, s2v_memory_size=5
    )

class GGNNTestCase(GNNTestCase):
    net = gnn.summation_mpnn_implementations.GGNN(
        NODE_FEATURES, EDGE_FEATURES, MESSAGE_SIZE, MESSAGE_PASSES, OUT_FEATURES,
        msg_hidden_dim=4, gather_width=7, gather_att_hidden_dim=9, gather_emb_hidden_dim=7, out_hidden_dim=3
    )

class AttentionENNS2VTestCase(GNNTestCase):
    net = gnn.aggregation_mpnn_implementations.AttentionENNS2V(
        NODE_FEATURES, EDGE_FEATURES, MESSAGE_SIZE, 10, OUT_FEATURES,
        enn_hidden_dim=4, out_hidden_dim=6, s2v_memory_size=5
    )

class AttentionGGNNTestCase(GNNTestCase):
    net = gnn.aggregation_mpnn_implementations.AttentionGGNN(
        NODE_FEATURES, EDGE_FEATURES, MESSAGE_SIZE, MESSAGE_PASSES, OUT_FEATURES,
        att_hidden_dim=9,
        msg_hidden_dim=4, gather_width=7, gather_att_hidden_dim=9, gather_emb_hidden_dim=7, out_hidden_dim=3
    )

class EMNImplementationTestCase(GNNTestCase):
    EDGE_EMBEDDING_SIZE = 7
    net = gnn.emn_implementations.EMNImplementation(
        edge_features=EDGE_FEATURES, edge_embedding_size=EDGE_EMBEDDING_SIZE, message_passes=MESSAGE_PASSES,
        out_features=OUT_FEATURES, node_features=NODE_FEATURES,
        msg_hidden_dim=4, gather_width=7, gather_att_hidden_dim=9, gather_emb_hidden_dim=7, out_hidden_dim=3
    )

import example
from tests.gnn_test_case import GNNTestCase


NODE_FEATURES = GNNTestCase.NODE_FEATURES
EDGE_FEATURES = GNNTestCase.EDGE_FEATURES
MESSAGE_SIZE = GNNTestCase.MESSAGE_SIZE
MESSAGE_PASSES = GNNTestCase.MESSAGE_PASSES
OUT_FEATURES = GNNTestCase.OUT_FEATURES


class ExampleMPNNTestCase(GNNTestCase):
    net = example.ExampleAttentionMPNN(
        NODE_FEATURES, EDGE_FEATURES, OUT_FEATURES
    )
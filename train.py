import torch
from torch import optim
from torch.utils.data import DataLoader

import gnn.summation_mpnn_implementations
import gnn.aggregation_mpnn_implementations
import gnn.emn_implementations
from losses import LOSS_FUNCTIONS
from train_logging import LOG_FUNCTIONS
from gnn.molgraph_data import MolGraphDataset, molgraph_collate_fn

import argparse


MODEL_CONSTRUCTOR_DICTS = {
    'ENNS2V': {
        'constructor': gnn.summation_mpnn_implementations.ENNS2V,
        'hyperparameters': {
            'message-passes': {'type': int, 'default': 5},
            'message-size': {'type': int, 'default': 50},
            'enn-depth': {'type': int, 'default': 3},
            'enn-hidden-dim': {'type': int, 'default': 100},
            'enn-dropout-p': {'type': float, 'default': 0.0},
            's2v-lstm-computations': {'type': int, 'default': 7},
            's2v-memory-size': {'type': int, 'default': 50},
            'out-depth': {'type': int, 'default': 2},
            'out-hidden-dim': {'type': int, 'default': 300},
            'out-dropout-p': {'type': float, 'default': 0.0}
        }
    },
    'GGNN': {
        'constructor': gnn.summation_mpnn_implementations.GGNN,
        'hyperparameters': {  # the below, batch size 50, learn rate 1.176e-5 and 1200 epochs is good for ESOL
            'message-passes': {'type': int, 'default': 1},
            'message-size': {'type': int, 'default': 25},
            'msg-depth': {'type': int, 'default': 2},
            'msg-hidden-dim': {'type': int, 'default': 50},
            'msg-dropout-p': {'type': float, 'default': 0.0},
            'gather-width': {'type': int, 'default': 45},
            'gather-att-depth': {'type': int, 'default': 2},
            'gather-att-hidden-dim': {'type': int, 'default': 26},
            'gather-att-dropout-p': {'type': float, 'default': 0.0},
            'gather-emb-depth': {'type': int, 'default': 2},
            'gather-emb-hidden-dim': {'type': int, 'default': 26},
            'gather-emb-dropout-p': {'type': float, 'default': 0.0},
            'out-depth': {'type': int, 'default': 2},
            'out-hidden-dim': {'type': int, 'default': 450},
            'out-dropout-p': {'type': float, 'default': 0.00463},
            'out-layer-shrinkage': {'type': float, 'default': 0.5028}
        }
    },
    'AttentionGGNN': {  # the below, batch size 50, learn rate 1.560e-5 and 600 epochs is good for BBBP
        'constructor': gnn.aggregation_mpnn_implementations.AttentionGGNN,
        'hyperparameters': {
            'message-passes': {'type': int, 'default': 8},
            'message-size': {'type': int, 'default': 25},
            'msg-depth': {'type': int, 'default': 2},
            'msg-hidden-dim': {'type': int, 'default': 50},
            'msg-dropout-p': {'type': float, 'default': 0.0},
            'att-depth': {'type': int, 'default': 2},
            'att-hidden-dim': {'type': int, 'default': 50},
            'att-dropout-p': {'type': float, 'default': 0.0},
            'gather-width': {'type': int, 'default': 45},
            'gather-att-depth': {'type': int, 'default': 2},
            'gather-att-hidden-dim': {'type': int, 'default': 45},
            'gather-att-dropout-p': {'type': float, 'default': 0.0},
            'gather-emb-depth': {'type': int, 'default': 2},
            'gather-emb-hidden-dim': {'type': int, 'default': 26},
            'gather-emb-dropout-p': {'type': float, 'default': 0.0},
            'out-depth': {'type': int, 'default': 2},
            'out-hidden-dim': {'type': int, 'default': 560},
            'out-dropout-p': {'type': float, 'default': 0.1},
            'out-layer-shrinkage': {'type': float, 'default': 0.6}
        }
    },
    'EMN': {  # the below, batch size 50, learn rate 1e-4 and 1000 epochs is good for SIDER
        'constructor': gnn.emn_implementations.EMNImplementation,
        'hyperparameters': {
            'message-passes': {'type': int, 'default': 8},
            'edge-embedding-size': {'type': int, 'default': 50},
            'edge-emb-depth': {'type': int, 'default': 2},
            'edge-emb-hidden-dim': {'type': int, 'default': 105},
            'edge-emb-dropout-p': {'type': float, 'default': 0.0},
            'att-depth': {'type': int, 'default': 2},
            'att-hidden-dim': {'type': int, 'default': 85},
            'att-dropout-p': {'type': float, 'default': 0.0},
            'msg-depth': {'type': int, 'default': 2},
            'msg-hidden-dim': {'type': int, 'default': 150},
            'msg-dropout-p': {'type': float, 'default': 0.0},
            'gather-width': {'type': int, 'default': 45},
            'gather-att-depth': {'type': int, 'default': 2},
            'gather-att-hidden-dim': {'type': int, 'default': 45},
            'gather-att-dropout-p': {'type': float, 'default': 0.0},
            'gather-emb-depth': {'type': int, 'default': 2},
            'gather-emb-hidden-dim': {'type': int, 'default': 45},
            'gather-emb-dropout-p': {'type': float, 'default': 0.0},
            'out-depth': {'type': int, 'default': 2},
            'out-hidden-dim': {'type': int, 'default': 450},
            'out-dropout-p': {'type': float, 'default': 0.1},
            'out-layer-shrinkage': {'type': float, 'default': 0.6}
        }
    }
}


common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

common_args_parser.add_argument('--cuda', action='store_true', default=False, help='Enables CUDA training')

common_args_parser.add_argument('--train-set', type=str, default='toydata/piece-of-tox21-train.csv.gz', help='Training dataset path')
common_args_parser.add_argument('--valid-set', type=str, default='toydata/piece-of-tox21-valid.csv.gz', help='Validation dataset path')
common_args_parser.add_argument('--test-set', type=str, default='toydata/piece-of-tox21-test.csv.gz', help='Testing dataset path')
common_args_parser.add_argument('--loss', type=str, default='MaskedMultiTaskCrossEntropy', choices=[k for k, v in LOSS_FUNCTIONS.items()])
common_args_parser.add_argument('--score', type=str, default='roc-auc', help='roc-auc or MSE')

common_args_parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
common_args_parser.add_argument('--batch-size', type=int, default=50, help='Number of graphs in a mini-batch')
common_args_parser.add_argument('--learn-rate', type=float, default=1e-5)

common_args_parser.add_argument('--savemodel', action='store_true', default=False, help='Saves model with highest validation score')
common_args_parser.add_argument('--logging', type=str, default='less', choices=[k for k, v in LOG_FUNCTIONS.items()])


main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = main_parser.add_subparsers(help=', '.join([k for k, v in MODEL_CONSTRUCTOR_DICTS.items()]), dest='model')
subparsers.required = True

model_parsers = {}
for model_name, constructor_dict in MODEL_CONSTRUCTOR_DICTS.items():
    subparser = subparsers.add_parser(model_name, parents=[common_args_parser])
    for hp_name, hp_kwargs in constructor_dict['hyperparameters'].items():
        subparser.add_argument('--' + hp_name, **hp_kwargs, help=model_name + ' hyperparameter')
    model_parsers[model_name] = subparser


def main():
    global args
    args = main_parser.parse_args()
    args_dict = vars(args)
    # dictionary of hyperparameters that are specific to the chosen model
    model_hp_kwargs = {
        name.replace('-', '_'): args_dict[name.replace('-', '_')]   # argparse converts to "_" implicitly
        for name, v in MODEL_CONSTRUCTOR_DICTS[args.model]['hyperparameters'].items()
    }

    train_dataset = MolGraphDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=molgraph_collate_fn)
    validation_dataset = MolGraphDataset(args.valid_set)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, collate_fn=molgraph_collate_fn)
    test_dataset = MolGraphDataset(args.test_set)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=molgraph_collate_fn)

    ((sample_adjacency, sample_nodes, sample_edges), sample_target) = train_dataset[0]
    net = MODEL_CONSTRUCTOR_DICTS[args.model]['constructor'](
        node_features=len(sample_nodes[0]), edge_features=len(sample_edges[0, 0]), out_features=len(sample_target),
        **model_hp_kwargs
    )
    if args.cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.learn_rate)
    criterion = LOSS_FUNCTIONS[args.loss]

    for epoch in range(args.epochs):
        net.train()
        for i_batch, batch in enumerate(train_dataloader):

            if args.cuda:
                batch = [tensor.cuda() for tensor in batch]
            adjacency, nodes, edges, target = batch

            optimizer.zero_grad()
            output = net(adjacency, nodes, edges)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 5.0)
            optimizer.step()

        with torch.no_grad():
            net.eval()
            LOG_FUNCTIONS[args.logging](
                net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args
            )


if __name__ == '__main__':
    main()

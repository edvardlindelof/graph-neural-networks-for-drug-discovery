import torch
from torch.utils.data import DataLoader

from gnn.molgraph_data import MolGraphDataset, molgraph_collate_fn

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--cuda', action='store_true', default=False, help='Enables CUDA training')

parser.add_argument('--modelpath', type=str, help='Path to saved model', required=True)
parser.add_argument('--datapath', type=str, default='toydata/piece-of-tox21-test.csv.gz', help='Testing dataset path')
parser.add_argument('--score', type=str, choices=['roc-auc', 'pr-auc', 'MSE', 'RMSE'], required=True)


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    with torch.no_grad():
        net = torch.load(args.modelpath)
        if args.cuda:
            net = net.cuda()
        else:
            net = net.cpu()
        net.eval()

        dataset = MolGraphDataset(args.datapath, prediction=True)
        dataloader = DataLoader(dataset, batch_size=50, collate_fn=molgraph_collate_fn)

        batch_outputs = []
        for i_batch, batch in enumerate(dataloader):
            if args.cuda:
                batch = [tensor.cuda() for tensor in batch]
            adjacency, nodes, edges, target = batch
            batch_output = net(adjacency, nodes, edges)
            if args.score == 'roc-auc' or args.score == 'pr-auc':
                batch_output = torch.sigmoid(batch_output)
            batch_outputs.append(batch_output)

        output = torch.cat(batch_outputs).cpu().numpy()

        print('\t'.join([str(col) for col in dataset.header_cols]))
        for i in range(len(output)):
            comment = dataset.comments[i]
            row_str = '\t'.join([str(x) for x in output[i]])
            print('{}, {}'.format(comment, row_str))

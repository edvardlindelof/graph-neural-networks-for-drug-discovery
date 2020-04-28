import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from example import ExampleAttentionMPNN  # necessary for torch.load(...)
from gnn.molgraph_data import MolGraphDataset, molgraph_collate_fn


if __name__ == '__main__':
    with torch.no_grad():
        net = torch.load('trainedembedder')
        net.eval()

        # set prediction=True if targets are not available
        # dataloader will then return 0s
        dataset = MolGraphDataset('toydata/piece-of-esol.csv.gz', prediction=False)
        dataloader = DataLoader(dataset, batch_size=50, collate_fn=molgraph_collate_fn)
        # feeding one batch at a time then concatenating
        # because feeding a whole dataset will use a crazy amount of memory
        embeddings = torch.cat([net(a, n, e) for a, n, e, _ in dataloader]).numpy()
        targets = torch.cat([t for _, _, _, t in dataloader]).numpy()

    linreg = LinearRegression().fit(embeddings, targets)
    preds = linreg.predict(embeddings)
    mse = mean_squared_error(targets, preds)
    print(f'MSE of linear regression fitted to embeddings: {mse}')
    print(f'linear regression weights: {linreg.coef_}')

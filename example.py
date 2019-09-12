import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from gnn.aggregation_mpnn import AggregationMPNN
from gnn.molgraph_data import MolGraphDataset, molgraph_collate_fn


class ExampleAttentionMPNN(AggregationMPNN):

    def __init__(self, node_features, edge_features, out_features, message_passes=3):
        super(ExampleAttentionMPNN, self).__init__(node_features, edge_features, node_features, message_passes, out_features)

        self.message_att_weight = nn.Linear(node_features, 1)
        self.message_emb_weight = nn.Linear(node_features, node_features)
        self.out_weight = nn.Linear(node_features, out_features)

    def aggregate_message(self, nodes, node_neighbours, edges, mask):
        neighbourhood = torch.cat([nodes.unsqueeze(1), node_neighbours], dim=1)

        neighbourhood_mask = torch.cat([torch.ones((mask.shape[0], 1)), mask], dim=1)
        energy_mask = (neighbourhood_mask == 0).float() * 1e6

        energies = self.message_att_weight(neighbourhood) - energy_mask.unsqueeze(-1)
        attention = torch.softmax(energies, dim=1)
        embedding = self.message_emb_weight(neighbourhood)
        messages = torch.sum(attention * embedding, dim=1)
        return messages

    def update(self, nodes, messages):
        hidden_nodes = torch.selu(messages)
        return hidden_nodes

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embedding = torch.sum(hidden_nodes, dim=1)
        output = self.out_weight(graph_embedding)
        return output


if __name__ == '__main__':
    print('loading data')
    train_dataset = MolGraphDataset('toydata/piece-of-esol.csv.gz')
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True, collate_fn=molgraph_collate_fn)

    print('instantiating ExampleAttentionMPNN')
    # 75 and 4 corresponds to MolGraphDataset, 1 corresponds to ESOL
    net = ExampleAttentionMPNN(node_features=75, edge_features=4, out_features=1)
    optimizer = optim.Adam(net.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    print('starting training')
    for epoch in range(10):
        for i_batch, batch in enumerate(train_dataloader):
            adjacency, nodes, edges, target = batch
            optimizer.zero_grad()
            output = net(adjacency, nodes, edges)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 5.0)
            optimizer.step()

        print('epoch: {}, training MSE: {}'.format(epoch + 1, loss))

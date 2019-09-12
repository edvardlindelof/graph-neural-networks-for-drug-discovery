import gzip
import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch.utils import data

from gnn.graph_features import atom_features
from collections import defaultdict


class MolGraphDataset(data.Dataset):
    r"""For datasets consisting of SMILES strings and target values.

    Expects a csv file formatted as:
    comment,smiles,targetName1,targetName2
    Some Comment,CN=C=O,0,1
    ,CC(=O)NCCC1=CNc2c1cc(OC)cc2,1,1

    Args:
        path
        inference: set to True if dataset contains no target values
    """

    def __init__(self, path, inference=False):
        with gzip.open(path, 'r') as file:
            self.header_cols = file.readline().decode('utf-8')[:-2].split('\t')
        n_cols = len(self.header_cols)

        self.target_names = self.header_cols[2:]
        self.comments = np.genfromtxt(path, delimiter='\t', skip_header=1, usecols=[0], dtype=np.str, comments=None)
        # comments=None because default is "#", that some smiles contain
        self.smiles = np.genfromtxt(path, delimiter='\t', skip_header=1, usecols=[1], dtype=np.str, comments=None)
        if inference:
            self.targets = np.empty((len(self.smiles), n_cols - 2))  # may be used to figure out number of targets etc
        else:
            self.targets = np.genfromtxt(path, delimiter='\t', skip_header=1, usecols=range(2, n_cols), comments=None).reshape(-1, n_cols - 2)

    def __getitem__(self, index):
        adjacency, nodes, edges = smile_to_graph(self.smiles[index])
        targets = self.targets[index, :]
        return (adjacency, nodes, edges), targets

    def __len__(self):
        return len(self.smiles)

rdLogger = rdkit.RDLogger.logger()
rdLogger.setLevel(rdkit.RDLogger.ERROR)

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return adjacency, node_features, edge_features

# rdkit GetBondType() result -> int
BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


class MolGraphDatasetSubset(MolGraphDataset):
    r"""Takes a subset of MolGraphDataset.

    The "Subset" class of pytorch does not allow column selection
    """

    def __init__(self, path, indices=None, columns=None):
        super(MolGraphDatasetSubset, self).__init__(path)
        if indices:
            self.smiles = self.smiles[indices]
            self.targets = self.targets[indices]
        if columns:
            self.target_names = [self.target_names[col] for col in columns]
            self.targets = self.targets[:, columns]


# data is list of ((g,h,e), [targets])
# to be passable to DataLoader it needs to have this signature,
# where the outer tuple is that which is returned by Dataset's __getitem__
def molgraph_collate_fn(data):
    n_samples = len(data)
    (adjacency_0, node_features_0, edge_features_0), targets_0 = data[0]
    n_nodes_largest_graph = max(map(lambda sample: sample[0][0].shape[0], data))
    n_node_features = node_features_0.shape[1]
    n_edge_features = edge_features_0.shape[2]
    n_targets = len(targets_0)

    adjacency_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph)
    node_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_node_features)
    edge_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph, n_edge_features)
    target_tensor = torch.zeros(n_samples, n_targets)

    for i in range(n_samples):
        (adjacency, node_features, edge_features), target = data[i]
        n_nodes = adjacency.shape[0]

        adjacency_tensor[i, :n_nodes, :n_nodes] = torch.Tensor(adjacency)
        node_tensor[i, :n_nodes, :] = torch.Tensor(node_features)
        edge_tensor[i, :n_nodes, :n_nodes, :] = torch.Tensor(edge_features)

        target_tensor[i] = torch.Tensor(target)

    return adjacency_tensor, node_tensor, edge_tensor, target_tensor

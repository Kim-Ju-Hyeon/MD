import os
from glob import glob

import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter

from utils.train_helper import mkdir


def make_mol_file_to_dataset(smile_csv, data, mol_state, test=False):
    types = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Si': 6, 'P': 7, 'S': 8, 'Cl': 9, 'Br': 10, 'I': 11}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    dataset = []

    for m_dir in data:
        temp = m_dir.split('/')[-1].split('_')
        index = int(temp[1])

        _state = temp[-1].split('.')[0]

        if _state == mol_state:
            if not test:
                if _state == 'g':
                    target = smile_csv.iloc[index].Reorg_g
                    state = 'g'
                elif _state == 'ex':
                    target = smile_csv.iloc[index].Reorg_ex
                    state = 'ex'
            else:
                target = 0
                if _state == 'g':
                    state = 'g'
                elif _state == 'ex':
                    state = 'ex'

            m = Chem.MolFromMolFile(m_dir)

            N = m.GetNumAtoms()

            conf = m.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in m.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []

            for bond in m.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=target, idx=index, state=state)

            dataset.append(data)

    return dataset


def make_qm9_hackerthon_to_dataset(data_dir):
    path = os.path.join(data_dir, 'geometric_dataset')
    train_path = os.path.join(path, 'train_dataset.pickle')
    if os.path.isfile(train_path):
        train = pickle.load(open(train_path, 'rb'))
    else:
        mkdir(path)
        train_data = torch.load(data_dir+'/qm9_train_data.pt')

        y = train_data['mu']
        num_nodes = train_data['num_atoms']
        num_edges = train_data['num_bonds']
        coords = train_data['x']
        atomic_numbers = train_data['atomic_numbers']
        edge = train_data['edge']

        train = []

        for i in range(len(y)):
            y_s = torch.tensor(y[i], dtype=torch.float)
            num_node_s = num_nodes[i]
            num_edge_s = num_edges[i]
            coord = torch.tensor(coords[i][:num_node_s])
            atomic_num = torch.tensor(atomic_numbers[i][:num_node_s, :], dtype=torch.long).squeeze()
            edge_index = torch.tensor(edge[0][:num_node_s, :2], dtype=torch.long).t()

            train.append(Data(pos=coord, z=atomic_num, y=y_s, edge_index=edge_index))

        pickle.dump(train, open(train_path, 'wb'))

    test_path = os.path.join(path, 'test_dataset.pickle')
    if os.path.isfile(test_path):
        test = pickle.load(open(test_path, 'rb'))
    else:
        mkdir(path)
        test_data = torch.load(data_dir+"/qm9_test_data.pt")

        num_nodes = test_data['num_atoms']
        num_edges = test_data['num_bonds']
        coords = test_data['x']
        atomic_numbers = test_data['atomic_numbers']

        test = []
        for i in range(len(num_nodes)):
            num_node_s = num_nodes[i]
            num_edge_s = num_edges[i]
            coord = torch.tensor(coords[i][:num_node_s])
            edge_index = torch.tensor(edge[0][:num_node_s, :2], dtype=torch.long).t()
            atomic_num = torch.tensor(atomic_numbers[i][:num_node_s, :], dtype=torch.long).squeeze()

            test.append(Data(pos=coord, z=atomic_num, edge_index=edge_index))

        pickle.dump(test, open(test_path, 'wb'))

    return train, test


def get_qm9_dataset(data_dir):
    train_dataset, test_dataset = make_qm9_hackerthon_to_dataset(data_dir)

    seed = np.random.randint(10000)
    random_state = np.random.RandomState(seed=seed)
    perm = torch.from_numpy(random_state.permutation(np.arange(len(train_dataset))))

    idx = int(len(train_dataset) * 0.99)
    train_idx = perm[:idx]
    val_idx = perm[idx:]

    train = []
    for ii in train_idx:
        train.append(train_dataset[ii])

    validation = []
    for jj in val_idx:
        validation.append(train_dataset[jj])

    return train, validation, test_dataset
        
        
def get_dataset(data_dir, mol_state):
    train_data_dirs = data_dir + '/mol_files/train_set'
    train_data = glob(train_data_dirs + '/*.mol')

    test_data_dirs = data_dir + '/mol_files/test_set'
    smile_csv = pd.read_csv(data_dir + '/train_set.ReorgE.csv', index_col=0)

    test_data = glob(test_data_dirs + '/*.mol')

    train_dataset = make_mol_file_to_dataset(smile_csv, train_data, mol_state, test=False)
    test_dataset = make_mol_file_to_dataset(smile_csv, test_data, mol_state, test=True)

    seed = np.random.randint(10000)
    random_state = np.random.RandomState(seed=seed)
    perm = torch.from_numpy(random_state.permutation(np.arange(len(train_dataset))))

    idx = int(len(train_dataset) * 0.8)
    train_idx = perm[:idx]
    val_idx = perm[idx:]

    train = []
    for ii in train_idx:
        train.append(train_dataset[ii])

    validation = []
    for jj in val_idx:
        validation.append(train_dataset[jj])

    return train, validation, test_dataset

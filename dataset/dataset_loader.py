import os
from glob import glob

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


def make_mol_file_to_dataset(smile_csv, data, test=False):
    types = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Si': 6, 'P': 7, 'S': 8, 'Cl': 9, 'Br': 10, 'I': 11}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    dataset = []

    for m_dir in data:
        temp = m_dir.split('/')[-1].split('_')
        index = int(temp[1])

        if not test:
            if 'g' in temp[-1]:
                target = smile_csv.iloc[index].Reorg_g
            else:
                target = smile_csv.iloc[index].Reorg_ex
        else:
            target = 0

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
                    edge_attr=edge_attr, y=target, idx=index)

        dataset.append(data)


def get_dataset(data_dir):
    train_data_dirs = data_dir + '/mol_files/train_set'
    test_data_dirs = data_dir + '/mol_files/test_set'
    smile_csv = pd.read_csv(data_dir+'/train_set.ReorgE.csv', index_col=0)

    train_data = glob(train_data_dirs + '/*.mol')
    test_data = glob(test_data_dirs+'/*.mol')

    train_dataset = make_mol_file_to_dataset(smile_csv, train_data, test=False)
    test_dataset = make_mol_file_to_dataset(smile_csv, test_data, test=True)

    seed = np.random.randint(10000)
    random_state = np.random.RandomState(seed=seed)
    perm = torch.from_numpy(random_state.permutation(np.arange(len(train_data))))

    idx = int(len(train_data) * 0.8)
    train_idx = perm[:idx]
    val_idx = perm[idx:]

    train = []
    for ii in train_idx:
        train.append(train_dataset[ii])

    validation = []
    for jj in val_idx:
        validation.append(train_dataset[jj])

    return train, validation, test_dataset

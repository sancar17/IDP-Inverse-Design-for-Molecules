import torch
from torch.utils.data import Dataset, DataLoader
from .mol_tree import MolTree
import numpy as np
from .jtnn_enc import JTNNEncoder
from .mpn import MPN
from .jtmpn import JTMPN
import pickle as pickle
import os, random
import pandas as pd

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])#, num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])#, num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolderJoint(object):

    def __init__(self, data_folder, vocab, batch_size, csv_file, num_workers=12, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.csv = pd.read_csv(csv_file)
        #(self.csv)

        '''print(self.data_folder)
        print(self.data_files)
        print(self.batch_size)
        print(self.vocab)
        print(self.csv)'''

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)
                filtered_data = []
                homo_values = []
                lumo_values = []
                r2_values = []
                total_trees = len(data)
                filtered_trees = 0

                for i in range(total_trees):
                    tree_size = len(data[i].nodes)
                    
                    if tree_size <= 100000:  # Filter condition: keep trees with 10 or fewer nodes
                        filtered_data.append(data[i])
                        smiles = data[i].smiles
                        row = self.csv.loc[self.csv['smiles'] == smiles]
                        if not row.empty:
                            homo = row.iloc[0]['normalized_homo']
                            lumo = row.iloc[0]['normalized_lumo']
                            r2 = row.iloc[0]['normalized_r2']

                            homo_values.append(homo)
                            lumo_values.append(lumo)
                            r2_values.append(r2)
                        else:
                            print(f"SMILES {smiles} not found in CSV")
                    else:
                        filtered_trees += 1

            #print(f"File: {fn}")
            '''print(f"Total trees: {total_trees}")
            print(f"Trees filtered out (>10 nodes): {filtered_trees}")
            print(f"Remaining trees: {len(filtered_data)}")'''

            if self.shuffle:
                combined = list(zip(filtered_data, homo_values, lumo_values, r2_values))
                random.shuffle(combined)
                filtered_data, homo_values, lumo_values, r2_values = map(list, zip(*combined))

            batches = [filtered_data[i : i + self.batch_size] for i in range(0, len(filtered_data), self.batch_size)]
            homos = [homo_values[i : i + self.batch_size] for i in range(0, len(filtered_data), self.batch_size)]
            lumos = [lumo_values[i : i + self.batch_size] for i in range(0, len(filtered_data), self.batch_size)]
            r2s = [r2_values[i : i + self.batch_size] for i in range(0, len(filtered_data), self.batch_size)]

            dataset = MolTreeDatasetJoint(batches, homos, lumos, r2s, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del filtered_data, batches, homos, lumos, r2s, dataset, dataloader

class MolTreeFolderJoint_1p(object):

    def __init__(self, data_folder, vocab, batch_size, csv_file, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.csv = pd.read_csv(csv_file)
        #(self.csv)

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)
                r2_values = []
                for i in range(len(data)):
                    smiles = data[i].smiles
                    row = self.csv.loc[self.csv['smiles'] == smiles]
                    if not row.empty:
                        r2 = row.iloc[0]['normalized_r2']
                        r2_values.append(r2)
                        #print(smiles, homo, lumo, r2)
                    else:
                        print(f"SMILES {smiles} not found in CSV")

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            r2s = [r2_values[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            
            if len(batches[-1]) < self.batch_size:
                batches.pop()
                r2s.pop()

            dataset = MolTreeDatasetJoint_1p(batches, r2s, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])#, num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, r2s, dataset, dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = list(zip(*self.data[idx]))
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)
    
class MolTreeDatasetJoint(Dataset):

    def __init__(self, data, homos, lumos, r2s, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm
        self.homos = [torch.tensor(batch, dtype=torch.float, device="cuda:0") for batch in homos]
        self.lumos = [torch.tensor(batch, dtype=torch.float, device="cuda:0") for batch in lumos]
        self.r2s = [torch.tensor(batch, dtype=torch.float, device="cuda:0") for batch in r2s]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorizeJoint(self.data[idx], self.homos[idx], self.lumos[idx], self.r2s[idx], self.vocab, assm=self.assm)
    
class MolTreeDatasetJoint_1p(Dataset):

    def __init__(self, data, r2s, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm
        self.r2s = r2s

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorizeJoint_1p(self.data[idx], self.r2s[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def tensorizeJoint(tree_batch, homos, lumos, r2s, vocab, assm=True):
    if not tree_batch:
        return None  # Return None if tree_batch is empty

    try:
        set_batch_nodeID(tree_batch, vocab)
        smiles_batch = [tree.smiles for tree in tree_batch]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
        mpn_holder = MPN.tensorize(smiles_batch)

        if not assm:
            return tree_batch, jtenc_holder, mpn_holder

        cands = []
        batch_idx = []
        for i, mol_tree in enumerate(tree_batch):
            for node in mol_tree.nodes:
                if node.is_leaf or len(node.cands) == 1:
                    continue
                cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
                batch_idx.extend([i] * len(node.cands))

        if not cands:
            return None  # Return None if there are no candidates

        jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
        if jtmpn_holder is None:
            return None  # Return None if JTMPN tensorization failed

        batch_idx = torch.LongTensor(batch_idx)

        return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx), homos, lumos, r2s

    except Exception as e:
        print(f"Error in tensorizeJoint: {e}")
        return None

def tensorizeJoint_1p(tree_batch, r2s, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx), r2s

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1

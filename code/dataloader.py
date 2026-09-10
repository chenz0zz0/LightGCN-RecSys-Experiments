# -*- coding: utf-8 -*-
"""Dataset loaders with deterministic validation splitting and sparse graph construction."""

import os
from os.path import join
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    """Abstract Base Class for recommendation datasets."""
    def __init__(self):
        pass
    @property
    def n_users(self): raise NotImplementedError
    @property
    def m_items(self): raise NotImplementedError
    @property
    def trainDataSize(self): raise NotImplementedError
    @property
    def testDict(self): raise NotImplementedError
    @property
    def allPos(self): raise NotImplementedError
    def getSparseGraph(self): raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset loader specific to the LastFM dataset.
    Includes social network graph information alongside user-item interactions.
    """
    def __init__(self, path="../data/lastfm"):
        cprint("loading [last fm]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        
        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData

        # Build the same deterministic per-user validation split used by Loader.
        # With val_ratio == 0, all original training interactions remain in train
        # and valUser/valItem are empty.
        original_train_user = np.asarray(trainData.iloc[:, 0], dtype=np.int64)
        original_train_item = np.asarray(trainData.iloc[:, 1], dtype=np.int64)

        val_ratio = float(world.config.get('val_ratio', 0.0))
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError('val_ratio must be in [0, 1).')

        train_user, train_item = [], []
        val_user, val_item = [], []

        for uid in np.unique(original_train_user):
            items = original_train_item[original_train_user == uid]

            if val_ratio > 0.0 and len(items) >= 2:
                rng = np.random.default_rng(world.seed + int(uid))
                n_val = min(
                    len(items) - 1,
                    max(1, int(round(len(items) * val_ratio)))
                )
                val_idx = set(
                    rng.choice(len(items), size=n_val, replace=False).tolist()
                )

                for idx, item in enumerate(items):
                    if idx in val_idx:
                        val_user.append(int(uid))
                        val_item.append(int(item))
                    else:
                        train_user.append(int(uid))
                        train_item.append(int(item))
            else:
                train_user.extend([int(uid)] * len(items))
                train_item.extend(int(item) for item in items)

        self.trainUser = np.asarray(train_user, dtype=np.int64)
        self.trainItem = np.asarray(train_item, dtype=np.int64)
        self.valUser = np.asarray(val_user, dtype=np.int64)
        self.valItem = np.asarray(val_item, dtype=np.int64)
        self.trainUniqueUsers = np.unique(self.trainUser)

        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users, users)
        self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1])), shape=(self.n_users, self.n_users))
        # (users, items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_users, self.m_items)) 
        
        # Pre-calculate positive and negative interactions
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_eval_dict(self.testUser, self.testItem)
        self.__valDict = self.__build_eval_dict(self.valUser, self.valItem)

    @property
    def n_users(self): return 1892
    @property
    def m_items(self): return 4489
    @property
    def trainDataSize(self): return len(self.trainUser)
    @property
    def testDict(self): return self.__testDict
    @property
    def valDict(self): return self.__valDict
    @property
    def allPos(self): return self._allPos

    def getSparseGraph(self):
        """Builds and caches the normalized adjacency matrix."""
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            
            # Construct sparse adjacency with the current PyTorch COO tensor API.
            self.Graph = torch.sparse_coo_tensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]), dtype=torch.int32)
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            
            # Construct sparse adjacency with the current PyTorch COO tensor API.
            self.Graph = torch.sparse_coo_tensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]), dtype=torch.float32)
            
            # MPS does not support the sparse propagation used here, so the sparse graph stays on CPU.
            self.Graph = self.Graph.coalesce().cpu()
        return self.Graph

    def __build_eval_dict(self, users, items):
        eval_data = {}
        for user, item in zip(users, items):
            uid = int(user)
            eval_data.setdefault(uid, []).append(int(item))
        return eval_data

    def getUserPosItems(self, users):
        return [self.UserItemNet[user].nonzero()[1] for user in users]
    
    def getUserNegItems(self, users):
        return [self.allNeg[user] for user in users]


class Loader(BasicDataset):
    """
    Standard dataset loader optimized for implicit feedback benchmarks.
    Supported formats: Gowalla, Yelp2018, Amazon-Book.
    """
    def __init__(self, config=world.config, path="../data/gowalla"):
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.n_user, self.m_item = 0, 0
        train_file, test_file = path + '/train.txt', path + '/test.txt'
        self.path = path
        
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        valItem, valUser = [], []
        val_ratio = float(config.get('val_ratio', 0.0))
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError('val_ratio must be in [0, 1).')
        self.traindataSize, self.testDataSize = 0, 0

        # Load Training Data
        with open(train_file) as f:
            for l in f.readlines():
                # Split on arbitrary whitespace so repeated or trailing spaces do not create empty tokens.
                l = l.strip().split() 
                if len(l) > 0:
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    self.n_user = max(self.n_user, uid)
                    
                    items = [int(i) for i in l[1:]]
                    if len(items) > 0:
                        self.m_item = max(self.m_item, max(items))
                        if val_ratio > 0.0 and len(items) >= 2:
                            rng = np.random.default_rng(world.seed + uid)
                            n_val = min(len(items) - 1, max(1, int(round(len(items) * val_ratio))))
                            val_idx = set(rng.choice(len(items), size=n_val, replace=False).tolist())
                            train_items = [item for idx, item in enumerate(items) if idx not in val_idx]
                            val_items = [item for idx, item in enumerate(items) if idx in val_idx]
                            valUser.extend([uid] * len(val_items))
                            valItem.extend(val_items)
                        else:
                            train_items = items
                        trainUser.extend([uid] * len(train_items))
                        trainItem.extend(train_items)
                        self.traindataSize += len(train_items)
        
        # Load Testing Data
        with open(test_file) as f:
            for l in f.readlines():
                # Apply the same whitespace-tolerant parsing to the test split.
                l = l.strip().split() 
                if len(l) > 0:
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    self.n_user = max(self.n_user, uid)
                    
                    items = [int(i) for i in l[1:]]
                    if len(items) > 0:
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.testDataSize += len(items)
        
        self.m_item += 1
        self.n_user += 1
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser, self.trainItem = np.array(trainUser), np.array(trainItem)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.valUser, self.valItem = np.array(valUser), np.array(valItem)
        self.testUser, self.testItem = np.array(testUser), np.array(testItem)
        
        self.Graph = None
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_eval_dict(self.testUser, self.testItem)
        self.__valDict = self.__build_eval_dict(self.valUser, self.valItem)
        print(f"{world.dataset} is ready to go.")

    @property
    def n_users(self): return self.n_user
    @property
    def m_items(self): return self.m_item
    @property
    def trainDataSize(self): return self.traindataSize
    @property
    def testDict(self): return self.__testDict
    @property
    def valDict(self): return self.__valDict
    @property
    def allPos(self): return self._allPos

    def _convert_sp_mat_to_sp_tensor(self, X):
        """Converts scipy sparse matrix to PyTorch sparse tensor."""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape)).coalesce()
        
    def getSparseGraph(self):
        """Constructs and caches the normalized adjacency matrix."""
        if self.Graph is None:
            try:
                cache_name = 's_pre_adj_mat.npz' if world.config.get('val_ratio', 0.0) == 0 else f"s_pre_adj_mat_val{world.config['val_ratio']:.4f}_seed{world.seed}.npz"
                pre_adj_mat = sp.load_npz(join(self.path, cache_name))
                norm_adj = pre_adj_mat
            except:
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
                sp.save_npz(join(self.path, cache_name), norm_adj)

            # Keep the coalesced sparse adjacency on CPU for the MPS compatibility path.
            graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce()
            self.Graph = graph.cpu() if world.device.type == 'mps' else graph.to(world.device)
        return self.Graph

    def __build_eval_dict(self, users, items):
        """Build a user -> ground-truth items dictionary for validation/test evaluation."""
        eval_data = {}
        for user, item in zip(users, items):
            uid = int(user)
            eval_data.setdefault(uid, []).append(int(item))
        return eval_data

    def __build_test(self):
        """Constructs a dictionary mapping users to their ground-truth test items."""
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            # Normalize user IDs to hashable Python integers when building evaluation dictionaries.
            clean_uid = int(np.array(user).flatten()[0]) if isinstance(user, (np.ndarray, list)) else int(user)
            
            if test_data.get(clean_uid):
                test_data[clean_uid].append(item)
            else:
                test_data[clean_uid] = [item]
        return test_data

    def getUserPosItems(self, users):
        return [self.UserItemNet[user].nonzero()[1] for user in users]
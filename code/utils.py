# -*- coding: utf-8 -*-
"""
Utility Functions for Recommendation System.
Original code inspired by: https://github.com/gusye1234/LightGCN-PyTorch
Modified by: chenz0zz0

Key Optimizations:
1. Native Python Sampling Fallback: Overrode C++ sampling dependencies to avoid 
   compilation issues on non-Linux environments (Mac/Windows).
2. Robust Evaluation Metrics: Added `np.atleast_1d` to flatten nested arrays 
   during label generation, fixing 'TypeError: length-1 arrays' bugs.
"""

import world
import torch
from torch import nn, optim
import numpy as np
from dataloader import BasicDataset
from time import time
import os

sample_ext = False

def device_sync():
    if world.device.type == 'cuda':
        torch.cuda.synchronize()
    elif world.device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()

def UniformSample_original(dataset, neg_ratio=1):
    return UniformSample_original_python(dataset)

def UniformSample_original_python(dataset):
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        positem = posForUser[np.random.randint(0, len(posForUser))]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem not in posForUser:
                break
        S.append([user, positem, negitem])
    return np.array(S)

class BPRLoss:
    def __init__(self, recmodel, config: dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.grad_clip = float(config.get('grad_clip', 0.0))
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg, profile=False):
        if profile:
            device_sync(); t0 = time()
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        loss = loss + reg_loss * self.weight_decay
        if profile:
            device_sync(); t1 = time()
        # set_to_none avoids clearing large dense embedding gradients with an extra fill.
        # It preserves optimizer semantics here because all LightGCN embeddings receive
        # gradients through full-graph propagation on every BPR step.
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.opt.step()
        if profile:
            device_sync(); t2 = time()
            return loss.detach().cpu().item(), {'forward_graph_s': t1-t0, 'backward_step_s': t2-t1}
        return loss.detach().cpu().item()

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / k
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data, r, k):
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    idcg = np.sum(test_matrix * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    return np.sum(dcg / idcg)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = set()
        for item in test_data[i]:
            val = np.atleast_1d(item)[0]
            groundTrue.add(int(val))
            
        predictTopK = pred_data[i]
        pred = []
        for x in predictTopK:
            val_x = np.atleast_1d(x)[0]
            pred.append(1.0 if int(val_x) in groundTrue else 0.0)
            
        r.append(np.array(pred))
    return np.array(r).astype('float')

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

def getFileName():
    file = f"{world.model_name}-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    for i in range(0, len(tensors[0]), batch_size):
        yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays):
    indices = np.arange(len(arrays[0]))
    np.random.shuffle(indices)
    return tuple(x[indices] for x in arrays)

class timer:
    TAPE = [-1]
    NAMED_TAPE = {}
    @staticmethod
    def get(): return timer.TAPE.pop() if len(timer.TAPE) > 1 else -1
    def __init__(self, tape=None, **kwargs):
        self.named = kwargs.get('name', False)
        self.tape = tape or timer.TAPE
    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] = timer.NAMED_TAPE.get(self.named, 0.) + (time() - self.start)
        else:
            self.tape.append(time() - self.start)
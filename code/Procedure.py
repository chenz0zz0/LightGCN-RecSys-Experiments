# -*- coding: utf-8 -*-
"""
Training and Evaluation Process Logic.
Original code inspired by: https://github.com/gusye1234/LightGCN-PyTorch
Modified by: chenz0zz0

Key Optimizations:
1. Tuple Wrapper Fix: Removed tuple wrappers during minibatch iterations to fix indexing bugs.
2. Device Compatibility: Moved test ratings to CPU before advanced indexing to support 
   Apple Silicon (MPS) safely without memory crashing.
"""

import world
import numpy as np
import torch
import utils
from utils import timer

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    import time as _time
    epoch_t0 = _time.perf_counter()
    sample_t0 = _time.perf_counter()
    S = utils.UniformSample_original(dataset)
    sample_s = _time.perf_counter() - sample_t0

    transfer_t0 = _time.perf_counter()
    users = torch.as_tensor(S[:, 0], dtype=torch.long, device=world.device)
    posItems = torch.as_tensor(S[:, 1], dtype=torch.long, device=world.device)
    negItems = torch.as_tensor(S[:, 2], dtype=torch.long, device=world.device)
    utils.device_sync()
    transfer_s = _time.perf_counter() - transfer_t0

    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    forward_s = backward_s = 0.0

    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
        if world.profile:
            cri, prof = bpr.stageOne(batch_users, batch_pos, batch_neg, profile=True)
            forward_s += prof['forward_graph_s']; backward_s += prof['backward_step_s']
        else:
            cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    aver_loss = aver_loss / total_batch
    epoch_s = _time.perf_counter() - epoch_t0
    info = f"loss {aver_loss:.3f}"
    if world.profile:
        other_s = max(0.0, epoch_s - sample_s - transfer_s - forward_s - backward_s)
        info += (f" | PROFILE total={epoch_s:.2f}s sample={sample_s:.2f}s "
                 f"h2d={transfer_s:.2f}s forward+graph={forward_s:.2f}s "
                 f"backward+step={backward_s:.2f}s other={other_s:.2f}s batches={total_batch}")
    return info

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall), 'precision': np.array(pre), 'ndcg': np.array(ndcg)}

def Test(dataset, Recmodel, epoch, w=None, multicore=0, eval_dict=None, split_name='test', extra_exclude_dict=None):
    u_batch_size = world.config['test_u_batch_size']
    testDict: dict = dataset.testDict if eval_dict is None else eval_dict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    with torch.no_grad():
        users = list(testDict.keys())
        rating_list, groundTrue_list = [], []
        
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            batch_users = batch_users[0]
            
            allPos = dataset.getUserPosItems(batch_users)
            
            groundTrue = []
            for u in batch_users:
                uid = int(np.atleast_1d(u)[0])
                groundTrue.append(testDict[uid])
                
            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
            rating = Recmodel.getUsersRating(batch_users_gpu)
            
            rating = rating.cpu()
            
            exclude_index, exclude_items = [], []
            for range_i, (uid_raw, items) in enumerate(zip(batch_users, allPos)):
                # Always mask training positives.  During the final test of a
                # validation-selected run, also mask held-out validation positives
                # so they cannot occupy recommendation slots meant for test items.
                known_items = set(int(x) for x in items)
                if extra_exclude_dict is not None:
                    uid = int(np.atleast_1d(uid_raw)[0])
                    known_items.update(int(x) for x in extra_exclude_dict.get(uid, []))
                exclude_index.extend([range_i] * len(known_items))
                exclude_items.extend(known_items)
            if exclude_items:
                rating[exclude_index, exclude_items] = -(1 << 10)
            
            _, rating_K = torch.topk(rating, k=max_K)
            rating_list.append(rating_K)
            groundTrue_list.append(groundTrue)
            
        X = zip(rating_list, groundTrue_list)
        pre_results = [test_one_batch(x) for x in X]
        
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        
        print(f"EPOCH[{epoch+1}] {split_name} Results: {results}")
        return results
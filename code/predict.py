# -*- coding: utf-8 -*-
"""Exact Top-K recommendation from a trained LightGCN checkpoint."""

import world
import torch
import utils
import register
from register import dataset
import numpy as np

def single_user_predict(user_id, top_k=10):
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    weight_file = utils.getFileName()
    Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
    Recmodel.eval()

    with torch.no_grad():
        all_users, all_items = Recmodel.computer()
        
        user_emb = all_users[user_id].unsqueeze(0)
        
        scores = torch.matmul(user_emb, all_items.t())
        
        pos_items = dataset.getUserPosItems([user_id])[0]
        scores[0, pos_items] = -1e10
        
        _, top_items = torch.topk(scores, k=top_k)
        
        return top_items.cpu().numpy()[0]

if __name__ == "__main__":
    test_user = 10 
    recs = single_user_predict(test_user)
    print(f"Top 10 Recommendations for User {test_user}: {recs}")
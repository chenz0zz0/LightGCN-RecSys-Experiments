# -*- coding: utf-8 -*-
"""
Inference & Prediction Logic.
Original architecture inspired by: https://github.com/gusye1234/LightGCN-PyTorch
Author/Added by: chenz0zz0

Key Optimizations:
1. Decoupled Inference API: Provides `single_user_predict` for easy integration into 
   backend services (e.g., Flask/FastAPI).
2. Business Logic Filtering: Excludes positively interacted items directly within the 
   vector retrieval logic.
3. Exact Top-K Retrieval: Scores all items by user-item inner product and applies
   `torch.topk`; this implementation does not use an approximate-nearest-neighbor index.
"""

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
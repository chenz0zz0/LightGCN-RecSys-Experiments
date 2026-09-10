# -*- coding: utf-8 -*-
"""LightGCN training core with validation-based model selection and fixed-epoch final evaluation."""
import csv
import json
import os
import time
from os.path import join

import torch

import Procedure
import register
import utils
import world
from register import dataset
from world import cprint

utils.set_seed(world.seed)
print(">>SEED:", world.seed)
print(">>DEVICE:", world.device)

Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
weight_file = utils.getFileName()

if world.LOAD:
    Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
    cprint(f"loaded model weights from {weight_file}")

run_dir = os.path.abspath(join(world.results_dir, world.run_name))
os.makedirs(run_dir, exist_ok=True)
history_path = join(run_dir, "history.csv")
summary_path = join(run_dir, "summary.json")
best_path = join(run_dir, "best_model.pt")

use_validation = bool(getattr(dataset, "valDict", {}))
best_recall = float("-inf")
best_epoch = None
stale_evals = 0
history = []
start_time = time.time()

for epoch in range(world.TRAIN_epochs):
    train_info = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=1, w=None)
    print(f"EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {train_info}")

    if use_validation and (epoch + 1) % world.eval_every == 0:
        cprint("[VALIDATION]")
        val = Procedure.Test(dataset, Recmodel, epoch, multicore=0,
                             eval_dict=dataset.valDict, split_name="validation")
        recall20 = float(val["recall"][0])
        row = {"epoch": epoch + 1, "val_recall": recall20,
               "val_precision": float(val["precision"][0]),
               "val_ndcg": float(val["ndcg"][0])}
        history.append(row)
        if recall20 > best_recall:
            best_recall = recall20
            best_epoch = epoch + 1
            stale_evals = 0
            torch.save(Recmodel.state_dict(), best_path)
            cprint(f"Best validation checkpoint saved: Recall@{world.topks[0]}={best_recall:.6f}")
        else:
            stale_evals += 1
            print(f"Patience: {stale_evals}/{world.patience}")
            if stale_evals >= world.patience:
                cprint(f"Early stopping at epoch {epoch + 1}; best epoch={best_epoch}")
                break

if use_validation:
    Recmodel.load_state_dict(torch.load(best_path, map_location=world.device))
else:
    # Original benchmark-compatible mode: no test-set model selection.
    torch.save(Recmodel.state_dict(), best_path)
    best_epoch = epoch + 1

cprint("[FINAL TEST]")
test = Procedure.Test(
    dataset, Recmodel, best_epoch - 1, multicore=0,
    eval_dict=dataset.testDict, split_name="test",
    extra_exclude_dict=dataset.valDict if use_validation else None,
)

if history:
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

summary = {
    "dataset": world.dataset,
    "model": world.model_name,
    "seed": world.seed,
    "device": str(world.device),
    "epochs_requested": world.TRAIN_epochs,
    "best_epoch": best_epoch,
    "validation_ratio": world.config.get("val_ratio", 0.0),
    "selection_metric": f"validation Recall@{world.topks[0]}" if use_validation else "none (fixed-epoch protocol)",
    "test": {k: [float(x) for x in v] for k, v in test.items()},
    "elapsed_seconds": time.time() - start_time,
    "checkpoint": best_path,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

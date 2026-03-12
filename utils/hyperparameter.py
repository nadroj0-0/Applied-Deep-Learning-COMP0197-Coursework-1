"""
Hyperparameter search utilities.

Implements staged random search with progressive pruning.

The script does NOT implement training itself. Instead it expects
a training function to be passed in that trains for a given number
of epochs and returns validation metrics.
"""

import random
import math
import json
import time

from utils.training_session import *
from utils.common import *

class Leaderboard:
    """
    Simple leaderboard tracking validation losses.
    """
    def __init__(self, sessions):
        self.entries = []
        for cfg, session in sessions:
            loss = session.history["epoch_metrics"][-1]["validation_loss"]
            self.entries.append({
                "config": cfg,
                "session": session,
                "loss": loss
            })
    def add(self, cfg, session):
        loss = session.history["epoch_metrics"][-1]["validation_loss"]
        self.entries.append({
            "config": cfg,
            "session": session,
            "loss": loss
        })
    def ranked(self):
        return sorted(self.entries, key=lambda x: x["loss"])
    def top(self, k):
        return self.ranked()[:k]

def sample_uniform(low, high):
    return random.uniform(low, high)


def sample_log_uniform(low, high):
    return 10 ** random.uniform(math.log10(low), math.log10(high))


def sample_parameter(low, high,  mode):
    if mode == "uniform":
        return sample_uniform(low, high)
    if mode == "log":
        return sample_log_uniform(low, high)
    raise ValueError(f"Unknown sampling mode: {mode}")

def sample_config(base_config, search_space):
    cfg = {} if base_config is None else base_config.copy()
    for param, (low, high, mode) in search_space.items():
        cfg[param] = sample_parameter(low, high, mode)
    return cfg

def prune(sessions, keep):
    leaderboard = Leaderboard(sessions)
    best = leaderboard.top(keep)
    return [(e["config"], e["session"]) for e in best]

def select_best(sessions):
    best_loss = float("inf")
    best = None
    for cfg, session in sessions:
        loss = session.history["epoch_metrics"][-1]["validation_loss"]
        if loss < best_loss:
            best_loss = loss
            best = (cfg, session)
    return best


def staged_search(search_space,images,labels,train_loader,val_loader,method, model_dir, base_config=None,
                  dropout_prob=0.0, training_step=baseline_step, save_outputs=False, schedule=None,
                  initial_models=10, search_name="hyperparameter_search",**kwargs):
    """
    Generic successive-halving hyperparameter search.
    """
    if schedule is None:
        schedule = [
            {"epochs": 10, "keep": 5, "new": 0},
            {"epochs": 10, "keep": 2, "new": 3},
            {"epochs": 20, "keep": 1, "new": 1},
        ]
    sessions = []
    run_records = []
    # --- initialise configs ---
    for i in range(initial_models):
        cfg = sample_config(base_config, search_space)
        session = create_training_session(images, labels, method, cfg.get("reg_dropout", dropout_prob), cfg,
                                          training_step, **cfg)
        sessions.append((cfg, session))
    # --- run remaining stages ---
    for stage_idx, stage in enumerate(schedule):
        epochs, keep = stage["epochs"], stage["keep"]
        new = stage.get("new", 0)
        print(f"\nStage {stage_idx}: training {len(sessions)} models for {epochs} epochs")
        for i, (cfg, session) in enumerate(sessions):
            full_train(name=f"search_stage{stage_idx}_{i}",images=images,labels=labels,train_loader=train_loader,
                       val_loader=val_loader,method=method,epochs=epochs,model_dir=model_dir,config=cfg,
                       dropout_prob=cfg.get("reg_dropout", dropout_prob),training_step=training_step,save_outputs=False,session=session)
        if keep is not None:
            sessions = prune(sessions, keep=keep)
            print(f"Pruned to {len(sessions)} models")
            # --- inject new random models (exploration) ---
            if new > 0:
                print(f"Injecting {new} new models")
                for j in range(new):
                    cfg = sample_config(base_config, search_space)
                    session = create_training_session(images,labels,method, cfg.get("reg_dropout", dropout_prob),cfg,
                                                      training_step,**cfg)
                    sessions.append((cfg, session))
    for cfg, session in sessions:
        run_records.append({
            "config": cfg.copy(),
            "total_epochs": session.epoch,
            "epoch_metrics": session.history["epoch_metrics"]
        })
    best_cfg, best_session = select_best(sessions)
    best_metrics = best_session.history["epoch_metrics"][-1]
    search_summary = {
        "search_type": "successive_halving",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_models": initial_models,
        "schedule": schedule,
        "search_space": search_space,
        "best_config": best_cfg,
        "best_validation_loss": best_metrics["validation_loss"],
        "best_validation_accuracy": best_metrics["validation_accuracy"],
        "runs": run_records
    }
    model_dir.mkdir(exist_ok=True)
    summary_path = model_dir / f"{search_name}.json"
    save_json(search_summary, summary_path)
    print(f"\nHyperparameter search summary saved to: {summary_path}")
    return best_cfg
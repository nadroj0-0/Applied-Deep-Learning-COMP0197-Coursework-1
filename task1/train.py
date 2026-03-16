import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.experiment import *

TASK_DIR = Path(__file__).resolve().parent

TRAIN_CONFIG = {
    'seed': 42,
    "epochs": 80,
    "optimiser": "SGD",
    "lr": 0.01619621907336969,
    "momentum": 0.9285729026103532,
    "weight_decay": 1.201179492735599e-06,
    "reg_dropout": 0.2193025904119891,
    "batch_size": 64,
    "validation_fraction": 0.2
}

SEARCH = False

BASE_SEARCH_SPACE = {
    "lr": (1e-4, 1e-1, "log"),
    "momentum": (0.8, 0.99, "uniform"),
}

REG_SEARCH_SPACE = {
    "weight_decay": (1e-6, 1e-3, "log"),
    "reg_dropout": (0.1, 0.7, "uniform")
}

FULL_REG_SEARCH_SPACE = {
    "lr": (1e-4, 1e-1, "log"),
    "momentum": (0.8, 0.99, "uniform"),
    "weight_decay": (1e-6, 1e-3, "log"),
    "reg_dropout": (0.1, 0.7, "uniform"),
}
HYPER_PARAM_INIT_MODELS = 20
HYPER_PARAM_SEARCH_SCHEDULE = [
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 2)},
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 4)},
    {"epochs": 20, "keep": 1},
]


def main():
    try:
        cfg = TRAIN_CONFIG.copy()
    except NameError:
        raise RuntimeError(
            "TRAIN_CONFIG must be defined before calling main(). "
            "It defines the experiment hyperparameters."
        )

    # --- Baseline ---
    baseline = Experiment("baseline", cfg, model_dir=get_model_dir("baseline", TASK_DIR))
    baseline.run(
        search_space=BASE_SEARCH_SPACE if SEARCH else None,
        schedule=HYPER_PARAM_SEARCH_SCHEDULE,
        initial_models=HYPER_PARAM_INIT_MODELS
    )

    # --- Regularised (inherits baseline's tuned lr/momentum) ---
    regularised = Experiment("regularised", baseline.cfg, model_dir=get_model_dir("regularised", TASK_DIR))
    regularised.run(
        search_space=REG_SEARCH_SPACE if SEARCH else None,
        augment=True,
        use_regularisation=True,
        schedule=HYPER_PARAM_SEARCH_SCHEDULE,
        initial_models=HYPER_PARAM_INIT_MODELS
    )

    # --- Full regularised (free search, commented out for submission) ---
    # if SEARCH:
    #     full_reg = Experiment("full_regularised", cfg, model_dir=get_model_dir("full_regularised", TASK_DIR))
    #     full_reg.run(
    #         search_space=FULL_REG_SEARCH_SPACE,
    #         augment=True,
    #         use_regularisation=True,
    #         schedule=HYPER_PARAM_SEARCH_SCHEDULE,
    #         initial_models=HYPER_PARAM_INIT_MODELS
    #     )



if __name__ == '__main__':
    main()
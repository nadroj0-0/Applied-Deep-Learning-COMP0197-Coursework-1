import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.experiment import *

TASK_DIR = Path(__file__).resolve().parent

TRAIN_CONFIG = {
    "seed": 42,
    "epochs": 50,
    "optimiser": "SGD",
    "lr": 0.02680966393455427,
    "momentum": 0.8012347643388317,
    "weight_decay": 1.897743576692889e-06,
    "reg_dropout": 0.1580298261000784,
    "batch_size": 64,
    "validation_fraction": 0.2,
    "mixup_alpha": 0.4,
    "label_smoothing": 0.05,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-4
}
SEARCH = True
BASELINE_FIXED_SEARCH_SPACE = {
    "mixup_alpha": (0.1, 0.8, "uniform"),
    "label_smoothing": (0.01, 0.2, "uniform"),
}

REG_FIXED_SEARCH_SPACE = {
    "mixup_alpha": (0.1, 0.8, "uniform"),
    "label_smoothing": (0.01, 0.2, "uniform"),
}

BASELINE_FREE_SEARCH_SPACE = {
    "lr": (1e-4, 1e-1, "log"),
    "momentum": (0.8, 0.99, "uniform"),
    "mixup_alpha": (0.1, 0.8, "uniform"),
    "label_smoothing": (0.01, 0.2, "uniform"),
}

REG_FREE_SEARCH_SPACE = {
    "lr": (1e-4, 1e-1, "log"),
    "momentum": (0.8, 0.99, "uniform"),
    "weight_decay": (1e-6, 1e-3, "log"),
    "reg_dropout": (0.1, 0.7, "uniform"),
    "mixup_alpha": (0.1, 0.8, "uniform"),
    "label_smoothing": (0.01, 0.2, "uniform"),
}

HYPER_PARAM_INIT_MODELS = 5
HYPER_PARAM_SEARCH_SCHEDULE = [
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 2)},
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 4)},
    {"epochs": 20, "keep": 1},
]


def main():
    try:
        cfg = TRAIN_CONFIG
    except NameError:
        raise RuntimeError(
            "TRAIN_CONFIG must be defined before calling main(). "
            "It defines the experiment hyperparameters."
        )

    experiments = {
        "baseline_fixed_mixup_ls": dict(
            search_space=BASELINE_FIXED_SEARCH_SPACE if SEARCH else None,
            training_step=mixup_smoothing_step,
            use_mixup=True,
            use_smoothing=True,
        ),
        "regularised_fixed_mixup_ls": dict(
            search_space=REG_FIXED_SEARCH_SPACE if SEARCH else None,
            training_step=mixup_smoothing_step,
            augment=True,
            use_regularisation=True,
            use_mixup=True,
            use_smoothing=True,
        ),
        "baseline_free_mixup_ls": dict(
            search_space=BASELINE_FREE_SEARCH_SPACE if SEARCH else None,
            training_step=mixup_smoothing_step,
            use_mixup=True,
            use_smoothing=True,
        ),
        "regularised_free_mixup_ls": dict(
            search_space=REG_FREE_SEARCH_SPACE if SEARCH else None,
            training_step=mixup_smoothing_step,
            augment=True,
            use_regularisation=True,
            use_mixup=True,
            use_smoothing=True,
        ),
    }

    for name, kwargs in experiments.items():
        exp = Experiment(name, cfg, model_dir=get_model_dir(name, TASK_DIR))
        exp.run(
            schedule=HYPER_PARAM_SEARCH_SCHEDULE,
            initial_models=HYPER_PARAM_INIT_MODELS,
            **kwargs
        )
if __name__ == "__main__":
    main()

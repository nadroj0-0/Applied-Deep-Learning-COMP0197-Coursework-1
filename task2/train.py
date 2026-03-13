import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.common import *
import torch
from utils.hyperparameter import staged_search


MODEL_DIR = Path(__file__).parent / "models"

TRAIN_CONFIG = {
    "seed": 42,
    "epochs": 100,
    "optimiser": "SGD",
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "validation_fraction": 0.2,
    "mixup_alpha": 0.4,
    "label_smoothing": 0.05,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-4
}

MIXUP_SMOOTH_SEARCH_SPACE = {
    "lr": (1e-4, 5e-2, "log"),
    "weight_decay": (1e-6, 1e-3, "log"),
    "mixup_alpha": (0.1, 0.8, "uniform"),
    "label_smoothing": (0.01, 0.2, "uniform"),
}

HYPER_PARAM_INIT_MODELS = 20
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
    generator = init_seed(cfg)
    train_dataset, test_dataset = download_data()

    images, labels, train_loader, val_loader = load_data_pytorch(
        train_dataset,
        batch_size=cfg["batch_size"],
        validation_fraction=cfg["validation_fraction"],
        generator=generator
    )

    # -------------------------------------------------
    # Hyperparameter search for MixUp + Label Smoothing
    # -------------------------------------------------
    best_mixup_smooth_cfg = None
    print("\nStarting mixup plus smoothing hyperparameter search (MixUp + Label Smoothing)")

    best_mixup_smooth_cfg = staged_search(MIXUP_SMOOTH_SEARCH_SPACE, images, labels, train_loader, val_loader,
                                           cfg["optimiser"], MODEL_DIR, base_config=cfg,
                                           training_step=mixup_smoothing_step, schedule=HYPER_PARAM_SEARCH_SCHEDULE,
                                           initial_models=HYPER_PARAM_INIT_MODELS, search_name="mixup_smoothing_hyperparameter_search")

    print("\nBest configuration found:")
    print(best_mixup_smooth_cfg)
    if best_mixup_smooth_cfg is not None:
        cfg = best_mixup_smooth_cfg.copy()

    # ---- test MixUp ----
    sample_images = images.to(device)
    sample_labels = labels.to(device)

    mixed_images, la, lb, lam = mixup_data(sample_images, sample_labels, alpha=cfg["mixup_alpha"])

    print("MixUp output shape:", mixed_images.shape)
    print("Lambda:", lam)

    # ---- test label smoothing ----
    outputs = torch.randn(4,10).to(device)
    test_labels = torch.tensor([1,3,5,2]).to(device)
    loss = label_smoothing_loss(outputs, test_labels, cfg["label_smoothing"])
    print("Label smoothing loss:", loss)

    experiments = [
        ("baseline_mixup", mixup_step, {"mixup_alpha": cfg["mixup_alpha"]}),
        ("baseline_smooth", smoothing_step, {"label_smoothing": cfg["label_smoothing"]}),
        ("baseline_mixup_smooth", mixup_smoothing_step, {"mixup_alpha": cfg["mixup_alpha"], "label_smoothing": cfg["label_smoothing"]}),
    ]
    for name, step_fn, params in experiments:
        print(f"\nRunning experiment: {name}")
        full_train(name, images,labels,train_loader,val_loader,cfg["optimiser"],
                   epochs=cfg["epochs"], model_dir=MODEL_DIR, config=cfg,
                   training_step=step_fn,**params,lr=cfg["lr"],momentum=cfg["momentum"],
                   weight_decay=cfg["weight_decay"]
        )

if __name__ == "__main__":
    main()
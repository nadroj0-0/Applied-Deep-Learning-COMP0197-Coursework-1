from utils.common import *
import torch
import torch.nn.functional as F
import numpy as np
import time

MODEL_DIR = Path(__file__).parent / "models"

TRAIN_CONFIG = {
    "seed": 42,
    "epochs": 50,
    "optimiser": "SGD",
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "validation_fraction": 0.2,
    "mixup_alpha": 0.4,
    "label_smoothing": 0.1,
    "early_stopping_patience": 5
}


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
        ("baseline_mixup", mixup_step, {"alpha": cfg["mixup_alpha"]}),
        ("baseline_smooth", smoothing_step, {"smoothing": cfg["label_smoothing"]}),
        ("baseline_mixup_smooth", mixup_smoothing_step, {"alpha": cfg["mixup_alpha"], "smoothing": cfg["label_smoothing"]}),
    ]
    for name, step_fn, params in experiments:
        print(f"\nRunning experiment: {name}")
        full_train(name, images,labels,train_loader,val_loader,cfg["optimiser"],
                   epochs=cfg["epochs"], model_dir=MODEL_DIR, config=TRAIN_CONFIG,
                   training_step=step_fn,**params,lr=cfg["lr"],momentum=cfg["momentum"],
                   weight_decay=cfg["weight_decay"]
        )

if __name__ == "__main__":
    main()
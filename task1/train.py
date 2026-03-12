import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.common import *
from utils.hyperparameter import staged_search

MODEL_DIR = Path(__file__).parent / "models"

TRAIN_CONFIG = {
    'seed': 42,
    'epochs': 50,
    'optimiser': 'SGD',
    'lr': 0.001,
    'momentum': 0.9,
    #'weight_decay': 1e-4,
    'weight_decay': 0,
    #'reg_dropout': 0.5,
    'reg_dropout': 0.0,
    'batch_size': 64,
    'validation_fraction': 0.2
}

BASE_SEARCH_SPACE = {
    "lr": (1e-4, 1e-1, "log"),
    "momentum": (0.8, 0.99, "uniform"),
}

REG_SEARCH_SPACE = {
    "weight_decay": (1e-6, 1e-3, "log"),
    "reg_dropout": (0.1, 0.7, "uniform")
}

HYPER_PARAM_SEARCH_SCHEDULE = [
    {"epochs":10, "keep":5, "new":3},
    {"epochs":10, "keep":2, "new":1},
    {"epochs":20, "keep":1, "new":0}
]
HYPER_PARAM_INIT_MODELS = 10

def main():
    try:
        cfg = TRAIN_CONFIG
    except NameError:
        raise RuntimeError(
            "TRAIN_CONFIG must be defined before calling main(). "
            "It defines the experiment hyperparameters."
        )
    generator = init_seed(cfg)
    train_dataset, _ = download_data()
    images, labels, train_loader, val_loader = load_data_pytorch(
        train_dataset, batch_size=cfg['batch_size'],
        validation_fraction=cfg['validation_fraction'],
        generator=generator
    )
    inspect_data(images, labels, train_dataset)
    best_base_cfg, best_reg_cfg = None, None
    print("\nStarting baseline hyperparameter search")
    best_base_cfg = staged_search(BASE_SEARCH_SPACE,images,labels,train_loader,val_loader,
                                      cfg["optimiser"], MODEL_DIR, base_config=cfg, schedule = HYPER_PARAM_SEARCH_SCHEDULE,
                                      initial_models=HYPER_PARAM_INIT_MODELS, search_name="baseline_hyperparameter_search")
    print("\nBest baseline configuration:")
    print(best_base_cfg)
    if best_base_cfg is not None:
        cfg = best_base_cfg.copy()
        best_reg_cfg = best_base_cfg.copy()
    base_model, base_history, base_model_path, base_history_path = full_train(
        'baseline', images, labels, train_loader, val_loader,
        cfg['optimiser'], epochs=cfg['epochs'], model_dir=MODEL_DIR,
        config=cfg, lr=cfg['lr'], momentum=cfg['momentum']
    )
    print('\nBase model:')
    print(base_model)
    print('\nBase final epoch metrics:')
    print(base_history['epoch_metrics'][-1])


    print("\nStarting regularised hyperparameter search")
    best_reg_cfg = staged_search(REG_SEARCH_SPACE, images, labels, train_loader, val_loader,
                                      cfg["optimiser"], MODEL_DIR, base_config=best_reg_cfg,
                                      schedule=HYPER_PARAM_SEARCH_SCHEDULE,
                                      initial_models=HYPER_PARAM_INIT_MODELS, search_name="regularised_hyperparameter_search" )
    print("\nBest regularised configuration:")
    print(best_reg_cfg)
    if best_reg_cfg is not None:
        cfg = best_reg_cfg.copy()
    reg_model, reg_history, reg_model_path, reg_history_path = full_train(
        'regularised', images, labels, train_loader, val_loader,
        cfg['optimiser'], epochs=cfg['epochs'], model_dir=MODEL_DIR,
        config=cfg, lr=cfg['lr'], momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'], dropout_prob=cfg['reg_dropout']
    )
    print('\nRegularised model:')
    print(reg_model)
    print('\nRegular final epoch metrics:')
    print(reg_history['epoch_metrics'][-1])

if __name__ == '__main__':
    main()
from utils.common import *
import time

MODEL_DIR = Path(__file__).parent / "models"

TRAIN_CONFIG = {
    'seed': 42,
    'epochs': 50,
    'optimiser': 'SGD',
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'reg_dropout': 0.5,
    'batch_size': 64,
    'validation_fraction': 0.2
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
        train_dataset, batch_size=cfg['batch_size'],
        validation_fraction=cfg['validation_fraction'],
        generator=generator
    )
    inspect_data(images, labels, train_dataset)

    base_model, base_history, base_model_path, base_history_path = full_train(
        'baseline', images, labels, train_loader, val_loader,
        cfg['optimiser'], epochs=cfg['epochs'], config=TRAIN_CONFIG, lr=cfg['lr'], momentum=cfg['momentum']
    )
    print('\nBase model:')
    print(base_model)
    print('\nBase final epoch metrics:')
    print(base_history['epoch_metrics'][-1])
    reg_model, reg_history, reg_model_path, reg_history_path = full_train(
        'regularised', images, labels, train_loader, val_loader,
        cfg['optimiser'], epochs=cfg['epochs'], model_dir=MODEL_DIR,
        config=TRAIN_CONFIG, lr=cfg['lr'], momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'], dropout_prob=cfg['reg_dropout']
    )
    print('\nRegularised model:')
    print(reg_model)
    print('\nRegular final epoch metrics:')
    print(reg_history['epoch_metrics'][-1])

if __name__ == '__main__':
    main()
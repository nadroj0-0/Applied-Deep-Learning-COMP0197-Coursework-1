from utils.common import *
import time


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


def full_train(name, images, labels, train_loader, val_loader, method, epochs, dropout_prob=0.0, **kwargs):
    start_time = time.time()
    model, outputs = init_model(images, dropout_prob)
    criterion, loss = init_loss(outputs, labels)
    #optim_method = init_optimiser(model, 'SGD', lr=0.001, momentum=0.9)
    optim_method = init_optimiser(model, method, **kwargs)
    #batch_losses, epoch_losses = train_model(epochs, train_loader, model, criterion, optim_method)
    history = train_model(epochs, train_loader, val_loader, model, criterion, optim_method)
    model_path = save_model(model, name)
    history_path = save_history(history, name, 'train', model, config=TRAIN_CONFIG)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{name} training completed in {elapsed:.2f} seconds")
    #return model, batch_losses, epoch_losses
    return model, history, model_path, history_path

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
        cfg['optimiser'], epochs=cfg['epochs'], lr=cfg['lr'], momentum=cfg['momentum']
    )
    print('\nBase model:')
    print(base_model)
    print('\nBase final epoch metrics:')
    print(base_history['epoch_metrics'][-1])
    reg_model, reg_history, reg_model_path, reg_history_path = full_train(
        'regularised', images, labels, train_loader, val_loader,
        cfg['optimiser'], epochs=cfg['epochs'], lr=cfg['lr'], momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'], dropout_prob=cfg['reg_dropout']
    )
    print('\nRegularised model:')
    print(reg_model)
    print('\nRegular final epoch metrics:')
    print(reg_history['epoch_metrics'][-1])

if __name__ == '__main__':
    main()
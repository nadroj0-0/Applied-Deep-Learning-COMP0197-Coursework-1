# GenAI usage statement: Claude (Anthropic) was used in an assistive role to help
# structure and refine this utility class. All implementation details and design
# decisions are the author's own.
import copy


class EarlyStopping:
    """
    Validation-based early stopping utility.

    Monitors validation loss during training and stops optimisation when no
    improvement is observed for a specified number of consecutive epochs
    ('patience'). The best model state is stored and can be restored after
    training.

    Attributes:
        patience (int): Number of epochs to wait without improvement before stopping.
        min_delta (float): Minimum change in validation loss required to qualify as improvement.
        best_val_loss (float): Lowest observed validation loss.
        best_epoch (int | None): Epoch at which best validation loss occurred.
        best_model_state (dict | None): Deep copy of model parameters at best epoch.
        counter (int): Number of consecutive epochs without improvement.
        stopped_epoch (int | None): Epoch at which early stopping was triggered.
        triggered (bool): Whether early stopping has been activated.
    """
    def __init__(self, patience: int, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_model_state = None
        self.counter = 0
        self.stopped_epoch = None
        self.triggered = False
    def update(self, val_loss, model, epoch):
        """
        Updates early stopping state using current validation loss.

    If validation loss improves beyond the defined threshold (min_delta),
    the best model state is updated and the counter is reset. Otherwise,
    the counter is incremented. Training is stopped once the counter
    exceeds the patience value.

    Args:
        val_loss (float): Current epoch validation loss.
        model (torch.nn.Module): Model being trained.
        epoch (int): Current training epoch.

    Returns:
        bool: True if early stopping condition is met, else False.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            self.triggered = True
            return True
        return False
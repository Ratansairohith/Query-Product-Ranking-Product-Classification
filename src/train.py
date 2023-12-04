from utils.metric_monitor import MetricMonitor
from tqdm import tqdm
from utils.config import config_params as params


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler=None):
    """
    The `train_fn` function is designed for training a deep learning model using a training data loader.
    It iterates through batches of data, computes losses, and updates model weights through backpropagation.

    Parameters:
    - train_loader: DataLoader
      - A PyTorch DataLoader object that provides batches of training data.
    - model: nn.Module
      - The deep learning model to be trained.
    - criterion: nn.Module
      - The loss function used for computing the loss between model predictions and target labels.
    - optimizer: torch.optim.Optimizer
      - The optimizer responsible for updating the model's parameters based on the computed gradients.
    - epoch: int
      - The current training epoch.
    - scheduler: torch.optim.lr_scheduler._LRScheduler, optional (default=None)
      - A learning rate scheduler that adjusts the learning rate during training.

    Returns:
    - None

    Description:
    - The `train_fn` function plays a central role in the training process of a deep learning model.
    - It sets the model in training mode, initializes a `MetricMonitor` to keep track of training metrics,
      and iterates through the training data loader.
    - For each batch of data, it performs the following steps:
      - Transfers input data, masks, and targets to the specified device (typically a GPU).
      - Computes model predictions (`logits`) by forwarding the input data through the model.
      - Calculates the loss between the predicted logits and the target labels using the specified loss criterion.
      - Updates the metric monitor with the computed loss.
      - Performs backpropagation by calling `backward` on the loss tensor.
      - Updates the model's parameters using the optimizer and, if provided, adjusts the learning rate with the scheduler.
      - Clears gradients with `optimizer.zero_grad()` to prepare for the next batch.
    - The function provides real-time progress updates using the `tqdm` library, displaying the current training
      epoch and metrics.
    - Training metrics, such as loss, are monitored and displayed to track model performance during training.

    The `train_fn` function is a fundamental component of training deep learning models and allows for efficient
    model optimization based on provided training data.
    """

    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)

    for i, batch in enumerate(stream, start=1):
        ids = batch['ids'].to(params['device'])
        mask = batch['mask'].to(params['device'])
        token_type_ids = batch['token_type_ids'].to(params['device'])
        target = batch['target'].to(params['device'])

        logits, _ = model(ids, token_type_ids, mask)
        loss = criterion(logits, target)
        metric_monitor.update('Loss', loss.item())
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")
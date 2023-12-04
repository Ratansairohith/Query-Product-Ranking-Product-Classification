from torch.optim.lr_scheduler import OneCycleLR
from utils.config import config_params as params


def get_scheduler(df, optimizer, scheduler_params=params):
    """
    The `get_scheduler` function is designed to create a learning rate scheduler for optimizing a deep learning model's
    training process. Specifically, it uses the OneCycleLR scheduler, a popular technique for scheduling learning rates
    during training.

    Parameters:
    - df: DataFrame
      - The input DataFrame containing the data for which the scheduler is being created. The DataFrame shape is used
        to determine the number of steps per epoch.
    - optimizer: torch.optim.Optimizer
      - The optimizer object responsible for updating the model's parameters.
    - scheduler_params: dict, optional (default=params)
      - A dictionary containing the parameters for configuring the OneCycleLR scheduler. These parameters control the
        learning rate schedule.

    Returns:
    - scheduler: torch.optim.lr_scheduler.OneCycleLR
      - An instance of the OneCycleLR scheduler tailored to the provided optimizer and training configuration.

    Description:
    - The `get_scheduler` function is an essential component of training deep learning models. It sets up a learning
      rate scheduler to adjust the learning rates during training.
    - The OneCycleLR scheduler is used to gradually increase and then decrease the learning rate during training,
      which can lead to faster convergence and better model performance.
    - The function takes as input the training data in the form of a DataFrame (`df`), the optimizer used for training
      (`optimizer`), and optional scheduler parameters (`scheduler_params`).
    - The scheduler parameters include the maximum learning rate (`max_lr`), the number of steps per epoch, the total
      number of training epochs, the percentage of the training phase for increasing the learning rate (`pct_start`),
      the annealing strategy (`anneal_strategy`), the division factor for learning rate annealing (`div_factor`),
      and the final division factor (`final_div_factor`).
    - The OneCycleLR scheduler is configured based on these parameters and returned as the result.
    - Incorporating a learning rate scheduler in training can help improve model training efficiency and convergence,
      making it an important tool in deep learning projects.
    """

    scheduler = OneCycleLR(
        optimizer,
        max_lr=scheduler_params['max_lr'],
        steps_per_epoch=int(df.shape[0] / params['batch_size']) + 1,
        epochs=scheduler_params['epochs'],
        pct_start=scheduler_params['pct_start'],
        anneal_strategy=scheduler_params['anneal_strategy'],
        div_factor=scheduler_params['div_factor'],
        final_div_factor=scheduler_params['final_div_factor'],
    )
    return scheduler

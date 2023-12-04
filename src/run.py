import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.dataset import ESCIdataset
from model import ProductModel
from utils.losses import calculate_class_weights, FocalLoss
from utils.scheduler import get_scheduler
from train import train_fn
from validate import validate_fn
from utils.config import config_params as params


def train_and_validate_folds(train_df, params):
    """
    The `train_and_validate_folds` function facilitates the training and validation of a deep learning model using
    a k-fold cross-validation approach. It iterates through multiple folds of the dataset, each time training the model
    on a subset and validating it on another.

    Parameters:
    - train_df: DataFrame
      - The input DataFrame containing the dataset to be used for training and validation.
    - params: dict
      - A dictionary containing various hyperparameters and configuration settings for model training.

    Returns:
    - best_models_of_each_fold: list
      - A list containing the filenames of the best saved models from each fold.

    Description:
    - The `train_and_validate_folds` function is a versatile tool for training and evaluating deep learning models
      using k-fold cross-validation, a common technique for assessing model performance.
    - It initializes an empty list `best_models_of_each_fold` to store the filenames of the best models found in each
      fold.
    - The function iterates through each fold, separating the dataset into a training set and a validation set.
    - For each fold, it performs the following steps:
      - Creates training and validation datasets using the `ESCIdataset` class, ensuring proper data preparation.
      - Initializes the model, loss function, optimizer, and scheduler according to the provided hyperparameters.
      - Trains the model using the `train_fn` function and validates it using the `validate_fn` function.
      - Keeps track of the best model based on the lowest validation loss and saves its state.
      - Prints a summary of the best model's performance for the fold.
    - After completing all folds, the function returns a list containing the filenames of the best models found in
      each fold.
    - The `train_and_validate_folds` function is a robust tool for optimizing and evaluating deep learning models,
      providing insights into their performance across multiple data subsets.

    """

    best_models_of_each_fold = []

    for fold in range(params['num_folds']):
        print(f'******************** Training Fold: {fold + 1} ********************')
        current_fold = fold + 1
        df_train = train_df[train_df['fold'] != current_fold].copy()
        df_valid = train_df[train_df['fold'] == current_fold].copy()

        train_dataset = ESCIdataset(df_train)
        valid_dataset = ESCIdataset(df_valid)

        train_dataloader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], pin_memory=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=params['batch_size'] * 2, shuffle=False,
            num_workers=params['num_workers'], pin_memory=True
        )
        label_map = {'exact': 0, 'substitute': 1, 'irrelevant': 2, 'complement': 3}
        class_weights = calculate_class_weights(train_df['esci_label'].map(label_map).to_list())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(params['device'])

        model = ProductModel()
        model = model.to(params['device'])
        criterion = FocalLoss(gamma=2.0, weight=class_weights)

        if params['no_decay']:
            no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=params['lr'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'])

        scheduler = get_scheduler(train_df, optimizer)

        # Training and Validation Loop
        best_loss = np.inf
        best_epoch = 0
        best_model_name = None
        for epoch in range(1, params['epochs'] + 1):
            train_fn(train_dataloader, model, criterion, optimizer, epoch, scheduler)
            valid_loss, ndcg, f1_micro, f1_micro_binary = validate_fn(valid_dataloader, model, criterion, epoch)
            if valid_loss <= best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                if best_model_name is not None:
                    os.remove(best_model_name)
                torch.save(model.state_dict(), f"{params['checkpoint'].split('/')[1]}_{epoch}_epoch_f{fold + 1}.pth")
                best_model_name = f"{params['checkpoint'].split('/')[1]}_{epoch}_epoch_f{fold + 1}.pth"

        # Print summary of this fold
        print('')
        print(f'The best LOSS: {best_loss} for fold {fold + 1} was achieved on epoch: {best_epoch}.')
        print(f'The Best saved model is: {best_model_name}')
        best_models_of_each_fold.append(best_model_name)
        del df_train, df_valid, train_dataset, valid_dataset, train_dataloader, valid_dataloader, model
        _ = gc.collect()
        torch.cuda.empty_cache()

    return best_models_of_each_fold


if __name__ == "__main__":
    # Define your params and load your train_df here
    train_df = pd.read_csv('../data/training_set_fraction.csv')
    train_df['product_title'].fillna('', inplace=True)
    train_df['product_brand'].fillna('', inplace=True)
    train_df['product_color_name'].fillna('', inplace=True)
    best_models = train_and_validate_folds(train_df, params)

    # You can perform any further actions or analysis with best_models here

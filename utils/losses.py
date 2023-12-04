import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def calculate_class_weights(labels):
    """
    `calculate_class_weights` Function:
    - Parameters:
        - labels: ndarray or list
            - An array or list containing class labels for a dataset.
    - Returns:
        - weights: ndarray
            - An array of class weights computed based on the frequency of each class in the input labels.
    - Description:
        - The `calculate_class_weights` function calculates class weights for a given set of class labels.
        - It uses the frequency of each class in the dataset to compute the weights, with the intention of addressing
          class imbalance during model training.
        - The function returns an array of class weights that can be used during loss computation to give more importance
          to underrepresented classes.
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts)
    return weights


class FocalLoss(nn.Module):
    """
    `FocalLoss` Class:
    - Parameters:
        - weight: torch.Tensor, optional (default=None)
            - A tensor containing class weights. If provided, it can be used to assign custom weights to different
              classes.
        - gamma: float, optional (default=2.0)
            - A hyperparameter that controls the focusing strength of the Focal Loss. Higher values of gamma give more
              focus to hard-to-classify examples.
        - reduction: str, optional (default='mean')
            - Specifies the reduction to apply to the loss. It can be 'none', 'mean', or 'sum'.
    - Description:
        - The `FocalLoss` class implements the Focal Loss, which is especially useful for addressing class imbalance and
          focusing on difficult-to-classify samples.
        - It inherits from the `nn.Module` class in PyTorch and can be used as a custom loss function during model training.
        - The `forward` method computes the Focal Loss based on the input logits and target labels.
        - It first calculates the Cross-Entropy (CE) loss and then applies the Focal Loss formula to it.
        - The `gamma` parameter controls how much the loss is focused on hard examples, and the `weight` parameter can
          be used to assign custom class weights.
        - The `reduction` parameter specifies how the loss should be reduced across all samples.
    """

    def __init__(self, weight=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt) ** self.gamma * CE_loss).mean() if self.reduction == 'mean' else (
                    (1 - pt) ** self.gamma * CE_loss)
        return F_loss

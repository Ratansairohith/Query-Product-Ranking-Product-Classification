from sklearn.metrics import f1_score, ndcg_score
from tqdm import tqdm
import numpy as np
import torch
from utils.config import config_params as params


def validate_fn(val_loader, model, criterion, epoch):
    """
    The `validate_fn` function is responsible for evaluating a deep learning model's performance
    on a validation dataset. It computes various evaluation metrics, including loss,
    nDCG (Normalized Discounted Cumulative Gain), and F1 scores for different classification tasks.

    Parameters:
    - val_loader: DataLoader
      - A PyTorch DataLoader object that provides batches of validation data.
    - model: nn.Module
      - The deep learning model to be evaluated.
    - criterion: nn.Module
      - The loss function used for computing the loss between model predictions and target labels.
    - epoch: int
      - The current validation epoch.

    Returns:
    - avg_loss: float
      - The average loss computed during validation.
    - avg_ndcg: float
      - The average nDCG score computed during validation.
    - f1_micro: float
      - The micro-averaged F1 score computed across all classes during validation.
    - f1_micro_binary: float
      - The micro-averaged F1 score specifically for binary classification during validation.

    Description:
    - The `validate_fn` function is a crucial component for assessing the performance of a
      trained deep learning model on a separate validation dataset.
    - It operates in evaluation mode by setting the model to `eval()` and computes various metrics
      for model evaluation.
    - During the evaluation process, it iterates through the validation data loader, making predictions using
      the model and calculating loss values.
    - Additionally, it computes the nDCG score for task 1, micro-averaged F1 scores for tasks 2 and 3, and reports
      these metrics.
    - The function also calculates and returns the average loss, nDCG score, micro-averaged F1 score for all classes,
      and micro-averaged F1 score for binary classification.
    - These metrics provide insights into the model's performance on different aspects of the validation data.

    The `validate_fn` function is essential for assessing the quality of a trained model and guiding further
    adjustments or fine-tuning as needed.
    """

    model.eval()
    all_loss = []
    all_targets = []
    all_predictions = []
    all_probabilities = []
    relevance_map = {'Exact': 1.0, 'Substitute': 0.1, 'Complement': 0.01, 'Irrelevant': 0.0}
    substitute_index = 1  # Update this index to match the 'Substitute' class in your data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader), start=1):
            ids = batch['ids'].to(params['device'])
            mask = batch['mask'].to(params['device'])
            token_type_ids = batch['token_type_ids'].to(params['device'])
            targets = batch['target'].to(params['device'])

            logits, probabilities = model(ids, token_type_ids, mask)
            loss = criterion(logits, targets)
            all_loss.append(loss.item())
            # print(probs)
            probabilities = probabilities.cpu().numpy()
            all_probabilities.append(probabilities)

            # Convert the targets to a one-hot encoded format if not already one-hot
            all_targets.append(targets.cpu().numpy())

            # Get the predicted class from probabilities for F1 calculation
            predicted_classes = np.argmax(probabilities, axis=1)
            all_predictions.append(predicted_classes)

    # Concatenate results from all batches
    all_targets = np.vstack(all_targets)
    all_probabilities = np.vstack(all_probabilities)
    all_predictions = np.concatenate(all_predictions)

    # Prepare relevance scores based on the predicted classes for nDCG
    relevance_scores = np.array([relevance_map['Exact'] if t[0] == 1 else
                                 relevance_map['Substitute'] if t[1] == 1 else
                                 relevance_map['Complement'] if t[2] == 1 else
                                 relevance_map['Irrelevant'] for t in all_targets])

    # Calculate average loss
    avg_loss = np.mean(all_loss)

    # Task 1: Calculate nDCG
    scores_pred = all_probabilities[np.arange(len(all_predictions)), all_predictions]
    avg_ndcg = ndcg_score([relevance_scores], [scores_pred], k=5)

    # Task 2: Micro Averaged F1 Score across all classes
    f1_micro = f1_score(np.argmax(all_targets, axis=1), all_predictions, average='micro')

    # Task 3: Micro Averaged F1 Score for binary substitute classification
    binary_targets = all_targets[:, substitute_index]
    substitute_probabilities = all_probabilities[:, 1]
    binary_predictions = (substitute_probabilities > 0.3).astype(int)
    f1_micro_binary = f1_score(binary_targets, binary_predictions, average='micro')

    print(
        f"Epoch: {epoch:02}. Valid. Loss: {avg_loss:.4f}. Task 1 nDCG: {avg_ndcg:.4f}. Task 2 F1 micro: {f1_micro:.4f}. Task 3 F1 micro: {f1_micro_binary:.4f}")

    return avg_loss, avg_ndcg, f1_micro, f1_micro_binary

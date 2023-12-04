import torch
import torch.nn as nn
from transformers import AutoModel
from utils.config import config_params as params


class ProductModel(nn.Module):
    """
    The `ProductModel` class is a PyTorch neural network module designed for product-related classification tasks,
    particularly for multi-class classification. It utilizes a pre-trained transformer-based model (such as BERT) as
    its base architecture and adds additional layers for fine-tuning and classification.

    Parameters:
    - checkpoint: str, optional (default=params['checkpoint'])
      - The name or path of the pre-trained transformer model checkpoint to be used as the base architecture.
    - params: dict, optional (default=params)
      - A dictionary containing various hyperparameters and model configuration settings.

    Attributes:
    - bert: AutoModel
      - The pre-trained transformer model (e.g., BERT) loaded from the specified checkpoint.
    - layer_norm: nn.LayerNorm
      - A layer normalization module applied to the output of the transformer.
    - dropout: nn.Dropout
      - A dropout layer applied to the output of the layer normalization.
    - dense: nn.Sequential
      - A sequential neural network module consisting of linear layers and activation functions for classification.

    Methods:
    - forward(input_ids, token_type_ids, attention_mask):
      - The forward method defines the computation performed when passing data through the model.
      - It takes input tensors, including input_ids, token_type_ids, and attention_mask, and computes predictions
        and class probabilities.
      - The input data is processed through the pre-trained transformer, followed by layer normalization, dropout,
        and a fully connected network for classification.
      - The output includes both raw predictions and class probabilities computed using softmax.

    Description:
    - The `ProductModel` class serves as a customizable framework for training and fine-tuning transformer-based models
      for product classification tasks.
    - It leverages a pre-trained model checkpoint as its backbone, allowing for transfer learning from large pre-trained
      language models.
    - Layer normalization and dropout are applied to improve model robustness and prevent overfitting.
    - The final classification layer produces class predictions and probabilities suitable for multi-class classification
      problems.
    - This class can be extended and customized for specific product classification tasks by adjusting the architecture
      and hyperparameters as needed.

    Utilizing the `ProductModel` class can significantly expedite the development and training of deep learning models
    for product-related classification tasks.
    """

    def __init__(self, checkpoint=params['checkpoint'], params=params):
        super(ProductModel, self).__init__()
        self.checkpoint = checkpoint
        self.bert = AutoModel.from_pretrained(checkpoint, return_dict=False)
        self.layer_norm = nn.LayerNorm(params['output_logits'])
        self.dropout = nn.Dropout(params['dropout'])
        self.dense = nn.Sequential(
            nn.Linear(params['output_logits'], 128),
            nn.SiLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(128, 4)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        return preds, torch.softmax(preds, dim=1)

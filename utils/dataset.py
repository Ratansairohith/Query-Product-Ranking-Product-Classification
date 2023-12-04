import torch
from transformers import AutoTokenizer
from utils.config import config_params as params


class ESCIdataset:
    """
    The `ESCIdataset` class is a custom PyTorch dataset designed for use in product classification tasks,
    particularly for classifying product relationships (e.g., 'exact', 'substitute', 'irrelevant', 'complement').
    It prepares and tokenizes input data for model training and inference.

    Parameters:
    - df: DataFrame
      - The input DataFrame containing the dataset to be used for training or evaluation.
    - max_len: int, optional (default=params['max_len'])
      - The maximum length to which input sequences will be padded or truncated during tokenization.
    - checkpoint: str, optional (default=params['checkpoint'])
      - The name or path of the pre-trained transformer model checkpoint to be used for tokenization.

    Attributes:
    - df: DataFrame
      - The dataset as a DataFrame, reset to ensure consistent indexing.
    - max_len: int
      - The maximum sequence length for input data.
    - checkpoint: str
      - The pre-trained model checkpoint used for tokenization.
    - tokenizer: AutoTokenizer
      - An instance of the tokenizer for the specified pre-trained model.
    - num_examples: int
      - The total number of examples in the dataset.

    Methods:
    - __len__():
      - Returns the number of examples in the dataset.

    - __getitem__(idx):
      - Retrieves and prepares a specific example from the dataset.
      - Tokenizes the input text (query, product title, brand, and color name) using the pre-trained tokenizer.
      - Converts the tokenized data into tensors (input IDs, attention mask, and token type IDs) and one-hot encodes
        the target label.
      - Returns a dictionary containing the input tensors and the one-hot encoded label.

    Description:
    - The `ESCIdataset` class is a versatile tool for handling input data preparation in product classification tasks,
      ensuring compatibility with transformer-based models.
    - It initializes with a DataFrame `df`, allowing flexibility in providing training and evaluation data.
    - The `max_len` parameter allows you to control the maximum sequence length when tokenizing input text, which is
      essential for efficient model training.
    - The class uses a pre-trained transformer model's tokenizer to tokenize and convert the input data into tensors
      suitable for model input.
    - It one-hot encodes the target label for multi-class classification, mapping label text to numerical values.
    - This custom dataset class simplifies the process of data preparation, making it easier to work with deep learning
      models for product classification tasks.
    """

    def __init__(self, df, max_len=params['max_len'], checkpoint=params['checkpoint']):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.num_examples = len(self.df)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        query = str(row.query)
        product_title = str(row.product_title)
        product_brand = str(row.product_brand)
        product_color_name = str(row.product_color_name)
        label_text = row.esci_label
        label_map = {'exact': 0, 'substitute': 1, 'irrelevant': 2, 'complement': 3}
        tokenized_text = self.tokenizer(
            query, product_title, product_brand, product_color_name,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids = tokenized_text['input_ids']
        mask = tokenized_text['attention_mask']
        token_type_ids = tokenized_text['token_type_ids']
        label = [0, 0, 0, 0]
        label[label_map[label_text]] = 1

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'target': torch.tensor(label, dtype=torch.float)}

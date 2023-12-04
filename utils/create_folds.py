import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def create_stratifiedgroupkfold(df, splits=5):
    """
    The `create_stratifiedgroupkfold` function is designed to perform Stratified Group K-Fold cross-validation on a given dataset. This type of cross-validation is particularly useful when dealing with datasets where samples are grouped or clustered in some way and you want to ensure that each fold maintains a similar distribution of target variables and respects the grouping of samples.

    Parameters:
    - df: DataFrame
      - The input DataFrame containing the data to be split into folds.
    - splits: int, optional (default=5)
      - The number of splits (folds) to create for cross-validation.

    Returns:
    - df: DataFrame
      - The original DataFrame with an additional 'fold' column that indicates the fold number to which each sample belongs based on Stratified Group K-Fold cross-validation.

    Description:
    - The function takes as input a DataFrame `df` that contains at least three columns: 'query', 'product_title', and 'esci_label'. 'query' and 'product_title' are features used for modeling, and 'esci_label' is the target variable.
    - It also assumes that there is another column 'query_id' that represents the actual query IDs, which are used for grouping the data.
    - The function uses the StratifiedGroupKFold technique, which ensures that each fold has a balanced distribution of the target variable 'esci_label' and respects the grouping of samples based on 'query_id'.
    - For each fold, the function assigns a fold number to each sample in the 'fold' column of the DataFrame, with fold numbers ranging from 1 to the number of splits specified.
    - The modified DataFrame with the 'fold' column is then returned as the result.

    This function is valuable for evaluating machine learning models in a way that accounts for both stratification of target variables and grouping of related samples, making it suitable for scenarios where maintaining these characteristics during cross-validation is essential.
    """
    X = df[['query', 'product_title']]  # Text features
    y = df['esci_label']  # Target variable
    query_id = df['query']  # Actual query IDs
    gkf = StratifiedGroupKFold(n_splits=splits)
    fold_column = np.zeros(len(df), dtype=int)
    for fold, (train_index, val_index) in enumerate(gkf.split(X, y, groups=query_id)):
        fold_column[val_index] = fold + 1
    df['fold'] = fold_column
    return df

import pandas as pd
from utils.cleaning import remove_punc, preprocess, lemmatize_sentence
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb


def vectorize_and_train(train, fold=1):
    """
    The `preprocess_and_train` function takes a DataFrame containing text data and performs a series of preprocessing
    steps followed by training a LightGBM model for multiclass classification.

    Parameters:
    - train: DataFrame
      - The input DataFrame containing text data, where each row represents a text sample.
    - fold: int (optional, default=1)
      - The fold number for cross-validation (used for splitting data into train and test sets).

    Returns:
    - preds: array-like
      - Predicted probabilities for each class obtained from the trained LightGBM model.
    - y_test: array-like
      - True class labels for the test set.

    Description:
    - The `preprocess_and_train` function is designed for text classification tasks and includes the following steps:
      1. Lowercasing: The text in the 'text' column of the input DataFrame is converted to lowercase to ensure consistency.
      2. Punctuation Removal: Punctuation characters are removed from the text using the `remove_punc` function.
      3. Tokenization and Stopword Removal: The text is tokenized and stopwords are removed using the `preprocess` function.
      4. Lemmatization: Lemmatization is applied to reduce words to their base form using the `lemmatize_sentence` function.
      5. Feature Extraction: TF-IDF vectorization is used to convert the preprocessed text into numerical features.
      6. Model Training: A LightGBM model is trained for multiclass classification using the specified parameters.
      7. Prediction: The trained model is used to make predictions on the test set, and the predicted probabilities
         are returned.
      8. True Labels: The true class labels for the test set are also returned.

    - The function is suitable for tasks where the goal is to classify text samples into one of multiple classes.
    - The `fold` parameter can be used to perform cross-validation by splitting the data into training and test sets
      based on the specified fold.
    """

    params = {
        'objective': 'multiclass',
        'num_classes': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    num_round = 100

    X_train = train[train['fold'] != fold]
    X_test = train[train['fold'] == fold]
    y_train = X_train['esci_label']
    y_test = X_test['esci_label']
    # Initialize the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=20000)
    tfidf_train_matrix = tfidf_vectorizer.fit_transform(X_train['text'])
    tfidf_train = pd.DataFrame(tfidf_train_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
    y_train = y_train.map({'exact': 0, 'substitute': 1, 'irrelevant': 2, 'complement': 3})
    train_data = lgb.Dataset(tfidf_train, label=y_train)
    tfidf_test_matrix = tfidf_vectorizer.transform(X_test['text'])
    tfidf_test = pd.DataFrame(tfidf_test_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
    y_test = y_test.map({'exact': 0, 'substitute': 1, 'irrelevant': 2, 'complement': 3})
    model = lgb.train(params, train_data, num_round)
    preds = model.predict(tfidf_test)
    return preds, y_test


if __name__ == "__main__":
    # Define your params and load your train_df here
    train_df = pd.read_csv('../data/training_set_fraction.csv')
    train_df.fillna('',inplace=True)
    train_df.drop(['query_id', 'product_description'], axis=1, inplace=True)
    train_df.sort_values(['fold', 'product_id'], inplace=True, ignore_index=True)
    train_df.reset_index(inplace=True, drop=True)
    train_df = train_df[['query','product_title', 'product_brand', 'product_color_name', 'fold', 'esci_label']]
    train_df['text'] = train_df['query'] + ' ' + train_df['product_title'] + ' ' + train_df['product_brand']
    train_df.drop(['query', 'product_title', 'product_color_name'], axis=1, inplace=True)
    # Preprocessing
    train_df['text'] = train_df['text'].str.lower()
    train_df['text'] = train_df['text'].apply(remove_punc)
    train_df['text'] = train_df['text'].apply(preprocess)
    train_df['text'] = train_df['text'].apply(lambda x: " ".join(x))
    train_df['text'] = train_df['text'].apply(lemmatize_sentence)

    for i in range(5):
        print(f'[Info] Training started for Fold {i}')
        probabilities, y_true = vectorize_and_train(train_df, i)


import gensim
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()


def remove_punc(message):
    """
    The `remove_punc` function takes a text message as input and removes any punctuation characters from it.

    Parameters:
    - message: str
      - The input text message containing punctuation that needs to be removed.

    Returns:
    - punc_removed_join: str
      - The processed text with all punctuation characters removed.

    Description:
    - The `remove_punc` function is a text preprocessing step commonly used to clean and simplify text data.
    - It takes a text message as input and iterates through each character in the message.
    - For each character, it checks if it belongs to the set of punctuation characters using the `string.punctuation`
      module.
    - Punctuation characters, such as periods, commas, and exclamation marks, are excluded from the result.
    - The function then joins the remaining characters together to form a cleaned text without punctuation.
    - The cleaned text, free of punctuation, is returned as the output.
    """

    punc_removed = [char for char in message if char not in string.punctuation]
    punc_removed_join = ''.join(punc_removed)
    return punc_removed_join


def preprocess(text):
    """
    The `preprocess` function is designed to preprocess a text by tokenizing it and filtering out stopwords and short tokens.

    Parameters:
    - text: str
      - The input text to be preprocessed.

    Returns:
    - result: list of str
      - A list of preprocessed tokens extracted from the input text, after removing stopwords and short tokens.

    Description:
    - The `preprocess` function is a common step in text data preprocessing used to prepare text for natural language
      processing tasks.
    - It takes an input text and tokenizes it using Gensim's `simple_preprocess` function, which splits the text into
      lowercase tokens.
    - For each token in the text, the function checks its length and filters out tokens that are shorter than a
      specified length threshold (commonly 2 characters).
    - Additionally, it filters out tokens that are found in a predefined set of stopwords to remove common,
      non-informative words.
    - The preprocessed tokens are collected in a list and returned as the result.
    - The `preprocess` function is useful for reducing noise and preparing text data for tasks such as text classification
      or topic modeling.
    """

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) > 2 and token not in stop_words:
            result.append(token)
    return result


def lemmatize_sentence(sentence):
    """
    The `lemmatize_sentence` function takes an input sentence and applies lemmatization to its individual words,
     returning the lemmatized sentence.

    Parameters:
    - sentence: str
      - The input sentence to be lemmatized.

    Returns:
    - lemmatized_sentence: str
      - The sentence with its words lemmatized and joined back into a single string.

    Description:
    - Lemmatization is a text preprocessing technique used to reduce words to their base or root form, ensuring consistent
      word representation.
    - The `lemmatize_sentence` function tokenizes the input sentence into individual words using the `word_tokenize`
      function.
    - It then applies lemmatization to each word using a lemmatizer (presumably defined elsewhere in the code).
    - The lemmatized words are collected and joined back together into a single string, forming the lemmatized sentence.
    - Lemmatization helps in text normalization and can improve the quality of text data for various natural language
      processing tasks.
    - This function is particularly useful for reducing the vocabulary size and capturing the essential meaning of words.
    """

    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

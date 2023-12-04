from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


def translate_to_english(df, lang, col, quantize=True):
    """
    The `translate_to_english` function is designed to translate text data from a specified language to English using
    the MBart model for conditional generation. This function is particularly useful when dealing with multilingual
    text data and you want to perform language translation tasks.

    Parameters:
    - df: DataFrame
      - The input DataFrame containing the text data to be translated.
    - lang: str
      - A string representing the source language code. It can be either 'jp' for Japanese or 'es' for Spanish.
    - col: str
      - The name of the DataFrame column containing the text to be translated.
    - quantize: bool, optional (default=True)
      - A flag indicating whether to use quantization for model optimization. If set to True, the model will be loaded
      with quantization configuration, which can reduce memory usage.

    Returns:
    - translated: list
      - A list of translated text corresponding to the input text in English.

    Description:
    - The function first configures the MBart model for conditional generation. It allows for translation from multiple
      source languages to English.
    - If the `quantize` flag is set to True, the function loads the model with quantization configuration, which can
      optimize memory usage at the cost of some inference speed.
    - The function identifies the source language based on the 'lang' parameter ('jp' for Japanese or 'es' for Spanish)
      and extracts unique text data from the DataFrame for translation.
    - It then uses the MBart model and tokenizer to translate each text entry to English.
    - The translated text is stored in a list, and the list is returned as the result.

    This function is valuable for automating the translation of text data from various source languages to English,
    making it easier to work with multilingual datasets and conduct analysis or modeling in a common language.
    """

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

    if quantize:
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                              quantization_config=nf4_config)
    else:
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    print(f"Loaded MBart model footprint is {model.get_memory_footprint()}")

    if lang == 'jp':
        tokenizer.src_lang = "ja_XX"
        text = df[df['query_locale'] == lang][col].unique()
    else:
        tokenizer.src_lang = "es_XX"
        text = df[df['query_locale'] == lang][col].unique()

    translated = []

    for i in tqdm(range(len(text)), desc="Processing"):
        encoded = tokenizer(text[i], return_tensors="pt")
        encoded = encoded.to("cuda:0")
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
        translated.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])

    return translated

from tokenizers import Tokenizer


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    """
    Count the number of tokens in the given text using the provided tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer (Tokenizer): The tokenizer to use for encoding the text.

    Returns:
        int: The number of tokens in the text.
    """
    encoded = tokenizer.encode(text)
    return len(encoded.ids)


from typing import List, Optional
from nltk import sent_tokenize


def sentence_tokenize(text: str, max_pos: Optional[int] = None) -> (List, List):
    """
    Split text into sentences and save info about delimiters between them to restore linebreaks,
    whitespaces, etc.
    """
    delimiters = []
    sentences = [sent.strip() for sent in sent_tokenize(text)]

    tokens = []
    if max_pos is not None:
        for i, sentence in enumerate(sentences):
            while sentence:
                tokens.append(sentence[:max_pos])
                sentence = sentence[max_pos:]
    else:
        tokens = sentences

    if len(tokens) == 0:
        return [''], ['']
    else:
        try:
            for sentence in tokens:
                idx = text.index(sentence)
                delimiters.append(text[:idx])
                text = text[idx + len(sentence):]
            delimiters.append(text)
        except ValueError:
            delimiters = ['', *[' ' for _ in range(len(tokens) - 1)], '']

    return tokens, delimiters

"""
Fallback preprocessing method to meet the bare minimum of tagged translation constraints.
All tags are removed and appended to the translation.
"""
import re
from typing import List

tag_patterns = {
    'document': r'</?(?:x|g|bx|ex)[^>]+>',
    'web': r'</?(?:a|abbr|acronym|em|strong|b|i|s|strike|u|span|del|ins|sub|sup|code|samp|kbd|var|small|mark|ruby|rt'
           r'|rp|bdi|bdo)[0-9]+>'
}


def _preprocess_tags(sentences: List[str], input_type: str) -> (List[str], List[str]):
    if input_type in tag_patterns:
        pattern = tag_patterns[input_type]
        tags = [''.join(re.findall(pattern, sentence)) for sentence in sentences]
        sentences = [re.sub(pattern, '', sentence) for sentence in sentences]
        sentences = [re.sub(r'(\s)+', r'\1', sentence).strip() for sentence in sentences]
    else:
        tags = ['' for _ in sentences]

    return sentences, tags


def _postprocess_tags(translations: List[str], tags: List[str]):
    translations = [f'{translation}{tag}' for tag, translation in zip(translations, tags)]

    return translations

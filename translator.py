import logging
from typing import List

logger = logging.getLogger("nmt-worker")


class Translator:
    def __init__(self):
        # TODO: load model
        pass

    def translate(self, sentences: List[str], src: str, tgt: str, domain: str, input_type: str = "plain") -> List[str]:
        # TODO
        pass

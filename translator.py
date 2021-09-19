import logging
from typing import List
from fairseq.models.transformer import TransformerModel


logger = logging.getLogger("nmt-worker")


class Translator:
    def __init__(self, checkpoint_path, spm_model):
        self.model = TransformerModel.from_pretrained(
            checkpoint_path,
            checkpoint_file='checkpoint_best.pt',
            bpe='sentencepiece',
            sentencepiece_model=spm_model
        )

    def translate(self, sentences: List[str], src: str, tgt: str, domain: str, input_type: str = "plain") -> List[str]:
        return self.model.translate(sentences)

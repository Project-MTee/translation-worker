import logging
from typing import List
from fairseq.models.transformer import TransformerModel


logger = logging.getLogger("nmt-worker")


class Translator:
    def __init__(self, checkpoint_path, dict_path, spm_model_path):
        self.model = TransformerModel.from_pretrained(
            checkpoint_path,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=dict_path,
            bpe='sentencepiece',
            sentencepiece_model=spm_model_path
        )

    def translate(self, sentences: List[str], src: str, tgt: str, domain: str, input_type: str = "plain") -> List[str]:
        return self.model.translate(sentences)

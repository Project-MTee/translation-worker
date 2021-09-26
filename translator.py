import logging
from typing import List
from fairseq.models.transformer import TransformerModel

from tag_utils import preprocess_tags, postprocess_tags

logger = logging.getLogger("nmt_worker")


class Translator:
    def __init__(self, checkpoint_path: str, spm_model: str):
        self.model = TransformerModel.from_pretrained(
            checkpoint_path,
            checkpoint_file='checkpoint_best.pt',
            bpe='sentencepiece',
            sentencepiece_model=spm_model
        )

    def translate(self, sentences: List[str], src: str, tgt: str, domain: str, input_type: str = "plain") -> List[str]:
        detagged, tags = preprocess_tags(sentences, input_type)

        # Translations of empty source sentences are deleted, because NMT may hallucinate
        translations = [translation if detagged[idx] != '' else '' for idx, translation in
                        enumerate(self.model.translate(detagged))]

        retagged = postprocess_tags(translations, tags, input_type)

        return retagged

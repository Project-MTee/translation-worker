import os

import itertools
import logging
import warnings
from typing import List

from .config import ModelConfig
from .schemas import Response, Request
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize
from .tokenization import sentence_tokenize

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:
    model = None

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._load_model()

        if model_config.modular:
            self.translate = self._translate_modular
        else:
            self.translate = self._translate_unidirectional

        logger.info("All NMT models loaded")

    def _load_model(self):
        sentencepiece_path = os.path.join(self.model_config.sentencepiece_dir, self.model_config.sentencepiece_prefix)
        if self.model_config.modular:
            from .modular_interface import ModularHubInterface
            self.model = ModularHubInterface.from_pretrained(
                model_path=self.model_config.checkpoint_path,
                sentencepiece_prefix=sentencepiece_path,
                dictionary_path=self.model_config.dict_dir)
        else:
            from fairseq.models.transformer import TransformerModel
            self.model = TransformerModel.from_pretrained(
                "./",
                checkpoint_file=self.model_config.checkpoint_path,
                bpe='sentencepiece',
                sentencepiece_model=f"{sentencepiece_path}.model",
                data_name_or_path=self.model_config.dict_dir
            )

    def _translate_unidirectional(
            self, sentences: List[str], request: Request, tags: List, tags_exist: bool) -> List[str]:
        translated = [translation if sentences[idx] != '' else '' for idx, translation in enumerate(
            self.model.translate(sentences))]
        if tags_exist:
            translated = postprocess_tags(translated, tags, request.input_type)
        return translated

    def _translate_modular(self, sentences: List[str], request: Request, tags: List, tags_exist: bool) -> List[str]:
        if tags_exist:
            translated, alignments = self.model.translate_align(sentences, src_language=request.src,
                                                                tgt_language=request.tgt,
                                                                replace_unks=True)
            # TODO
            return translated
        else:
            return [translation if sentences[idx] != '' else '' for idx, translation in enumerate(
                self.model.translate(sentences, src_language=request.src, tgt_language=request.tgt))]

    def process_request(self, request: Request) -> Response:
        inputs = [request.text] if type(request.text) == str else request.text
        translations = []

        for text in inputs:
            sentences, delimiters = sentence_tokenize(text)
            detagged, tags, tags_exist = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            translated = self.translate(normalized, request, tags, tags_exist)
            translations.append(
                ''.join(itertools.chain.from_iterable(zip(delimiters, translated))) + delimiters[-1])

            response = Response(translation=translations[0] if type(request.text) == str else translations)

        return response

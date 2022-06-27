import itertools
import logging
from typing import List
import warnings

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

        try:
            self._load_model()
        except OSError as e:
            logger.exception("Model loading failed. Trying to download model...")
            self.model_config.download()
            self._load_model()

        if model_config.modular:
            self.translate = self._translate_modular
        else:
            self.translate = self._translate

        logger.info(f"All NMT models loaded; "
                    f"language pairs: {self.model_config.language_pairs}; "
                    f"domains: {self.model_config.domains}")

    def _load_model(self):
        if self.model_config.modular:
            from .modular_interface import ModularHubInterface
            self.model = ModularHubInterface.from_pretrained(
                model_path=self.model_config.checkpoint,
                sentencepiece_prefix=self.model_config.sentencepiece_prefix,
                dictionary_path=self.model_config.dict_dir)
        else:
            from fairseq.models.transformer import TransformerModel
            self.model = TransformerModel.from_pretrained(
                "./",
                checkpoint_file=self.model_config.checkpoint,
                bpe='sentencepiece',
                sentencepiece_model=f"{self.model_config.sentencepiece_prefix}.model",
                data_name_or_path=self.model_config.dict_dir
            )

    def _translate(self, sentences: List[str], **_) -> List[str]:
        return self.model.translate(sentences)

    def _translate_modular(self, sentences: List[str], src: str, tgt: str, **_) -> List[str]:
        return self.model.translate(sentences, src_language=src, tgt_language=tgt)

    def process_request(self, request: Request) -> Response:
        inputs = [request.text] if type(request.text) == str else request.text
        translations = []

        for text in inputs:
            sentences, delimiters = sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            translated = [translation if normalized[idx] != '' else '' for idx, translation in enumerate(
                self.translate(normalized, src=request.src, tgt=request.tgt, domain=request.domain))]
            retagged = postprocess_tags(translated, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])

        response = Response(translation=translations[0] if type(request.text) == str else translations)

        return response

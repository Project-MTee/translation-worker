import itertools
import logging
from typing import List
import warnings

from nltk import sent_tokenize
from .utils import Response, Request
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:import itertools
import logging
import warnings
from typing import List, Tuple

from .config import ModelConfig
from .schemas import Response, Request
from .tag_utils import preprocess_tags, postprocess_tags, postprocess_tags_with_alignment
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
            self.translate = self._translate

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

    def _translate(self, sentences: List[str], **_) -> List[str]:
        return self.model.translate(sentences)

    def _translate_modular(self, sentences: List[str], src: str, tgt: str, **_) -> List[str]:
        return self.model.translate(sentences, src_language=src, tgt_language=tgt)

    def _translate_modular_with_align(self, sentences: List[str], src: str, tgt: str, **_) \
            -> Tuple[List[str], List[List[Tuple[int, int]]]]:
        return self.model.translate_align(sentences, src_language=src, tgt_language=tgt, replace_unks=True)

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

    model = None

    def __init__(self, modular: bool,
                 checkpoint_path: str, dict_dir: str, sentencepiece_dir, sentencepiece_prefix: str):
        if modular:
            self._load_modular_model(checkpoint_path, dict_dir, sentencepiece_dir, sentencepiece_prefix)
            self.translate = self._translate_modular
        else:
            self._load_model(checkpoint_path, dict_dir, sentencepiece_dir, sentencepiece_prefix)
            self.translate = self._translate
        logger.info("All NMT models loaded")

    @staticmethod
    def _sentence_tokenize(text: str) -> (List, List):
        """
        Split text into sentences and save info about delimiters between them to restore linebreaks,
        whitespaces, etc.
        """
        delimiters = []
        sentences = [sent.strip() for sent in sent_tokenize(text)]
        if len(sentences) == 0:
            return [''], ['']
        else:
            try:
                for sentence in sentences:
                    idx = text.index(sentence)
                    delimiters.append(text[:idx])
                    text = text[idx + len(sentence):]
                delimiters.append(text)
            except ValueError:
                delimiters = ['', *[' ' for _ in range(len(sentences) - 1)], '']

        return sentences, delimiters

    def _load_model(self, checkpoint_path: str, dict_dir: str, sentencepiece_dir, sentencepiece_prefix: str):
        from fairseq.models.transformer import TransformerModel
        sentencepiece_dir = sentencepiece_dir.rstrip('/')
        self.model = TransformerModel.from_pretrained(
            "./",
            checkpoint_file=checkpoint_path,
            bpe='sentencepiece',
            sentencepiece_model=f"{sentencepiece_dir}/{sentencepiece_prefix}.model",
            data_name_or_path=dict_dir
        )

    def _load_modular_model(self, checkpoint_path: str, dict_dir: str, sentencepiece_dir: str,
                            sentencepiece_prefix: str):
        from .modular_interface import ModularHubInterface
        sentencepiece_dir = sentencepiece_dir.rstrip('/')
        self.model = ModularHubInterface.from_pretrained(
            model_path=checkpoint_path,
            sentencepiece_prefix=f"{sentencepiece_dir}/{sentencepiece_prefix}",
            dictionary_path=dict_dir)

    def _translate(self, sentences: List[str], **_) -> List[str]:
        return self.model.translate(sentences)

    def _translate_modular(self, sentences: List[str], src: str, tgt: str, **_) -> List[str]:
        return self.model.translate(sentences, src_language=src, tgt_language=tgt)

    def process_request(self, request: Request) -> Response:
        inputs = [request.text] if type(request.text) == str else request.text
        translations = []

        for text in inputs:
            sentences, delimiters = self._sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            translated = [translation if normalized[idx] != '' else '' for idx, translation in enumerate(
                self.translate(normalized, src=request.src, tgt=request.tgt, domain=request.domain))]
            retagged = postprocess_tags(translated, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])

        response = Response(translation=translations[0] if type(request.text) == str else translations)

        return response

import itertools
import logging
import warnings
from typing import List, Tuple

from nltk import sent_tokenize
from .utils import Response, Request
from .normalization import normalize
from .tag_utils import preprocess_tags, postprocess_tags, postprocess_tags_with_alignment

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:
    model = None

    def __init__(self, modular: bool, checkpoint_path: str, dict_dir: str, sentencepiece_dir,
                 sentencepiece_prefix: str, with_alignment: bool = False):
        if modular:
            self._load_modular_model(checkpoint_path, dict_dir, sentencepiece_dir, sentencepiece_prefix,
                                     with_alignment=with_alignment)
            self.translate = self._translate_modular
            if with_alignment:
                self.translate_align = self._translate_modular_with_align
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
                            sentencepiece_prefix: str, with_alignment: bool):
        if with_alignment:
            from .modular_interface import ModularHubInterfaceWithAlignment as ModularModel
        else:
            from .modular_interface import ModularHubInterface as ModularModel
        sentencepiece_dir = sentencepiece_dir.rstrip('/')
        self.model = ModularModel.from_pretrained(
            model_path=checkpoint_path,
            sentencepiece_prefix=f"{sentencepiece_dir}/{sentencepiece_prefix}",
            dictionary_path=dict_dir)

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
            sentences, delimiters = self._sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            if len(tags[0]) > 1:
                hyps_aligned, alignments = self.translate_align(normalized, src=request.src, tgt=request.tgt,
                                                                domain=request.domain)
                translated = [translation if normalized[idx] != '' else '' for idx, translation in
                              enumerate(hyps_aligned)]
                retagged = postprocess_tags_with_alignment(normalized, translated, tags, request.input_type, alignments)
                translations.append(retagged)
            else:
                translated = [translation if normalized[idx] != '' else '' for idx, translation in enumerate(
                    self.translate(normalized, src=request.src, tgt=request.tgt, domain=request.domain))]
                retagged = postprocess_tags(translated, tags, request.input_type)
                translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])

        response = Response(translation=translations[0] if type(request.text) == str else translations)

        return response

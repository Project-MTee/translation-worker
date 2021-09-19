import itertools
import logging
from typing import Optional, List, Union

from nltk import sent_tokenize
from helpers import Response, Request

import settings
from translator import Translator

logger = logging.getLogger("nmt_worker")


class TranslationWorker:
    def __init__(self, nmt_path, spm_model):
        self.translator = Translator(
            checkpoint_path=nmt_path,
            spm_model=spm_model
        )
        logger.info("All NMT models loaded")


    @staticmethod
    def _sentence_tokenize(text: Union[str, List]) -> (List, Optional[List]):
        """
        Split text into sentences and save info about delimiters between them to restore linebreaks,
        whitespaces, etc.
        """
        delimiters = None
        if type(text) == str:
            sentences = [sent.strip() for sent in sent_tokenize(text)]
            try:
                delimiters = []
                for sentence in sentences:
                    idx = text.index(sentence)
                    delimiters.append(text[:idx])
                    text = text[idx + len(sentence):]
                delimiters.append(text)
            except ValueError:
                delimiters = ['', *[' ' for _ in range(len(sentences) - 1)], '']
        else:
            sentences = [sent.strip() for sent in text]

        return sentences, delimiters

    def process_request(self, request: Request) -> Response:

        sentences, delimiters = self._sentence_tokenize(request.text)
        length = sum([len(sent) for sent in sentences])

        if length == 0:
            # No translatable content in input
            if type(request.text) == str:
                return Response(translation="")
            else:
                return Response(translation=[])

        else:
            src_lang = request.src
            tgt_lang = request.tgt

            translations = self.translator.translate(sentences,
                                                     src=src_lang,
                                                     tgt=tgt_lang,
                                                     domain=request.text,
                                                     input_type=request.input_type)
            if delimiters:
                translations = ''.join(itertools.chain.from_iterable(zip(delimiters, translations))) + delimiters[-1]

            return Response(translation=translations)


if __name__ == "__main__":
    from mq_consumer import MQConsumer

    worker = TranslationWorker(**settings.WORKER_PARAMETERS)
    consumer = MQConsumer(worker=worker,
                          connection_parameters=settings.MQ_PARAMETERS,
                          exchange_name=settings.EXCHANGE_NAME,
                          routing_keys=settings.ROUTING_KEYS)

    consumer.start()

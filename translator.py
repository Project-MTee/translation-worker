import logging
from typing import List
from fairseq.models.transformer import TransformerModel

from tag_utils import preprocess_tags, postprocess_tags

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
        detagged, tags = preprocess_tags(sentences, input_type)

        # Translations of empty source sentences are deleted, because NMT may hallucinate
        translations = [translation if detagged[idx] != '' else '' for idx, translation in
                        enumerate(self.model.translate(detagged))]

        retagged = postprocess_tags(translations, tags)

        return retagged


if __name__ == "__main__":
    # File translation mode
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser()
    parser.add_argument('checkpoint', help="checkpoint folder path")
    parser.add_argument('spm', type=FileType('r'), help="sentencepiece .model file path")
    parser.add_argument('input', type=FileType('r'), help="input text file")
    parser.add_argument('output', type=FileType('w'), help="output text file")
    args = parser.parse_args()

    translator = Translator(args.checkpoint, args.spm.name)

    in_file = open(args.input.name, 'r', encoding='utf-8')
    out_file = open(args.output.name, 'w', encoding='utf-8')

    while True:
        text = in_file.readline()
        if not text:
            break
        out_file.write(translator.translate([text], 'en', 'et', 'general', 'test')[0]+'\n')

    in_file.close()
    out_file.close()

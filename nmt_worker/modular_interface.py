from __future__ import annotations
import itertools
import logging
import copy
from collections import OrderedDict
from typing import Dict, List, Iterator, Any, Optional, Tuple, Iterable, Set, Union

from fairseq.data import Dictionary, LanguagePairDataset, FairseqDataset
from fairseq import utils, search, hub_utils
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment
from fairseq.utils import replace_unk

from omegaconf import open_dict, DictConfig

from sentencepiece import SentencePieceProcessor

import torch
from torch import Tensor, LongTensor
from torch.nn import ModuleList, Module

logger = logging.getLogger(__name__)


class ModularHubInterface(Module):
    def __init__(
            self,
            models: List[MultilingualTransformerModel],
            task: MultilingualTranslationTask,
            cfg: DictConfig,
            sp_models: Dict[str, SentencePieceProcessor]
    ):
        super().__init__()

        self.sp_models = sp_models
        self.models = ModuleList(models)
        self.task = task
        self.cfg = cfg
        self.dicts: Dict[str, Dictionary] = task.dicts
        self.langs = task.langs

        for model in self.models:
            model.prepare_for_inference_(self.cfg)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @classmethod
    def from_pretrained(
            cls,
            model_path: str,
            sentencepiece_prefix: str,
            dictionary_path: str,
            override_args: Optional[Dict[str, Any]] = None
    ) -> ModularHubInterface:
        """
        @param model_path: path to the model checkpoint file
        @param sentencepiece_prefix: prefix so that the sp model is located at {sentencepiece_prefix}.{lang}.model
        @param dictionary_path: path to the directory with dict.{lang}.txt files
        @param override_args: model state arguments to override
        @return: ModularHubInterface instance
        """
        x = hub_utils.from_pretrained(
            "./",
            checkpoint_file=model_path,
            archive_map={},
            data_name_or_path=dictionary_path,
            task="multilingual_translation",
            **({} if override_args is None else override_args)
        )

        sp_models = {
            lang: SentencePieceProcessor(
                model_file=f"{sentencepiece_prefix}.{lang}.model"
            ) for lang in x["task"].langs
        }

        return cls(
            models=x["models"],
            task=x["task"],
            cfg=x["args"],
            sp_models=sp_models,
        )

    @property
    def device(self):
        return self._float_tensor.device

    def binarize(self, sentence: str, language: str) -> LongTensor:
        return self.dicts[language].encode_line(sentence, add_if_not_exist=False).long()

    def apply_bpe(self, sentence: str, language: str) -> str:
        return " ".join(self.sp_models[language].encode(sentence, out_type=str))

    def string(self, tokens: Tensor, language: str) -> str:
        return self.dicts[language].string(tokens)

    @staticmethod
    def remove_bpe(sentence: str) -> str:
        return sentence.replace(" ", "").replace("\u2581", " ").strip()

    def encode(self, sentence: str, language: str) -> LongTensor:
        bpe_token_sent = self.apply_bpe(sentence, language)
        logger.debug(f"Preprocessed: {sentence} into {bpe_token_sent}.")
        return self.binarize(bpe_token_sent, language)

    def decode(self, tokens: Tensor, language: str) -> str:
        bpe_token_sent = self.string(tokens, language)
        decoded_sent = self.remove_bpe(bpe_token_sent)
        logger.debug(f"Postprocessed: {bpe_token_sent} into {decoded_sent}.")
        return decoded_sent

    def translate(
            self,
            sentences: List[str],
            src_language: str,
            tgt_language: str,
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = 1000,
    ) -> List[str]:
        """
        :param sentences: list of sentences to be translated
        :param src_language: source language
        :param tgt_language: target language
        :param beam: beam size for the beam search algorithm (decoding)
        :param max_sentences: max number of sentences in each batch
        :param max_tokens: max number of tokens in each batch, all sentences must be shorter than max_tokens.
        :return: list of translations corresponding to the input sentences
        """
        logger.debug(f"Translating from {src_language} to {tgt_language}")
        tokenized_sentences = [self.encode(sentence, src_language) for sentence in sentences]
        batched_hypos = self._generate(
            tokenized_sentences,
            src_language,
            tgt_language,
            beam=beam,
            max_sentences=max_sentences,
            max_tokens=max_tokens
        )
        return [self.decode(hypos[0]["tokens"], tgt_language) for hypos in batched_hypos]

    def _generate(
            self,
            tokenized_sentences: List[LongTensor],
            src_lang: str,
            tgt_lang: str,
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = None,
            skip_invalid_size_inputs=False,
            **kwargs
    ) -> List[List[Dict[str, Union[Tensor, List]]]]:
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self._build_generator(src_lang, tgt_lang, gen_args)

        results = []
        for batch in self._build_batches(
                tokenized_sentences,
                src_lang,
                tgt_lang,
                skip_invalid_size_inputs=skip_invalid_size_inputs,
                max_sentences=max_sentences,
                max_tokens=max_tokens
        ):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        return outputs

    def _build_dataset_for_inference(
            self, src_tokens: List[LongTensor],
            src_lengths: LongTensor,
            src_lang: str,
            tgt_lang: str,
    ) -> FairseqDataset:
        return self.task.alter_dataset_langtok(
            LanguagePairDataset(
                src_tokens, src_lengths, self.dicts[src_lang]
            ),
            src_eos=self.dicts[src_lang].eos(),
            src_lang=src_lang,
            tgt_eos=self.dicts[tgt_lang].eos(),
            tgt_lang=tgt_lang,
        )

    def _build_batches(
            self,
            tokens: List[LongTensor],
            src_lang: str,
            tgt_lang: str,
            skip_invalid_size_inputs: bool,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        lengths = LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self._build_dataset_for_inference(tokens, lengths, src_lang, tgt_lang),
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=self.max_positions[f"{src_lang}-{tgt_lang}"],
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def _build_generator(self, src_lang, tgt_lang, args):
        return SequenceGenerator(
            ModuleList([model.models[f"{src_lang}-{tgt_lang}"] for model in self.models]),
            self.dicts[tgt_lang],
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(self.dicts[tgt_lang]),
        )


class ModularHubInterfaceWithAlignment(ModularHubInterface):
    def __init__(
            self,
            models: List[MultilingualTransformerModel],
            task: MultilingualTranslationTask,
            cfg: DictConfig,
            sp_models: Dict[str, SentencePieceProcessor]
    ):
        from fairseq.models.multilingual_transformer_align import MultilingualTransformerAlignModel
        if not all(isinstance(model, MultilingualTransformerAlignModel) for model in models):
            raise ValueError(f"All models must be instances of {MultilingualTransformerAlignModel.__name__}")

        super().__init__(models, task, cfg, sp_models)

    @classmethod
    def from_pretrained_multilingual_transformer(
            cls,
            model_path: str,
            sentencepiece_prefix: str,
            dictionary_path: str,
            alignment_heads=8,
            alignment_layer=2,
            full_context_alignment=False
    ) -> ModularHubInterfaceWithAlignment:
        """
        Loads the MultilingualTransformer model as MultilingualTransformerAlign and adds alignment parameters.

        @param model_path: path to the model checkpoint file
        @param sentencepiece_prefix: prefix so that the sp model is located at {sentencepiece_prefix}.{lang}.model
        @param dictionary_path: path to the directory with dict.{lang}.txt files
        @param alignment_heads: the number of heads to use for the alignment
        @param alignment_layer: layer to extract alignments from (indexing from 0)
        @param full_context_alignment: use the full context (without triangular mask) for alignments
        @return: ModularHubInterfaceWithAlignment instance
        """
        return cls.from_pretrained(
            model_path,
            sentencepiece_prefix,
            dictionary_path,
            override_args={
                "arch": "multilingual_transformer_align",
                "alignment_heads": alignment_heads,
                "alignment_layer": alignment_layer,
                "full_context_alignment": full_context_alignment,
                "_name": None
            }
        )

    @staticmethod
    def bpe_to_word_alignment(
            src_bpe_sent: str,
            tgt_bpe_sent: str,
            bpe_alignment: List[Tuple[int, int]],
            alignment_ignore_tokens: Set[str]
    ) -> List[Tuple[int, int]]:
        def bpe_to_word_map(tokens):
            return [x - 1 for x in itertools.accumulate([int("\u2581" in x) for x in tokens])]

        src_tokens = src_bpe_sent.split(" ")
        tgt_tokens = tgt_bpe_sent.split(" ")

        src_map = bpe_to_word_map(src_tokens)
        tgt_map = bpe_to_word_map(tgt_tokens)

        word_alignments = (
            (src_map[a], tgt_map[b]) for a, b in bpe_alignment
            if src_tokens[a] not in alignment_ignore_tokens and tgt_tokens[b] not in alignment_ignore_tokens
        )

        # removing duplicates
        return list(OrderedDict.fromkeys(word_alignments))

    def _replace_unks_with_alignment(self, src_bpe_sent, tgt_bpe_sent, alignments, tgt_lang):
        return replace_unk(
            tgt_bpe_sent,
            src_bpe_sent,
            [align[0] for align in alignments],
            {},
            self.dicts[tgt_lang].unk_string()
        )

    def translate_align_bpe(
            self,
            bpe_sentences: List[str],
            src_language: str,
            tgt_language: str,
            alignment: str = "hard_shifted",
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = 1000,
            replace_unks: bool = False
    ) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
        """
        :param bpe_sentences: list of bpe encoded sentences to be translated
        :param src_language: source language
        :param tgt_language: target language
        :param alignment: method for generating alignments (recommended values are "hard" or "hard_shifted")
        :param beam: beam size for the beam search algorithm (decoding)
        :param max_sentences: max number of sentences in each batch
        :param max_tokens: max number of tokens in each batch, all sentences must be shorter than max_tokens.
        :param replace_unks: replace <unk>-s with the aligned src token.
        :return: list of bpe translations corresponding to the input sentences and list of alignments (src_idx, tgt_idx)
        """
        logger.debug(f"Translating from {src_language} to {tgt_language}")

        batched_hypos = self._generate(
            [self.binarize(sentence, src_language) for sentence in bpe_sentences],
            src_language,
            tgt_language,
            beam=beam,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            print_alignment=alignment
        )

        tgt_bpe_sents = [self.string(hypos[0]["tokens"], tgt_language) for hypos in batched_hypos]
        alignments = [hypos[0]["alignment"] for hypos in batched_hypos]

        if replace_unks:
            tgt_bpe_sents = [
                self._replace_unks_with_alignment(src, tgt, align, tgt_language)
                for src, tgt, align in zip(bpe_sentences, tgt_bpe_sents, alignments)
            ]

        return tgt_bpe_sents, alignments

    def translate_align(
            self,
            sentences: List[str],
            src_language: str,
            tgt_language: str,
            alignment: str = "hard_shifted",
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = 1000,
            alignment_ignore_tokens: Iterable[str] = tuple(),
            replace_unks: bool = False
    ) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
        """
        :param sentences: list of sentences to be translated
        :param src_language: source language
        :param tgt_language: target language
        :param alignment: method for generating alignments (recommended values are "hard" or "hard_shifted")
        :param beam: beam size for the beam search algorithm (decoding)
        :param max_sentences: max number of sentences in each batch
        :param max_tokens: max number of tokens in each batch, all sentences must be shorter than max_tokens.
        :param alignment_ignore_tokens: tokens to ignore when converting bpe alignments to word alignments.
        :param replace_unks: replace <unk>-s with the aligned src token.
        :return: list of translations corresponding to the input sentences and list of word alignments (src_idx, tgt_idx)
        """
        src_bpe_sents = [self.apply_bpe(sentence, src_language) for sentence in sentences]

        tgt_bpe_sents, alignments = self.translate_align_bpe(
            src_bpe_sents,
            src_language=src_language,
            tgt_language=tgt_language,
            alignment=alignment,
            beam=beam,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            replace_unks=replace_unks
        )

        tgt_sents = [self.remove_bpe(tgt_sent) for tgt_sent in tgt_bpe_sents]
        word_alignments = [
            self.bpe_to_word_alignment(src, tgt, align, set(alignment_ignore_tokens))
            for src, tgt, align in zip(src_bpe_sents, tgt_bpe_sents, alignments)
        ]

        return tgt_sents, word_alignments

    def _build_generator(self, src_lang, tgt_lang, args):
        print_alignment = getattr(args, "print_alignment", "hard_shifted")
        if print_alignment is None:
            return super()._build_generator(src_lang, tgt_lang, args)

        return SequenceGeneratorWithAlignment(
            ModuleList([model.models[f"{src_lang}-{tgt_lang}"] for model in self.models]),
            self.dicts[tgt_lang],
            print_alignment=print_alignment,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(self.dicts[tgt_lang]),
        )

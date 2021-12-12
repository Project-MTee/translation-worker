"""
Fallback preprocessing method to meet the bare minimum of tagged translation constraints.
All tags are removed and appended to the translation.
"""
import re
import html
from typing import List, Tuple
from collections import defaultdict
from itertools import chain

tagged_input_types = ('document', 'web')
tag_pattern = r'<[^/>]*>\s*</[^>]*>|' \
              r'<[^>]*>'

bpt = re.compile(r'<[^/>]*>')
ept = re.compile(r'</[^>]*>')
ph = re.compile(r'<[^/>]*>\s*</[^>]*>|<[^>]*/>')

# Other symbols do not need replacing
html_entities = {'<': '&lt;',
                 '>': '&gt;',
                 '&': '&amp;'}


def preprocess_tags(sentences: List[str], input_type: str) -> (List[str], List[List[Tuple[str, int, str]]]):
    # TODO: search for cases, where bpt and ept are with zero span, eg
    # TODO: '<g id="1">BLA <g id="3"> </g> YouTube</g>'
    # <g id="1">EDEXIM European Database Export Import of Dangerous Chemicals,</g>
    # <g id="2">http://edexim.jrc.it/</g><g id="3"> </g><g id="4">(last accessed 15.05.2011).</g>
    if input_type in tagged_input_types:
        clean_sentences = []
        tags = []
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_tags = []  # list of tuples (tag, indexes, tag_type)

            tokens = list(filter(None, re.split(rf' |{tag_pattern}', sentence)))
            tokens_w_tags = list(filter(None, re.split(rf' |({tag_pattern})', sentence)))

            clean_sentences.append(' '.join(tokens).strip())

            for idx, item in enumerate(tokens_w_tags):
                idx = idx - len(sentence_tags)
                if len(tokens) <= idx or item != tokens[idx]:
                    if len(tokens) <= idx:
                        idx = -1

                    if re.match(ph, item):
                        sentence_tags.append((item, idx, 'ph'))
                    elif re.match(bpt, item):
                        sentence_tags.append((item, idx, 'bpt'))
                    elif re.match(ept, item):
                        sentence_tags.append((item, idx, 'ept'))
                    else:
                        # TODO: What is this? 00:02

                        raise RuntimeError(f"Illegal tag: {item} found.")

            tags.append(sentence_tags)

    else:
        clean_sentences = sentences
        tags = [[] for _ in sentences]

    clean_sentences = [html.unescape(sentence) for sentence in clean_sentences]

    return clean_sentences, tags


def postprocess_tags(translations: List[str], tags: List[List[Tuple[str, int, str]]], input_type: str):
    if input_type in tagged_input_types:
        for symbol, entity in html_entities.items():
            translations = [sentence.replace(symbol, entity) for sentence in translations]

    retagged = []

    for translation, sentence_tags in zip(translations, tags):
        retagged_sentence = []

        tokens = translation.split(' ')

        for idx, token in enumerate(tokens):
            whitespace_added = False
            while sentence_tags and sentence_tags[0][1] == idx:
                if not whitespace_added and sentence_tags[0][2] == 'bpt':
                    retagged_sentence.append(' ')
                    whitespace_added = True
                retagged_sentence.append(sentence_tags.pop(0)[0])
            if not whitespace_added:
                retagged_sentence.append(' ')
            retagged_sentence.append(token)

        retagged.append((''.join(retagged_sentence) + ''.join([tag for tag, _, _ in sentence_tags])).strip())

    return retagged


def postprocess_tags_with_alignment(sources: List[str], translations: List[str], tags: List[List[Tuple[str, int, str]]],
                                    input_type: str, alignments: List[List[Tuple[int, int]]]):
    if input_type in tagged_input_types:
        for symbol, entity in html_entities.items():
            translations = [sentence.replace(symbol, entity) for sentence in translations]
    hyps_split = " ".join(translations).split()
    max_al_ix_src = 0
    max_al_ix_tgt = 0
    aligns_extnd = []
    tags_extnd = []
    for snt_idx, (sent_alignments, sent_tags) in enumerate(zip(alignments, tags)):
        for _align in sent_alignments:
            aligns_extnd.append((_align[0] + max_al_ix_src, _align[1] + max_al_ix_tgt))
        for tag in sent_tags:
            if tag[1] == -1:
                new_tag_idx = max_al_ix_src + len(sources[snt_idx].split())
            else:
                new_tag_idx = max_al_ix_src + tag[1]
            tags_extnd.append((tag[0], new_tag_idx, tag[2]))

        max_al_ix_src = max(i[0] for i in aligns_extnd) + 1
        max_al_ix_tgt = max(i[1] for i in aligns_extnd) + 1

    try:
        if input_type == "web":
            retagged = _postproc_html_sent_with_alignment(hyps_split, tags_extnd, aligns_extnd)
        elif input_type == "document":
            raise NotImplementedError
        else:
            raise NotImplementedError
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        retagged = postprocess_tags(translations, tags, input_type)
    return retagged


def _postproc_html_sent_with_alignment(hyp_split: List[str], tags: List[Tuple[str, int, str]],
                                       alignment: List[Tuple[int, int]]):
    try:
        out_tokens = hyp_split.copy()
        alignment_map = defaultdict(list)
        for _al in alignment:
            alignment_map[_al[0]].append(_al[1])

        tags_to_project = tag_projection_order_html(tags)
        loser_tags = []

        for tag in tags_to_project:
            if len(tag) == 2:
                # Paired tag
                bpt_tag = tag[0]
                ept_tag = tag[1]
                alignment_projection = list(chain(*[alignment_map[jj] for jj in range(bpt_tag[1], ept_tag[1])]))
                if len(alignment_projection) == 0:
                    raise RuntimeError("Alignment projection is empty")
                t_min, t_max = min(alignment_projection), max(alignment_projection)

                # Actual projection part
                # TODO: Fix BPT stuff
                out_tokens[t_min] = f" {bpt_tag[0].replace(' ', '▁')}{out_tokens[t_min].strip()}"
                out_tokens[t_max] = f"{out_tokens[t_max].strip()}{ept_tag[0].replace(' ', '▁')} "
            else:
                alignment_projection = None
                for src_ph_tag_idx in range(tag[1], max(alignment_map.keys())):
                    alignment_projection = alignment_map[src_ph_tag_idx]
                    if len(alignment_projection) > 0:
                        break
                if alignment_projection:
                    t_min = min(alignment_projection)
                    out_tokens[t_min] = f" {tag[0].replace(' ', '▁')}{out_tokens[t_min].strip()}"
                else:
                    loser_tags.append(f"{out_tokens[-1].strip()}{tag[0].replace(' ', '▁')}")

        res = " ".join(out_tokens)

        # TODO: Replace later for clarity. Used FOR DEBUGGING PURPOSES
        first_out = re.sub(" +", " ", res)
        second_out = first_out.replace('▁', ' ').strip()
        return second_out
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        raise RuntimeError(f"Something broken in the code: {str(e)}")


def tag_projection_order(tag_list):
    tag_translation_order = []
    LIFO_paired_tags = []

    for tag in tag_list:
        if tag[2] == "bpt":
            LIFO_paired_tags.append(tag)
        elif tag[2] == "ept":
            try:
                tmp_open_tag_bpt = LIFO_paired_tags.pop()
            except IndexError:
                raise RuntimeError("Malformed alignment!")
            tag_translation_order.append((tmp_open_tag_bpt, tag))
        else:
            tag_translation_order.append(tag)
    if len(LIFO_paired_tags) > 0:
        raise RuntimeError("Malformed/Broken input!")
    return tag_translation_order


def tag_projection_order_html(tag_list):
    tag_translation_order = []
    LIFO_paired_tags = {}

    for tag in tag_list:
        if tag[2] == "bpt":
            tag_identifier = tag[0][1:-1]
            LIFO_paired_tags[tag_identifier] = tag
        elif tag[2] == "ept":
            try:
                tag_identifier = tag[0][2:-1]
                tmp_open_tag_bpt = LIFO_paired_tags.pop(tag_identifier, None)
            except IndexError:
                print(f"Html tag alignment error: {LIFO_paired_tags}")
                raise RuntimeError("Malformed HTML tag-alignment!")
            tag_translation_order.append((tmp_open_tag_bpt, tag))
        else:
            tag_translation_order.append(tag)
    return tag_translation_order

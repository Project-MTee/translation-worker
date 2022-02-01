"""
Fallback preprocessing method to meet the bare minimum of tagged translation constraints.
All tags are removed and appended to the translation.
"""
import re
import html
from dataclasses import dataclass
from typing import List, Tuple
from itertools import chain
from collections import defaultdict

from nmt_worker.schemas import InputType

tag_patterns = {
    InputType.XML: r'</?(?:x|g|bx|ex)[^>]*>',
    InputType.HTML: r'</?(?:a|abbr|acronym|em|strong|b|i|s|strike|u|span|del|ins|sub|sup|code|samp|kbd|var|small|mark|'
                    r'ruby|rt|rp|bdi|bdo)[0-9]+/?>'
}

# tagged_input_types = ('document', 'web')
# tag_pattern = r'<[^/>]*>\s*</[^>]*>|' \
#               r'<[^>]*>'

bpt = re.compile(r'<[^/>]*>')
ept = re.compile(r'</[^>]*>')
ph = re.compile(r'<[^/>]*>\s*</[^>]*>|<[^>]*/>')

# Other symbols do not need replacing
html_entities = {'<': '&lt;',
                 '>': '&gt;',
                 '&': '&amp;'}


def preprocess_tags(sentences: List[str], input_type: InputType) -> (List[str], List[List[Tuple[str, int, str]]]):
    # TODO: search for cases, where bpt and ept are with zero span, eg
    # TODO: '<g id="1">BLA <g id="3"> </g> YouTube</g>'
    # <g id="1">EDEXIM European Database Export Import of Dangerous Chemicals,</g>
    # <g id="2">http://edexim.jrc.it/</g><g id="3"> </g><g id="4">(last accessed 15.05.2011).</g>
    if input_type in tag_patterns:
        pattern = tag_patterns[input_type]
        clean_sentences = []
        tags = []
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_tags = []  # list of tuples (tag, indexes, tag_type)

            tokens = list(filter(None, re.split(rf' |{pattern}', sentence)))
            tokens_w_tags = list(filter(None, re.split(rf' |({pattern})', sentence)))

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


def postprocess_tags(translations: List[str], tags: List[List[Tuple[str, int, str]]], input_type: InputType):
    translations = [sentence.replace("<unk>", "") for sentence in translations]

    if input_type in tag_patterns:
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
    # Ideal case (Tags only in the beginning and the end
    if set(i[1] for i in chain(*tags)) == {0, -1}:
        return " ".join(postprocess_tags(translations, tags, input_type))
    else:
        if input_type in tag_patterns:
            for symbol, entity in html_entities.items():
                translations = [sentence.replace(symbol, entity) for sentence in translations]
        if len(translations) > 1:
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
        else:
            hyps_split = translations[0].split()

            aligns_extnd = alignments[0]
            tags_extnd = []
            for j in tags[0]:
                new_jdx = j[1] if j[1] > -1 else len(sources[0].split())
                tags_extnd.append((j[0], new_jdx, j[2]))

        try:
            if input_type == "document":
                return _postproc_xml_sent_with_alignment(hyps_split, tags_extnd, aligns_extnd)
            elif input_type == "web":
                return _postproc_html_sent_with_alignment(hyps_split, tags_extnd, aligns_extnd)
            else:
                raise NotImplementedError
        except Exception as e:
            print(f"Exception, reverting to fallback: {str(e)}")
            return " ".join(postprocess_tags(translations, tags, input_type))


def _postproc_html_sent_with_alignment(hyp_split: List[str], tags: List[Tuple[str, int, str]],
                                       alignments: List[Tuple[int, int]]):
    # Prepare date structures
    out_tokens = hyp_split.copy()
    alignment_map = build_alignment_map(alignments)
    try:
        tags_to_project = tag_projection_order_html(tags)
    except Exception as e:
        raise RuntimeError(f"Error preparing data for HTML tag projection: {str(e)}")

    try:
        for tag in tags_to_project:
            if len(tag) == 2:
                # Paired tag
                bpt_tag = tag[0]
                ept_tag = tag[1]
                alignment_projection = list(chain(*[alignment_map[jj] for jj in range(bpt_tag[1], ept_tag[1])]))
                if len(alignment_projection) > 0:
                    t_min, t_max = min(alignment_projection), max(alignment_projection)
                    out_tokens[t_min] = f" {bpt_tag[0].replace(' ', '▁')}{out_tokens[t_min].strip()}"
                    out_tokens[t_max] = f"{out_tokens[t_max].strip()}{ept_tag[0].replace(' ', '▁')} "
                else:
                    out_tokens = place_ph_token(tag, alignment_map, out_tokens)
            else:
                out_tokens = place_ph_token(tag, alignment_map, out_tokens)
    except Exception as e:
        raise RuntimeError(f"Error in the tag projecting loop: {str(e)}")

    return re.sub(" +", " ", " ".join(out_tokens)).replace('▁', ' ').strip()


def place_ph_token(tag, alignment_map, out_tokens, xml_min_start=None, xml_max_end=None):
    if len(tag) == 1:
        tag_text = tag[0].replace(' ', '▁')
    elif len(tag) == 2:
        tag_text = f"{tag[0][0].replace(' ', '▁')} {tag[1][0].replace(' ', '▁')}"
        raise NotImplementedError("I guess the code works, but it is easier to use fallback in this case")
    else:
        raise NotImplementedError("Place Unpaired token Error: Number of tags is not 1 or 2.")
    if tag[1] == 0:
        out_tag_index = 0
    elif tag[1] == -1:
        out_tag_index = -1
    else:
        alignment_projection = None
        start_src_index = tag[1] if len(tag) == 1 else tag[0][1]
        end_src_index = max(alignment_map.keys())
        if xml_min_start and xml_max_end:
            start_src_index = start_src_index if xml_max_end > start_src_index > xml_min_start else xml_min_start
            end_src_index = xml_max_end

        for src_ph_tag_idx in range(start_src_index, end_src_index):
            alignment_projection = alignment_map[src_ph_tag_idx]
            if len(alignment_projection) > 0:
                break
        out_tag_index = min(alignment_projection) if alignment_projection else -1
    if out_tag_index >= 0:
        out_tokens[out_tag_index] = f" {tag_text}{out_tokens[out_tag_index].strip()}"
    else:
        out_tokens[-1] = f"{out_tokens[-1].strip()}{tag_text}"
    return out_tokens


def tag_projection_order_old(tag_list):
    tag_translation_order = []
    lifo_paired_tags = []

    for tag in tag_list:
        if tag[2] == "bpt":
            lifo_paired_tags.append(tag)
        elif tag[2] == "ept":
            try:
                tmp_open_tag_bpt = lifo_paired_tags.pop()
            except IndexError:
                raise RuntimeError("Malformed alignment!")
            tag_translation_order.append((tmp_open_tag_bpt, tag))
        else:
            tag_translation_order.append(tag)
    if len(lifo_paired_tags) > 0:
        raise RuntimeError("Malformed/Broken input!")
    return tag_translation_order


def tag_projection_order_html(tag_list):
    tag_translation_order = []
    lifo_paired_tags = {}

    for tag in tag_list:
        if tag[2] == "bpt":
            tag_identifier = tag[0][1:-1]
            lifo_paired_tags[tag_identifier] = tag
        elif tag[2] == "ept":
            try:
                tag_identifier = tag[0][2:-1]
                tmp_open_tag_bpt = lifo_paired_tags.pop(tag_identifier, None)
            except IndexError:
                print(f"Html tag alignment error: {lifo_paired_tags}")
                raise RuntimeError("Malformed HTML tag-alignment!")
            tag_translation_order.append((tmp_open_tag_bpt, tag))
        else:
            tag_translation_order.append(tag)
    return tag_translation_order


@dataclass
class Tag:
    """Class for readability"""
    tag_type: str
    idx: int
    string: str


def tag_projection_tree_builder(tag_list):
    if len(tag_list) <= 2:
        return tag_list
    else:
        bind = -1
        eind = -1
        lifo = 0
        res = []

        children_indexes = []
        for jdx, j in enumerate(tag_list):
            if j.tag_type == "bpt":
                bind = bind if lifo > 0 else jdx
                lifo += 1
            elif j.tag_type == "ept":
                lifo -= 1
                eind = eind if lifo > 0 else jdx
                if lifo == 0:
                    children_indexes.append((bind, eind))
            elif (lifo == 0) and j.tag_type == "ph":
                children_indexes.append((jdx,))
            else:
                pass

        for child in children_indexes:
            if len(child) == 1:
                leaf = tag_list[child[0]]
                res.append(leaf)
            else:
                node_info1 = tag_list[child[0]]
                node_info2 = tag_list[child[1]]
                sub_list = tag_list[child[0] + 1: child[1]]
                _children = tag_projection_tree_builder(sub_list)
                if len(_children) == 0:
                    dt_child = [node_info1, node_info2]
                else:
                    dt_child = [node_info1, _children, node_info2]
                res.append(dt_child)
        return res


def build_alignment_map(alignments: List[Tuple[int, int]]):
    alignment_map = defaultdict(list)
    for _al in alignments:
        alignment_map[_al[0]].append(_al[1])
    return alignment_map


def recursive_tag_projection_with_trees(out_tokens, alignment_map, tag_tree):
    res = out_tokens.copy()
    src_max = max(alignment_map.keys()) + 1

    # min_idx and max_idx are (currently) source indexes
    def _helper(curr_tree, min_idx=0, max_idx=src_max):
        if isinstance(curr_tree, Tag):
            place_ph_token_inplace(curr_tree, alignment_map, res, min_idx, max_idx)
            print(f"Leaf: {curr_tree}, {min_idx} and {max_idx}")
        else:
            if len(curr_tree) == 0:
                return
            elif isinstance(curr_tree[0], Tag) and curr_tree[0].tag_type == "bpt":
                bpt_tag, ept_tag = curr_tree[0], curr_tree[-1]
                min_src_idx, max_src_idx = bpt_tag.idx, ept_tag.idx
                sub_tree_min_src_idx = min_src_idx if min_src_idx > min_idx else min_idx
                sub_tree_max_src_idx = max_src_idx if max_src_idx < max_idx else max_idx

                _helper(curr_tree[1:-1], sub_tree_min_src_idx, sub_tree_max_src_idx)
                print(f"ddbl: {bpt_tag} - {ept_tag}, {min_idx} and {max_idx}")

                # Finally project it
                alignment_projection = list(
                    chain(*[alignment_map[jj] for jj in range(sub_tree_min_src_idx, sub_tree_max_src_idx)]))
                if len(alignment_projection) > 0:
                    t_min, t_max = min(alignment_projection), max(alignment_projection)
                    res[t_min] = f" {bpt_tag.string.replace(' ', '▁')}{res[t_min].strip()}"
                    res[t_max] = f"{res[t_max].strip()}{ept_tag.string.replace(' ', '▁')} "
                else:
                    place_ph_token_inplace([bpt_tag, ept_tag], alignment_map, res, min_idx, max_idx)
            else:
                for sub_tree in curr_tree:
                    _helper(sub_tree)

    _helper(tag_tree)
    return res



def _postproc_xml_sent_with_alignment(hyp_split: List[str], tags: List[Tuple[str, int, str]],
                                      alignments: List[Tuple[int, int]]):
    # Prepare date structures
    out_tokens = hyp_split.copy()
    alignment_map = build_alignment_map(alignments)
    try:
        tag_tree = tag_projection_tree_builder([Tag(i[2], i[1], i[0]) for i in tags])
    except Exception as e:
        raise RuntimeError(f"Error preparing the Tree for XML tag projection: {str(e)}")

    out_tokens = recursive_tag_projection_with_trees(out_tokens, alignment_map, tag_tree)

    return re.sub(" +", " ", " ".join(out_tokens)).replace('▁', ' ').strip()


def place_ph_token_inplace(tag, alignment_map, out_tokens, xml_min_start=None, xml_max_end=None):
    if isinstance(tag, Tag):
        tag_text = tag.string.replace(' ', '▁')
    else:
        tag_text = f"{tag[0].string.replace(' ', '▁')} {tag[1].string.replace(' ', '▁')}"
        tag = tag[0]
    if tag.idx == 0:
        out_tag_index = 0
    elif tag.idx == -1:
        out_tag_index = -1
    else:
        alignment_projection = None
        start_src_index = tag.idx
        end_src_index = max(alignment_map.keys())
        if xml_min_start and xml_max_end:
            start_src_index = start_src_index if xml_max_end > start_src_index > xml_min_start else xml_min_start
            end_src_index = xml_max_end

        for src_ph_tag_idx in range(start_src_index, end_src_index):
            alignment_projection = alignment_map[src_ph_tag_idx]
            if len(alignment_projection) > 0:
                break
        out_tag_index = min(alignment_projection) if alignment_projection else -1
    if out_tag_index >= 0:
        out_tokens[out_tag_index] = f" {tag_text}{out_tokens[out_tag_index].strip()}"
    else:
        out_tokens[-1] = f"{out_tokens[-1].strip()}{tag_text}"
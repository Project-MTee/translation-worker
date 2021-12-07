# adapted from moses punctuation normalization script
import re

# define and compile all the patterns
REGEXES = (
    (re.compile(r'\r'), r''),

    # remove extra spaces
    (re.compile(r'\('), r' ('),
    (re.compile(r'\)'), r') '),
    (re.compile(r' +'), r' '),
    (re.compile(r'\) ([.!:?;,])'), r')\1'),
    (re.compile(r'\( '), r'('),
    (re.compile(r' \)'), r')'),
    (re.compile(r'(\d) %'), r'\1%'),
    (re.compile(r' :'), ':'),
    (re.compile(r' ;'), ';'),
    # normalize unicode punctuation
    (re.compile(r'`'), r"'"),
    (re.compile(r"''"), r' " '),

    (re.compile(r'„'), r'"'),
    (re.compile(r'“'), r'"'),
    (re.compile(r'”'), r'"'),
    (re.compile(r'–'), r'-'),
    (re.compile(r'—'), r' - '),
    (re.compile(r' +'), r' '),
    (re.compile(r'´'), r"'"),
    (re.compile(r'([a-z])‘([a-z])', flags=re.IGNORECASE), r"\1'\2"),
    (re.compile(r'([a-z])’([a-z])', flags=re.IGNORECASE), r"\1'\2"),
    (re.compile(r'‘'), r"'"),
    (re.compile(r'‚'), r"'"),
    (re.compile(r'’'), r"'"),
    (re.compile(r"''"), r'"'),
    (re.compile(r'´´'), r'"'),
    (re.compile(r'…'), r'...'),
    # French quotes
    (re.compile(r' « '), r' "'),
    (re.compile(r'« '), r'"'),
    (re.compile(r'«'), r'"'),
    (re.compile(r' » '), r'" '),
    (re.compile(r' »'), r'"'),
    (re.compile(r'»'), r'"'),
    # handle pseudo-spaces
    (re.compile(r' %'), r'%'),
    (re.compile(r'nº '), r'nº '),
    (re.compile(r' :'), r':'),
    (re.compile(r' ºC'), r' ºC'),
    (re.compile(r' cm'), r' cm'),
    (re.compile(r' \?'), r'?'),
    (re.compile(r' !'), r'!'),
    (re.compile(r' ;'), r';'),
    (re.compile(r', '), r', '),
    (re.compile(r' +'), r' ')
)


def normalize(sentence: str, language_code: str = None):
    for regex, sub in REGEXES:
        sentence = regex.sub(sub, sentence)
    return sentence

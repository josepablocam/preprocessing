#!/usr/bin/env python3
import ast
import astunparse
from functools import reduce
import keyword
import os
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import tqdm

ENGLISH_STOPWORDS = stopwords.words("english")
ENGLISH_STEMMER = PorterStemmer()


class AbstractAstTokenCollector(ast.NodeVisitor):
    def __init__(self):
        self.tokens = []

    def run(self, src):
        self.tokens = []
        try:
            self.visit(ast.parse(src))
            return list(self.tokens)
        except:
            print("Parse failed")
            return []


class CallCollector(AbstractAstTokenCollector):
    def visit_Call(self, node):
        func = astunparse.unparse(node.func).strip()
        self.tokens.append(func)


class NameCollector(AbstractAstTokenCollector):
    def visit_Name(self, node):
        name = node.id.strip()
        self.tokens.append(name)
        self.generic_visit(node)


class DefinitionNameCollector(AbstractAstTokenCollector):
    def visit_FunctionDef(self, node):
        self.tokens.append(node.name.strip())


def extract_call_tokens(src):
    assert isinstance(src, str)
    return CallCollector().run(src)


def extract_name_tokens(src):
    assert isinstance(src, str)
    return NameCollector().run(src)


def extract_def_name(src):
    assert isinstance(src, str)
    return DefinitionNameCollector().run(src)


def remove_code_from_nl(nl):
    raise NotImplementedError()


def remove_keywords(_input):
    if isinstance(_input, str):
        _input = split_on_whitespace(_input)
    return [t for t in _input if not keyword.iskeyword(t)]


def split_on_whitespace(text):
    return text.split()


def lower_case(_input):
    if isinstance(_input, str):
        _input = split_on_whitespace(_input)
    return [t.lower() for t in _input]


def _split_camel_case_regex(token):
    return re.sub('([a-z])([A-Z])', r'\1 \2', token).split()


def _split_on_camel_case_single(_str):
    # replace into spaces
    replace_regex = "[!\?'\"\*\+\-\=\(\)\[\]\{\}:,]"
    clean_str = re.sub(replace_regex, " ", _str)
    # split on underscore, period, and whitespace
    split_regex = "[_\\.\s+]"
    tokens = re.split(split_regex, clean_str)
    # split again but on camel case and lower case them
    tokens = [
        subtoken.lower() for token in tokens
        for subtoken in _split_camel_case_regex(token) if len(token) > 0
    ]
    return tokens


def split_on_camel_case(_input):
    if isinstance(_input, str):
        _input = split_on_whitespace(_input)
    return [st for t in _input for st in _split_on_camel_case_single(t)]


def remove_english_stopwords(_input):
    if isinstance(_input, str):
        _input = split_on_whitespace(_input)
    return [t for t in _input if t not in ENGLISH_STOPWORDS]


def stem_english_words(_input):
    if isinstance(_input, str):
        _input = split_on_whitespace(_input)
    return [ENGLISH_STEMMER.stem(t) for t in _input]


def sequence(*fs):
    return lambda _input: reduce(lambda _in, f: f(_in), fs, _input)


def plus(*fs):
    return lambda _input: sum([f(_input) for f in fs], [])


class CanonicalInput(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.transformed = corpus
        # only set if downsampled at some point
        self.downsampled = False
        self.full_corpus = None

    def reset(self):
        if self.downsampled:
            assert self.full_corpus is not None
            self.corpus = self.full_corpus
        self.downsampled = False
        self.transformed = self.corpus

    def apply_pipeline(self, pipeline, which):
        print("Applying pipeline to {} dimension".format(which))
        new_dataset = []
        for obs in tqdm.tqdm(self.transformed):
            code = obs["code"]
            nl = obs["nl"]

            if which == "code":
                code = " ".join(pipeline(code))
            elif which == "nl":
                nl = " ".join(pipeline(nl))
            elif which == "both":
                code = " ".join(pipeline(code))
                nl = " ".join(pipeline(nl))
            else:
                raise ValueError("Unknown member name")

            if len(code) > 0 and len(nl) > 0:
                new_dataset.append({"code": code, "nl": nl})

        self.transformed = new_dataset

    def downsample(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # back up before downsampling
        self.downsampled = True
        self.full_corpus = list(self.corpus)

        np.random.shuffle(self.corpus)
        self.corpus = self.corpus[:n]
        self.transformed = self.corpus

    def to_text(self, code_output, nl_output):
        with open(code_output, "w") as code, open(nl_output, "w") as nl:
            for obs in self.transformed:
                code.write(obs["code"] + "\n")
                nl.write(obs["nl"] + "\n")

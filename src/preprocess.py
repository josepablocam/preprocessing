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
        func = func.replace("\n", " ")
        self.tokens.append(func)


class NameCollector(AbstractAstTokenCollector):
    def visit_Name(self, node):
        name = node.id.strip()
        name = name.replace("\n", " ")
        self.tokens.append(name)
        self.generic_visit(node)


class DefinitionNameCollector(AbstractAstTokenCollector):
    def visit_FunctionDef(self, node):
        name = node.name.strip()
        name = name.replace("\n", " ")
        self.tokens.append(name)
        self.generic_visit(node)


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


def _split_on_code_characters_single(_str):
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


def split_on_code_characters(_input):
    """
    Splits on camel case, snake case, period, certain operators, parentheses,
    braces, commas, and colons
    """
    if isinstance(_input, str):
        _input = split_on_whitespace(_input)
    return [st for t in _input for st in _split_on_code_characters_single(t)]


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

        # only if transformed at some point
        self.is_transformed = False
        self.transformed = None

        # only set if downsampled at some point
        self.is_downsampled = False
        self.full_corpus = None

    def reset(self):
        if self.is_downsampled:
            assert self.full_corpus is not None
            self.corpus = self.full_corpus

        self.is_downsampled = False
        self.full_corpus = None

        self.is_transformed = False
        self.transformed = None

    def apply_pipeline(self, pipeline, which):
        print("Applying pipeline to {} dimension".format(which))
        new_dataset = []
        # incrementally apply transformations to previous if available
        if self.is_transformed:
            assert self.is_transformed is not None
            dataset = self.transformed
        else:
            dataset = self.corpus

        for obs in tqdm.tqdm(dataset):
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
        self.is_transformed = True

    def downsample(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # back up before downsampling
        self.downsampled = True
        self.full_corpus = list(self.corpus)

        if self.is_transformed:
            assert self.is_transformed is not None
            np.random.shuffle(self.transformed)
            self.transformed = self.transformed[:n]
        else:
            np.random.shuffle(self.corpus)
            self.corpus = self.corpus[:n]

    def to_text(self, code_output, nl_output):
        if self.is_transformed:
            assert self.is_transformed is not None
            dataset = self.transformed
        else:
            dataset = self.corpus

        with open(code_output, "w") as code, open(nl_output, "w") as nl:
            for obs in dataset:
                code.write(obs["code"] + "\n")
                nl.write(obs["nl"] + "\n")

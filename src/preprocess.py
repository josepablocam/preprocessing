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
            if getattr(self, "debug", None) is not None:
                import pdb
                pdb.set_trace()
            print("Parse failed")
            return []


class CallCollector(AbstractAstTokenCollector):
    def visit_Call(self, node):
        func = astunparse.unparse(node.func).strip()
        # do arguments first so that order of execution
        # matches order of collection
        args = node.args + node.keywords
        for arg in args:
            self.visit(arg)
        func = func.replace("\n", " ")
        self.tokens.append(func)


class NameCollector(AbstractAstTokenCollector):
    def visit_Name(self, node):
        name = node.id.strip()
        name = name.replace("\n", " ")
        self.tokens.append(name)
        self.generic_visit(node)


class DefinitionNameCollector(AbstractAstTokenCollector):
    def visit_ClassDef(self, node):
        name = node.name.strip()
        name = name.replace("\n", " ")
        self.tokens.append(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        name = node.name.strip()
        name = name.replace("\n", " ")
        self.tokens.append(name)
        self.generic_visit(node)


class QualifiedDefinitionNameCollector(AbstractAstTokenCollector):
    def __init__(self):
        super().__init__()
        self.context = []

    def qualify_name(self, name):
        if len(self.context) == 0:
            return name
        prefix = ".".join(self.context)
        return prefix + "." + name

    def visit_ClassDef(self, node):
        name = node.name
        name = name.replace("\n", " ")
        full_name = self.qualify_name(name)
        self.tokens.append(full_name)

        self.context.append(name)
        self.generic_visit(node)
        self.context.pop()

    def visit_FunctionDef(self, node):
        name = node.name.strip()
        name = name.replace("\n", " ")
        full_name = self.qualify_name(name)
        self.tokens.append(full_name)

        self.context.append(name)
        self.generic_visit(node)
        self.context.pop()


def extract_call_tokens(src):
    assert isinstance(src, str)
    return CallCollector().run(src)


def extract_name_tokens(src):
    assert isinstance(src, str)
    return NameCollector().run(src)


def extract_def_name(src):
    assert isinstance(src, str)
    return DefinitionNameCollector().run(src)


def extract_def_name_and_call_tokens_resilient(src):
    """
    tries to extract def-name etc but if no name (i.e. conala snippet), just
    returns original source to be further processed
    """

    name_tokens = NameCollector().run(src)
    if len(name_tokens) == 0:
        return src
    call_tokens = CallCollector().run(src)
    return name_tokens + call_tokens


def extract_qualified_def_name(src):
    assert isinstance(src, str)
    return QualifiedDefinitionNameCollector().run(src)


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


def remove_params_and_returns(nl):
    assert isinstance(nl, str)
    # Remove params and returns in docstring
    indicate_regex = "\nparam|\nParam|\nparams|\nParams|\nPARAM|\nPARAMS|\nreturn|\nreturns|\nReturn|\nReturns|\nRETURN|\nRETURNS|:param|:Param|:params|:Params|:PARAM|:PARAMS|:return|:returns|:Return|:Returns|:RETURN|:RETURNS"
    result = re.search(indicate_regex, nl)
    if result is not None:
        truncated = nl[:result.span()[0]]
        if len(truncated) > 0:
            return truncated
        else:
            return nl
    else:
        return nl


def take_first_sentence(nl):
    # take the first "sentence"
    assert isinstance(nl, str)
    return nl.split(".")[0]


def sequence(*fs):
    return lambda _input: reduce(lambda _in, f: f(_in), fs, _input)


def plus(*fs):
    return lambda _input: sum([f(_input) for f in fs], [])


class CanonicalInput(object):
    def __init__(self, corpus, transformed=None):
        self.corpus = corpus

        # only if transformed at some point
        self.is_transformed = transformed is not None
        self.transformed = transformed

    def reset(self):
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
        if self.is_transformed:
            assert self.is_transformed is not None
            chosen = np.random.choice(self.transformed, size=n)
            return CanonicalInput(self.corpus, transformed=chosen)
        else:
            assert self.is_transformed is None
            chosen = np.random.choice(self.corpus, size=n)
            return CanonicalInput(chosen, transformed=None)

    def to_text(self, code_output, nl_output):
        if self.is_transformed:
            assert self.is_transformed is not None
            dataset = self.transformed
        else:
            dataset = self.corpus

        with open(code_output, "w") as code, open(nl_output, "w") as nl:
            for obs in dataset:
                # Some observations can be empty after we strip all
                # non-ascii characters
                code_txt = obs["code"].encode("ascii",
                                              "ignore").decode().strip()
                nl_txt = obs["nl"].encode("ascii", "ignore").decode().strip()
                if len(code_txt) == 0 or len(nl_txt) == 0:
                    continue
                code.write(code_txt + "\n")
                nl.write(nl_txt + "\n")

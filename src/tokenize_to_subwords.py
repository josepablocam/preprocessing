#!/usr/bin/env python3

import argparse

import tqdm

from . import data


def get_args():
    parser = argparse.ArgumentParser(description="Text to subword tokens")
    parser.add_argument("-i", "--input", type=str, help="Input text")
    parser.add_argument("-o", "--output", type=str, help="Output path")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    tokens = data.read_doc_as_tokenized_lines(args.input)
    with open(args.output, "w") as fout:
        for line_tokens in tqdm.tqdm(tokens):
            line = " ".join(line_tokens)
            fout.write(line + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

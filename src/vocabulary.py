#!/usr/bin/env python3
import argparse
import pickle

import numpy as np
import tqdm

from . import data
from .precomputed_embeddings import read_embeddings


def encode_doc_to_matrix(doc, vocab_encoder, length):
    if isinstance(doc[0], str):
        # assume we have not tokenized yet, since first line is string not list
        tokens = [line.split() for line in tqdm.tqdm(doc)]
    else:
        tokens = doc
    encoded = vocab_encoder.encode(tokens)
    # pad to appropriate length so we can create matrix
    matrix = data.create_padded_dataset(encoded, length)
    return matrix


def compute_encoder(input_path, top_k, output_path):
    tokens = data.read_doc_as_tokenized_lines(input_path)
    vocab = data.build_vocab(tokens, top_k=top_k)
    vocab_encoder = data.VocabEncoder(vocab)
    with open(output_path, "wb") as fout:
        pickle.dump(vocab_encoder, fout)


def encoder_from_embeddings(embeddings_path, output_path):
    embs = read_embeddings(embeddings_path)
    vocab_encoder = data.VocabEncoder(list(embs.keys()))
    with open(output_path, "wb") as fout:
        pickle.dump(vocab_encoder, fout)


def apply_encoder(input_path, encoder_path, target_len, format, output_path):
    tokens = data.read_doc_as_tokenized_lines(input_path)
    with open(encoder_path, "rb") as fin:
        vocab_encoder = pickle.load(fin)
    encoded = encode_doc_to_matrix(tokens, vocab_encoder, target_len)
    if format == "numpy":
        np.save(output_path, encoded)
    elif format == "text":
        with open(output_path, "w") as fout:
            for row in encoded:
                fout.write(" ".join(row) + "\n")
    else:
        raise ValueError("Unknown format")


def get_args():
    parser = argparse.ArgumentParser("Vocabulary encoding")
    subparsers = parser.add_subparsers(help="Actions")

    load_parser = subparsers.add_parser("load-from-embeddings")
    load_parser.set_defaults(which="load-from-embeddings")
    load_parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input embeddings .vec file",
    )
    load_parser.add_argument(
        "-o", "--output", type=str, help="Path to dump pickled vocab encoder"
    )

    compute_parser = subparsers.add_parser("compute")
    compute_parser.set_defaults(which="compute")
    compute_parser.add_argument(
        "-i", "--input", type=str, help="Input text file"
    )
    compute_parser.add_argument(
        "-k", "--top_k", type=int, help="Top k words for vocab"
    )
    compute_parser.add_argument(
        "-o", "--output", type=str, help="Path to dump pickled vocab encoder"
    )

    apply_parser = subparsers.add_parser("apply")
    apply_parser.set_defaults(which="apply")
    apply_parser.add_argument(
        "-i", "--input", type=str, help="Input text file"
    )
    apply_parser.add_argument(
        "-v", "--vocab", type=str, help="Path to pickled vocab encoder"
    )
    apply_parser.add_argument(
        "-l", "--length", type=int, help="Target padded length for each entry"
    )
    apply_parser.add_argument(
        "-o", "--output", type=str, help="Path to dump output"
    )
    apply_parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["numpy", "text"],
        help="Type of output",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.which == "compute":
        compute_encoder(args.input, args.top_k, args.output)
    elif args.which == "apply":
        apply_encoder(
            args.input,
            args.vocab,
            args.length,
            args.format,
            args.output,
        )
    elif args.which == "load-from-embeddings":
        encoder_from_embeddings(
            args.input,
            args.output,
        )
    else:
        raise Exception("Unknown command: {}".format(args.which))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

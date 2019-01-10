import argparse
import subprocess
import tempfile

import numpy as np
import tqdm

from . import data
from . import utils

FASTTEXT_EXECUTABLE = "fastText-0.1.0/fasttext"


def make_embeddings(tokenized_doc, dim, output_file_path):
    f = tempfile.NamedTemporaryFile("w", delete=True)
    for row_of_tokens in tokenized_doc:
        f.write(" ".join(row_of_tokens))
        f.write("\n")
    f.flush()
    f.seek(0)

    subprocess.call([
        FASTTEXT_EXECUTABLE,
        "skipgram",
        "-input",
        f.name,
        "-output",
        output_file_path,
        "-dim",
        str(dim),
    ], )
    f.close()


def run_fasttext(dim, code_path, embeddings_path):
    code_tokens = data.read_doc_as_tokenized_lines(code_path)
    make_embeddings(
        code_tokens,
        dim,
        embeddings_path,
    )


def read_embeddings(fasttext_file_path):
    embeddings = {}
    with open(fasttext_file_path, "r") as fin:
        # toss out header
        fin.readline()
        for line in tqdm.tqdm(fin):
            vals = line.split()
            word = vals[0]
            embedding = np.array([float(v) for v in vals[1:]])
            embeddings[word] = embedding
    return embeddings


def get_args():
    parser = argparse.ArgumentParser("Create precomputed embeddings")
    parser.add_argument(
        "-c",
        "--code",
        type=str,
        help="Code to use for precomputed embeddings",
        choices=["github", "conala", "downsampled"],
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        help="Dimensionality",
        default=500,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input tokens to use to compute embeddings",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for embeddings",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.code is None:
        code_path = args.input
        embeddings_path = args.output
    else:
        if args.code == "github":
            code_path = "../data/github/train.function_relevant"
            embeddings_path = "../data/train/precomputed_code_embeddings"
        elif args.code == "conala":
            code_path = "../data/conala-corpus/conala-mined-code.txt"
            embeddings_path = "../data/conala-train-mined/precomputed_code_embeddings"
        elif args.code == "downsampled":
            code_path = "../data/github_downsampled/train.function_relevant"
            embeddings_path = "../data/github_downsampled/precomputed_code_embeddings"
        else:
            raise ValueError("Unknown option: {}".format(args.code))
        code_path = utils.relative_to_src_dir(code_path)
        embeddings_path = utils.relative_to_src_dir(embeddings_path)

    run_fasttext(args.dim, code_path, embeddings_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

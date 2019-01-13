import argparse
import subprocess
import tempfile

import numpy as np
import tqdm

from . import data

FASTTEXT_EXECUTABLE = "fastText-0.1.0/fasttext"


def make_embeddings(tokenized_doc, min_count, dim, output_file_path):
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
        "-minCount",
        str(min_count),
    ], )
    f.close()


def run_fasttext(code_path, min_count, dim, embeddings_path):
    code_tokens = data.read_doc_as_tokenized_lines(code_path)
    make_embeddings(
        code_tokens,
        min_count,
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
        "-d",
        "--dim",
        type=int,
        help="Dimensionality",
        default=500,
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        help="Min count for word occurrences",
        default=5,
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
    run_fasttext(args.input, args.count, args.dim, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

import argparse
import json

import numpy as np
import tqdm
import pickle

from . import configuration
from . import data
from . import utils


def encode_doc_to_matrix(doc, vocab_encoder, length):
    if isinstance(doc[0], str):
        # assume we have not tokenized yet, since first line is string not list
        tokens = [data.split_into_subtokens(line) for line in tqdm.tqdm(doc)]
    else:
        tokens = doc
    encoded = vocab_encoder.encode(tokens)
    # pad to appropriate length so we can create matrix
    matrix = data.create_padded_dataset(encoded, length)
    return matrix


def save_tokens(tokens, path):
    with open(path, "w") as fout:
        for row in tokens:
            fout.write(" ".join(row))
            fout.write("\n")


def build_train_datasets_github_so(config):
    code_top_k = config["code_vocab_top_k"]
    nl_top_k = config["nl_vocab_top_k"]
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    # Github Python functions -> code
    code_path = utils.relative_to_src_dir(
        "../data/github/train.function_relevant"
    )
    code_tokens = data.read_doc_as_tokenized_lines(code_path)

    save_tokens(
        code_tokens,
        utils.relative_to_src_dir("../data/github/verify.function_relevant")
    )

    code_vocab = data.build_vocab(code_tokens, top_k=code_top_k)
    code_vocab_encoder = data.VocabEncoder(code_vocab)
    github_code_matrix = encode_doc_to_matrix(
        code_tokens,
        code_vocab_encoder,
        code_length,
    )

    # Github Python docstrings
    docstring_path = utils.relative_to_src_dir(
        "../data/github/train.docstring"
    )
    # Stackoverflow Python question titles
    queries_path = utils.relative_to_src_dir(
        "../data/stackoverflow/python_titles.txt"
    )

    # Jointly they constitute NL
    docstring_tokens = data.read_doc_as_tokenized_lines(docstring_path)
    save_tokens(
        docstring_tokens,
        utils.relative_to_src_dir("../data/github/verify.docstring")
    )
    queries_tokens = data.read_doc_as_tokenized_lines(queries_path)

    nl_tokens = docstring_tokens + queries_tokens
    nl_vocab = data.build_vocab(nl_tokens, top_k=nl_top_k)
    nl_vocab_encoder = data.VocabEncoder(nl_vocab)

    # Github training docstrings
    github_docstrings_matrix = encode_doc_to_matrix(
        docstring_tokens, nl_vocab_encoder, nl_length
    )

    # stackoverflow titles/queries
    so_queries_matrix = encode_doc_to_matrix(
        queries_tokens, nl_vocab_encoder, nl_length
    )

    # align github code/docstrings to remove entries with just padding
    github_code_matrix, github_docstrings_matrix = data.remove_empty_entries([
        github_code_matrix,
        github_docstrings_matrix,
    ])

    # SO titles dont need to be aligned
    so_queries_matrix = data.remove_empty_entries(so_queries_matrix)

    # save datasets out
    utils.create_dir(utils.relative_to_src_dir("../data/train/"))
    np.save(
        utils.relative_to_src_dir("../data/train/github_code.npy"),
        github_code_matrix,
    )
    np.save(
        utils.relative_to_src_dir("../data/train/github_docstrings.npy"),
        github_docstrings_matrix,
    )
    np.save(
        utils.relative_to_src_dir("../data/train/so_queries.npy"),
        so_queries_matrix,
    )

    # save down vocab encoders
    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/train/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "wb") as fout:
        pickle.dump(code_vocab_encoder, fout)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/train/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "wb") as fout:
        pickle.dump(nl_vocab_encoder, fout)


def build_train_datasets_conala_mined(config):
    """ Build dataset using conala-mined as single training dataset, no adversarial """
    code_top_k = config["code_vocab_top_k"]
    nl_top_k = config["nl_vocab_top_k"]
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    # Github Python functions -> code
    code_path = utils.relative_to_src_dir(
        "../data/conala-corpus/conala-mined-code.txt"
    )
    code_tokens = data.read_doc_as_tokenized_lines(code_path)

    code_vocab = data.build_vocab(code_tokens, top_k=code_top_k)
    code_vocab_encoder = data.VocabEncoder(code_vocab)
    code_matrix = encode_doc_to_matrix(
        code_tokens,
        code_vocab_encoder,
        code_length,
    )

    queries_path = utils.relative_to_src_dir(
        "../data/conala-corpus/conala-mined-queries.txt"
    )
    queries_tokens = data.read_doc_as_tokenized_lines(queries_path)

    nl_vocab = data.build_vocab(queries_tokens, top_k=nl_top_k)
    nl_vocab_encoder = data.VocabEncoder(nl_vocab)

    # conala titles/queries
    queries_matrix = encode_doc_to_matrix(
        queries_tokens, nl_vocab_encoder, nl_length
    )
    # align github code/docstrings to remove entries with just padding
    code_matrix, queries_matrix = data.remove_empty_entries([
        code_matrix,
        queries_matrix,
    ])

    # save datasets out
    utils.create_dir(utils.relative_to_src_dir("../data/conala-train-mined/"))
    np.save(
        utils.relative_to_src_dir("../data/conala-train-mined/code.npy"),
        code_matrix,
    )
    np.save(
        utils.relative_to_src_dir("../data/conala-train-mined/queries.npy"),
        queries_matrix,
    )

    # save down vocab encoders
    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-train-mined/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "wb") as fout:
        pickle.dump(code_vocab_encoder, fout)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-train-mined/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "wb") as fout:
        pickle.dump(nl_vocab_encoder, fout)


def build_train_datasets_advconala(config):
    '''Build dataset using conala-mined as adversarial queries'''
    code_top_k = config["code_vocab_top_k"]
    nl_top_k = config["nl_vocab_top_k"]
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    # Github Python functions -> code
    code_path = utils.relative_to_src_dir(
        "../data/github/train.function_relevant"
    )
    code_tokens = data.read_doc_as_tokenized_lines(code_path)

    code_vocab = data.build_vocab(code_tokens, top_k=code_top_k)
    code_vocab_encoder = data.VocabEncoder(code_vocab)
    github_code_matrix = encode_doc_to_matrix(
        code_tokens,
        code_vocab_encoder,
        code_length,
    )

    # Github Python docstrings
    docstring_path = utils.relative_to_src_dir(
        "../data/github/train.docstring"
    )
    # Conala-mined intents
    queries_path = utils.relative_to_src_dir(
        "../data/conala-corpus/conala-mined-intents.txt"
    )

    # Jointly they constitute NL
    docstring_tokens = data.read_doc_as_tokenized_lines(docstring_path)

    queries_tokens = data.read_doc_as_tokenized_lines(queries_path)

    nl_tokens = docstring_tokens + queries_tokens
    nl_vocab = data.build_vocab(nl_tokens, top_k=nl_top_k)
    nl_vocab_encoder = data.VocabEncoder(nl_vocab)

    # Github training docstrings
    github_docstrings_matrix = encode_doc_to_matrix(
        docstring_tokens, nl_vocab_encoder, nl_length
    )

    # conala titles/queries
    conala_queries_matrix = encode_doc_to_matrix(
        queries_tokens, nl_vocab_encoder, nl_length
    )

    # align github code/docstrings to remove entries with just padding
    github_code_matrix, github_docstrings_matrix = data.remove_empty_entries([
        github_code_matrix,
        github_docstrings_matrix,
    ])

    # SO titles dont need to be aligned
    conala_queries_matrix = data.remove_empty_entries(conala_queries_matrix)

    # save datasets out
    utils.create_dir(utils.relative_to_src_dir("../data/conala-adv-train/"))
    np.save(
        utils.relative_to_src_dir("../data/conala-adv-train/github_code.npy"),
        github_code_matrix,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/conala-adv-train/github_docstrings.npy"),
        github_docstrings_matrix,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/conala-adv-train/conala_queries.npy"),
        conala_queries_matrix,
    )

    # save down vocab encoders
    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-adv-train/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "wb") as fout:
        pickle.dump(code_vocab_encoder, fout)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-adv-train/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "wb") as fout:
        pickle.dump(nl_vocab_encoder, fout)


def build_train_datasets_downsampled(config):
    '''Build dataset using downsampled github corpus'''
    code_top_k = config["code_vocab_top_k"]
    nl_top_k = config["nl_vocab_top_k"]
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    # Github Python functions -> code
    code_path = utils.relative_to_src_dir(
        "../data/github_downsampled/train.function_relevant"
    )
    code_tokens = data.read_doc_as_tokenized_lines(code_path)

    code_vocab = data.build_vocab(code_tokens, top_k=code_top_k)
    code_vocab_encoder = data.VocabEncoder(code_vocab)
    github_code_matrix = encode_doc_to_matrix(
        code_tokens,
        code_vocab_encoder,
        code_length,
    )

    # Github Python docstrings
    docstring_path = utils.relative_to_src_dir(
        "../data/github_downsampled/train.docstring"
    )
    # Conala-mined intents
    queries_path = utils.relative_to_src_dir(
        "../data/conala-corpus/conala-mined-queries.txt"
    )

    # Jointly they constitute NL
    docstring_tokens = data.read_doc_as_tokenized_lines(docstring_path)

    queries_tokens = data.read_doc_as_tokenized_lines(queries_path)

    nl_tokens = docstring_tokens + queries_tokens
    nl_vocab = data.build_vocab(nl_tokens, top_k=nl_top_k)
    nl_vocab_encoder = data.VocabEncoder(nl_vocab)

    # Github training docstrings
    github_docstrings_matrix = encode_doc_to_matrix(
        docstring_tokens, nl_vocab_encoder, nl_length
    )

    # conala titles/queries
    conala_queries_matrix = encode_doc_to_matrix(
        queries_tokens, nl_vocab_encoder, nl_length
    )

    # align github code/docstrings to remove entries with just padding
    github_code_matrix, github_docstrings_matrix = data.remove_empty_entries([
        github_code_matrix,
        github_docstrings_matrix,
    ])

    # SO titles dont need to be aligned
    conala_queries_matrix = data.remove_empty_entries(conala_queries_matrix)

    # save datasets out
    utils.create_dir(utils.relative_to_src_dir("../data/github_downsampled/train/"))
    np.save(
        utils.relative_to_src_dir("../data/github_downsampled/train/github_code.npy"),
        github_code_matrix,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/github_downsampled/train/github_docstrings.npy"),
        github_docstrings_matrix,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/github_downsampled/train/conala_queries.npy"),
        conala_queries_matrix,
    )

    # save down vocab encoders
    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/github_downsampled/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "wb") as fout:
        pickle.dump(code_vocab_encoder, fout)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/github_downsampled/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "wb") as fout:
        pickle.dump(nl_vocab_encoder, fout)


def parse_conala_file(conala_file_path):
    with open(conala_file_path, "r") as fin:
        data = json.load(fin)
    code = []
    queries = []
    for entry in tqdm.tqdm(data):
        code.append(entry["snippet"])
        queries.append(entry["intent"])
    return code, queries


def encode_conala_file(
        conala_file_path,
        code_vocab_encoder,
        nl_vocab_encoder,
        code_length,
        nl_length,
):
    conala_code, conala_nl = parse_conala_file(conala_file_path)
    conala_code_matrix = encode_doc_to_matrix(
        conala_code, code_vocab_encoder, code_length
    )
    conala_nl_matrix = encode_doc_to_matrix(
        conala_nl, nl_vocab_encoder, nl_length
    )
    # remove empty just padding
    conala_code_matrix, conala_nl_matrix = data.remove_empty_entries([
        conala_code_matrix,
        conala_nl_matrix,
    ])
    return conala_code_matrix, conala_nl_matrix


def build_test_datasets_github_so(config):
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/train/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "rb") as fin:
        code_vocab_encoder = pickle.load(fin)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/train/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "rb") as fin:
        nl_vocab_encoder = pickle.load(fin)

    # We use github test data as validation data
    github_code_path = utils.relative_to_src_dir(
        "../data/github/test.function_relevant"
    )
    with open(github_code_path, "r") as fin:
        github_code_matrix = encode_doc_to_matrix(
            fin.readlines(), code_vocab_encoder, code_length
        )

    github_docstrings_path = utils.relative_to_src_dir(
        "../data/github/test.docstring"
    )
    with open(github_docstrings_path, "r") as fin:
        github_docstrings_matrix = encode_doc_to_matrix(
            fin.readlines(), nl_vocab_encoder, nl_length
        )

    # remove empty entries of only padding
    github_code_matrix, github_docstrings_matrix = data.remove_empty_entries([
        github_code_matrix,
        github_docstrings_matrix,
    ])

    # We use CONALA training data as validation data
    conala_valid_code, conala_valid_nl = encode_conala_file(
        utils.relative_to_src_dir("../data/conala-corpus/conala-train.json"),
        code_vocab_encoder,
        nl_vocab_encoder,
        code_length,
        nl_length,
    )

    # We use CONALA test data as test data
    conala_test_code, conala_test_nl = encode_conala_file(
        utils.relative_to_src_dir("../data/conala-corpus/conala-test.json"),
        code_vocab_encoder,
        nl_vocab_encoder,
        code_length,
        nl_length,
    )

    utils.create_dir(utils.relative_to_src_dir("../data/valid/"))
    utils.create_dir(utils.relative_to_src_dir("../data/test/"))
    np.save(
        utils.relative_to_src_dir("../data/valid/github_code.npy"),
        github_code_matrix,
    )
    np.save(
        utils.relative_to_src_dir("../data/valid/github_docstrings.npy"),
        github_docstrings_matrix,
    )
    np.save(
        utils.relative_to_src_dir("../data/valid/conala_code.npy"),
        conala_valid_code,
    )
    np.save(
        utils.relative_to_src_dir("../data/valid/conala_queries.npy"),
        conala_valid_nl,
    )
    np.save(
        utils.relative_to_src_dir("../data/test/conala_code.npy"),
        conala_test_code,
    )
    np.save(
        utils.relative_to_src_dir("../data/test/conala_queries.npy"),
        conala_test_nl,
    )


def build_test_datasets_advconala(config):
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-adv-train/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "rb") as fin:
        code_vocab_encoder = pickle.load(fin)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-adv-train/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "rb") as fin:
        nl_vocab_encoder = pickle.load(fin)

    # We use github test data as validation data
    github_code_path = utils.relative_to_src_dir(
        "../data/github/test.function_relevant"
    )
    with open(github_code_path, "r") as fin:
        github_code_matrix = encode_doc_to_matrix(
            fin.readlines(), code_vocab_encoder, code_length
        )

    github_docstrings_path = utils.relative_to_src_dir(
        "../data/github/test.docstring"
    )
    with open(github_docstrings_path, "r") as fin:
        github_docstrings_matrix = encode_doc_to_matrix(
            fin.readlines(), nl_vocab_encoder, nl_length
        )

    # remove empty entries of only padding
    github_code_matrix, github_docstrings_matrix = data.remove_empty_entries([
        github_code_matrix,
        github_docstrings_matrix,
    ])

    # We use CONALA training data as validation data
    conala_valid_code, conala_valid_nl = encode_conala_file(
        utils.relative_to_src_dir("../data/conala-corpus/conala-train.json"),
        code_vocab_encoder,
        nl_vocab_encoder,
        code_length,
        nl_length,
    )

    # We use CONALA test data as test data
    conala_test_code, conala_test_nl = encode_conala_file(
        utils.relative_to_src_dir("../data/conala-corpus/conala-test.json"),
        code_vocab_encoder,
        nl_vocab_encoder,
        code_length,
        nl_length,
    )

    utils.create_dir(utils.relative_to_src_dir("../data/conala-adv-valid/"))
    utils.create_dir(utils.relative_to_src_dir("../data/conala-adv-test/"))
    np.save(
        utils.relative_to_src_dir("../data/conala-adv-valid/github_code.npy"),
        github_code_matrix,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/conala-adv-valid/github_docstrings.npy"),
        github_docstrings_matrix,
    )
    np.save(
        utils.relative_to_src_dir("../data/conala-adv-valid/conala_code.npy"),
        conala_valid_code,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/conala-adv-valid/conala_queries.npy"),
        conala_valid_nl,
    )
    np.save(
        utils.relative_to_src_dir("../data/conala-adv-test/conala_code.npy"),
        conala_test_code,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/conala-adv-test/conala_queries.npy"),
        conala_test_nl,
    )


def build_test_datasets_conala_mined(config):
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-train-mined/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "rb") as fin:
        code_vocab_encoder = pickle.load(fin)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/conala-train-mined/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "rb") as fin:
        nl_vocab_encoder = pickle.load(fin)

    # We use CONALA test data as test data
    conala_test_code, conala_test_nl = encode_conala_file(
        utils.relative_to_src_dir("../data/conala-corpus/conala-test.json"),
        code_vocab_encoder,
        nl_vocab_encoder,
        code_length,
        nl_length,
    )

    utils.create_dir(utils.relative_to_src_dir("../data/conala-test-mined/"))
    np.save(
        utils.relative_to_src_dir("../data/conala-test-mined/conala_code.npy"),
        conala_test_code,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/conala-test-mined/conala_queries.npy"),
        conala_test_nl,
    )


def build_test_datasets_downsampled(config):
    code_length = config["code_length"]
    nl_length = config["nl_length"]

    code_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/github_downsampled/code_vocab_encoder.pkl"
    )
    with open(code_vocab_encoder_path, "rb") as fin:
        code_vocab_encoder = pickle.load(fin)

    nl_vocab_encoder_path = utils.relative_to_src_dir(
        "../data/github_downsampled/nl_vocab_encoder.pkl"
    )
    with open(nl_vocab_encoder_path, "rb") as fin:
        nl_vocab_encoder = pickle.load(fin)

    # We use CONALA test data as test data
    conala_test_code, conala_test_nl = encode_conala_file(
        utils.relative_to_src_dir("../data/conala-corpus/conala-test.json"),
        code_vocab_encoder,
        nl_vocab_encoder,
        code_length,
        nl_length,
    )

    utils.create_dir(utils.relative_to_src_dir("../data/github_downsampled/conala-test/"))
    np.save(
        utils.relative_to_src_dir("../data/github_downsampled/conala-test/conala_code.npy"),
        conala_test_code,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/github_downsampled/conala-test/conala_queries.npy"),
        conala_test_nl,
    )

    # add test data for downsampled
    github_code_path = utils.relative_to_src_dir(
        "../data/github/test.function_relevant"
    )
    with open(github_code_path, "r") as fin:
        github_code_matrix = encode_doc_to_matrix(
            fin.readlines(), code_vocab_encoder, code_length
        )

    github_docstrings_path = utils.relative_to_src_dir(
        "../data/github/test.docstring"
    )
    with open(github_docstrings_path, "r") as fin:
        github_docstrings_matrix = encode_doc_to_matrix(
            fin.readlines(), nl_vocab_encoder, nl_length
        )

    # remove empty entries of only padding
    github_code_matrix, github_docstrings_matrix = data.remove_empty_entries([
        github_code_matrix,
        github_docstrings_matrix,
    ])
    utils.create_dir(utils.relative_to_src_dir("../data/github_downsampled/github-test/"))
    np.save(
        utils.relative_to_src_dir("../data/github_downsampled/github-test/github_code.npy"),
        github_code_matrix,
    )
    np.save(
        utils.
        relative_to_src_dir("../data/github_downsampled/github-test/github_queries.npy"),
        github_docstrings_matrix,
    )



def get_args():
    parser = argparse.ArgumentParser(description="Build data matrices")
    parser.add_argument("config", type=str, help="Configuration name")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Choose dataset to build",
        choices=["github-so", "advconala", "conala-mined", "downsampled"]
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config = configuration.get_configuration(args.config)

    if args.dataset == "github-so":
        print("Building Github/SO dataset")
        build_train_datasets_github_so(config)
        build_test_datasets_github_so(config)
    elif args.dataset == "advconala":
        # Use the conala-mined as the queries for adversarial training
        print("Building CONALA-based adversarial dataset")
        config = configuration.get_configuration('basic')
        build_train_datasets_advconala(config)
        build_test_datasets_advconala(config)
    elif args.dataset == "conala-mined":
        print("Build simple dataset (no adversarial) using CONALA mined data")
        build_train_datasets_conala_mined(config)
        build_test_datasets_conala_mined(config)
    elif args.dataset == "downsampled":
        print("Building downsampled corpus")
        build_train_datasets_downsampled(config)
        build_test_datasets_downsampled(config)
    else:
        raise ValueError("Unknown dataset type: {}".format(args.dataset))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()

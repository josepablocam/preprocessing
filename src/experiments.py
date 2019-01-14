#!/usr/bin/env python3
import argparse
import glob
import json
import os
import pickle
import subprocess

import numpy as np

from .evaluation import (
    load_evaluation_data,
    evaluate_with_known_answer,
    load_model,
)
from . import preprocess
from .preprocess import CanonicalInput  # noqa F401
from .precomputed_embeddings import run_fasttext
from .vocabulary import encoder_from_embeddings, apply_encoder
from .train import train
from . import utils

# length used for both code and NL (for padding/truncating purposes)
TARGET_LEN = 50
# dimension of embeddings produced
DIMENSION = 100


def produce_experiment_configuration(
        train_data,
        test_data,
        code_pipeline,
        nl_pipeline,
        min_vocab_count,
        output_dir,
        target_len=TARGET_LEN,
        dim=DIMENSION,
        seed=None,
):

    if seed is not None:
        np.random.seed(seed)

    utils.create_dir(output_dir)
    train_data.reset_transformed()
    test_data.reset_transformed()

    print("Transforming training data")
    train_data.apply_pipeline(code_pipeline, which="code")
    train_data.apply_pipeline(nl_pipeline, which="nl")

    print("Transforming test data")
    test_data.apply_pipeline(code_pipeline, which="code")
    test_data.apply_pipeline(nl_pipeline, which="nl")

    train_code_path = os.path.join(output_dir, "train-code.txt")
    train_nl_path = os.path.join(output_dir, "train-nl.txt")
    train_data.to_text(
        train_code_path,
        train_nl_path,
    )

    test_code_path = os.path.join(output_dir, "test-code.txt")
    test_nl_path = os.path.join(output_dir, "test-nl.txt")

    test_data.to_text(
        test_code_path,
        test_nl_path,
    )

    combined_train_path = os.path.join(output_dir, "train-code-and-nl.txt")
    subprocess.call(
        "cat {} > {}".format(train_code_path, combined_train_path),
        shell=True,
    )
    subprocess.call(
        "cat {} >> {}".format(train_nl_path, combined_train_path),
        shell=True,
    )

    # concatenate code and nl into single file
    embeddings_raw_path = os.path.join(output_dir, "embeddings")
    embeddings_path = embeddings_raw_path + ".vec"
    run_fasttext(combined_train_path, min_vocab_count, dim,
                 embeddings_raw_path)
    # vocabulary encoder from embeddings
    encoder_path = os.path.join(output_dir, "encoder.pkl")
    print("Building encoder to {}.pkl".format(encoder_path))
    encoder_from_embeddings(embeddings_path, encoder_path)

    text_files = [
        "train-code.txt", "train-nl.txt", "test-code.txt", "test-nl.txt"
    ]
    text_files = [os.path.join(output_dir, p) for p in text_files]

    for input_path in text_files:
        output_path = os.path.splitext(input_path)[0] + ".npy"
        print("Encoding {} w/ {} to {}".format(input_path, encoder_path,
                                               output_path))
        apply_encoder(input_path, encoder_path, target_len, "numpy",
                      output_path)


def generate_experiments(
        train_data_path,
        test_data_path,
        output_dir,
        train_downsample=None,
        seed=None,
):
    with open(train_data_path, "rb") as fin:
        train_data = pickle.load(fin)

    with open(test_data_path, "rb") as fin:
        test_data = pickle.load(fin)

    if train_downsample is not None:
        train_data.downsample(train_downsample, seed=seed)

    utils.create_dir(output_dir)

    raw_lower_case_tokens = preprocess.sequence(
        preprocess.split_on_whitespace,
        preprocess.lower_case,
    )
    experiment_1 = {
        "name": "base",
        "code": raw_lower_case_tokens,
        "nl": raw_lower_case_tokens,
        "min_count": 5,
    }

    random_seeds = [1, 2, 10, 42, 100]
    experiments = []
    for seed in random_seeds:
        exp = dict(experiment_1)
        exp["seed"] = seed
        exp["output_dir"] = os.path.join(output_dir, "seed-{}".format(seed))
        experiments.append(exp)

    for exp in experiments:
        print("Generating experiment: {}".format(exp["name"]))
        produce_experiment_configuration(
            train_data,
            test_data,
            code_pipeline=exp["code"],
            nl_pipeline=exp["nl"],
            min_vocab_count=exp["min_count"],
            output_dir=exp["output_dir"],
            seed=exp.get("seed", None),
        )


def run_experiments(base_dir):
    experiment_folders = [
        os.path.join(base_dir, p) for p in os.listdir(base_dir)
    ]
    for experiment in experiment_folders:
        code_path = os.path.join(experiment, "train-code.npy")
        nl_path = os.path.join(experiment, "train-nl.npy")
        embeddings_path = os.path.join(experiment, "embeddings.vec")
        encoder_path = os.path.join(experiment, "encoder.pkl")
        train(
            code_path,
            nl_path,
            embeddings_path,
            encoder_path,
            print_every=10,
            save_every=1,
            output_folder=experiment,
        )

        test_code, test_queries = load_evaluation_data(
            os.path.join(experiment, "test-code.npy"),
            os.path.join(experiment, "test-nl.npy"),
        )

        model_paths = glob.glob(os.path.join(experiment, "models", "*latest"))
        assert len(
            model_paths) == 1, "Should only have 1 symlinked latest model"

        model = load_model(model_paths[0])

        eval_results = evaluate_with_known_answer(
            model,
            test_code,
            test_queries,
        )
        eval_results_path = os.path.join(experiment, "results.json")
        with open(eval_results_path, "w") as fout:
            json.dump(eval_results, fout)


def get_args():
    parser = argparse.ArgumentParser(
        description="Setup and run preprocessing experiments")
    subparsers = parser.add_subparsers(help="Actions")
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument(
        "--train",
        type=str,
        help="Path to train data in canonical input form",
    )
    gen_parser.add_argument(
        "--test",
        type=str,
        help="Path to test data in canonical input form",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        help="Output directory",
    )
    gen_parser.add_argument(
        "--downsample",
        type=int,
        help="Downsample training data to n observations",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed",
        default=42,
    )
    gen_parser.set_defaults(which="generate")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Root directory with experiment subfolders generated")
    run_parser.set_defaults(which="run")
    return parser.parse_args()


def main():
    args = get_args()
    if args.which == "generate":
        generate_experiments(
            args.train,
            args.test,
            args.output,
            train_downsample=args.downsample,
            seed=args.seed,
        )
    elif args.which == "run":
        run_experiments(args.data)
    else:
        raise ValueError("Unknown action: {}".format(args.which))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

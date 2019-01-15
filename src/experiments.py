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


def produce_embeddings(train_data, min_vocab_count, dim, output_dir):
    """
    Builds embeddings using fastText
    """
    code_path = os.path.join(output_dir, "/tmp/code.txt")
    nl_path = os.path.join(output_dir, "/tmp/nl.txt")
    train_data.to_text(
        code_path,
        nl_path,
    )
    combined_path = os.path.join(output_dir, "embedding-input.txt")
    subprocess.call(
        "cat {} > {}".format(code_path, combined_path),
        shell=True,
    )
    subprocess.call(
        "cat {} >> {}".format(nl_path, combined_path),
        shell=True,
    )

    # concatenate code and nl into single file
    embeddings_raw_path = os.path.join(output_dir, "embeddings")
    embeddings_path = embeddings_raw_path + ".vec"
    run_fasttext(combined_path, min_vocab_count, dim, embeddings_raw_path)
    return embeddings_path


def generate_experiment_folder(
        train_data,
        test_data,
        code_pipeline,
        nl_pipeline,
        min_vocab_count,
        output_dir,
        target_len=TARGET_LEN,
        dim=DIMENSION,
        downsample_n=None,
        seed=None,
):
    """
    Produces experiment subfolder with all necessary data.
    Order of operations:
        * Applies preprocessing pipelines to code and nl
        * Generates embeddings from *complete* transformed training data
            (starts with code input and appends at end of file NL input)
        * Downsamples training data if relevant
        * Extracts vocabulary from embeddings file
        * Writes out encoded version of training/test data using vocabulary
    """

    if seed is not None:
        np.random.seed(seed)

    train_data.reset()
    test_data.reset()

    print("Transforming training data")
    train_data.apply_pipeline(code_pipeline, which="code")
    train_data.apply_pipeline(nl_pipeline, which="nl")

    print("Transforming test data")
    test_data.apply_pipeline(code_pipeline, which="code")
    test_data.apply_pipeline(nl_pipeline, which="nl")

    utils.create_dir(output_dir)
    print("Producing embeddings")
    embeddings_path = produce_embeddings(
        train_data,
        min_vocab_count,
        dim,
        output_dir,
    )

    if downsample_n is not None:
        train_data.downsample(downsample_n, seed=seed)

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

    # vocabulary encoder from embeddings file
    encoder_path = os.path.join(output_dir, "encoder.pkl")
    print("Building encoder to {}.pkl".format(encoder_path))
    encoder_from_embeddings(embeddings_path, encoder_path)

    text_files = [
        "train-code.txt", "train-nl.txt", "test-code.txt", "test-nl.txt"
    ]
    text_files = [os.path.join(output_dir, p) for p in text_files]

    for input_path in text_files:
        output_path = os.path.splitext(input_path)[0] + ".npy"
        print("Encoding {} w/ {} to {}".format(
            input_path,
            encoder_path,
            output_path,
        ))
        apply_encoder(
            input_path,
            encoder_path,
            target_len,
            "numpy",
            output_path,
        )


def generate_experiments(
        train_data_path,
        test_data_path,
        output_dir,
        train_downsample=None,
        seed=None,
        force=False,
):
    """
    Generates subfolders for experiments
    """
    with open(train_data_path, "rb") as fin:
        train_data = pickle.load(fin)

    with open(test_data_path, "rb") as fin:
        test_data = pickle.load(fin)

    utils.create_dir(output_dir)


    empty_experiment = {
        "code": None,
        "nl": None,
        "min_count": 5,
        "downsample_n": 10000,
        "seed": 42,
    }

    # pipelines
    lower_case_pipeline = preprocess.sequence(
        preprocess.split_on_whitespace,
        preprocess.lower_case,
    )

    reasonable_code = preprocess.sequence(
        preprocess.plus(
            preprocess.extract_def_name,
            preprocess.extract_call_tokens,
        ),
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
    )

    reasonable_nl = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
    )


    # Experiments
    experiments = []

    raw_experiment = dict(empty_experiment)
    raw_experiment["code"] = lower_case_pipeline
    raw_experiment["nl"] = lower_case_pipeline
    raw_experiment["output_dir"] = os.path.join(output_dir, "raw")
    experiments.append(raw_experiment)


    reasonable_experiment = dict(empty_experiment)
    reasonable_experiment["code"] = reasonable_code
    reasonable_experiment["nl"] = reasonable_nl
    reasonable_experiment["output_dir"] = os.path.join(output_dir, "reasonable")
    experiments.append(reasonable_experiment)


    for exp in experiments:
        print("Generating experiment: {}".format(exp["output_dir"]))
        if os.path.exists(exp["output_dir"]) and not force:
            print("Skipping {}, output folder exists".format(exp["output_dir"]))
            print("Use --force if re-run is desired")
            continue

        generate_experiment_folder(
            train_data,
            test_data,
            code_pipeline=exp["code"],
            nl_pipeline=exp["nl"],
            min_vocab_count=exp["min_count"],
            output_dir=exp["output_dir"],
            downsample_n=exp["downsample_n"],
            seed=exp.get("seed", seed),
        )


def run_experiments(base_dir, force=False):
    experiment_folders = [
        os.path.join(base_dir, p) for p in os.listdir(base_dir)
    ]
    for experiment in experiment_folders:
        eval_results_path = os.path.join(experiment, "results.json")
        if os.path.exists(eval_results_path) and not force:
            print("Skipping {}, results.json exists".format(experiment))
            print("Use --force if re-run is desired")
            continue

        code_path = os.path.join(experiment, "train-code.npy")
        nl_path = os.path.join(experiment, "train-nl.npy")
        embeddings_path = os.path.join(experiment, "embeddings.vec")
        encoder_path = os.path.join(experiment, "encoder.pkl")
        train(
            code_path,
            nl_path,
            embeddings_path,
            encoder_path,
            print_every=1000,
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
    gen_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force generation of data, otherwise if folder exists skips",
    )
    gen_parser.set_defaults(which="generate")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Root directory with experiment subfolders generated")
    run_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force rerun of subdirectories with results (else skips)",
    )
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
            force=args.force,
        )
    elif args.which == "run":
        run_experiments(args.data, force=args.force)
    else:
        raise ValueError("Unknown action: {}".format(args.which))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

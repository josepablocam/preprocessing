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

# validation seed
VALID_SEED = 123123

# Test data is always modified with the same pipeline
TEST_PIPELINE = preprocess.sequence(
    preprocess.split_on_code_characters,
    preprocess.lower_case,
    preprocess.remove_english_stopwords,
    preprocess.stem_english_words,
)

EMPTY_EXPERIMENT = {
    "code": None,
    "nl": None,
    "code_test": None,
    "nl_test": None,
    "min_count": 5,
    "downsample_train": 10000,
    "downsample_valid": 500,
    #"seeds": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # DEBUG
    "seeds": [10, 20],
}


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
        code_test_pipeline,
        nl_test_pipeline,
        min_vocab_count,
        output_dir,
        target_len=TARGET_LEN,
        dim=DIMENSION,
        downsample_train=None,
        downsample_valid=None,
        seeds=None,
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
    if seeds is not None:
        # set initial seed in case tranformations in pipeline have randomness
        if not isinstance(seeds, list):
            seeds = [seeds]
        assert VALID_SEED not in seeds, "{} reserved for internal seed".format(
            VALID_SEED
        )
        np.random.seed(seeds[0])

    train_data.reset()
    test_data.reset()

    print("Transforming training data")
    train_data.apply_pipeline(code_pipeline, which="code")
    train_data.apply_pipeline(nl_pipeline, which="nl")

    print("Transforming test data")
    test_data.apply_pipeline(code_test_pipeline, which="code")
    test_data.apply_pipeline(nl_test_pipeline, which="nl")

    utils.create_dir(output_dir)
    print("Producing embeddings")
    embeddings_path = produce_embeddings(
        train_data,
        min_vocab_count,
        dim,
        output_dir,
    )

    # vocabulary encoder from embeddings file
    encoder_path = os.path.join(output_dir, "encoder.pkl")
    print("Building encoder to {}.pkl".format(encoder_path))
    encoder_from_embeddings(embeddings_path, encoder_path)

    # Test data: shared across all seeds
    test_code_path = os.path.join(output_dir, "test-code.txt")
    test_nl_path = os.path.join(output_dir, "test-nl.txt")
    encode_dataset(
        test_data,
        test_code_path,
        test_nl_path,
        encoder_path,
        target_len,
    )

    # held out downsampled data, shared by all seeds
    valid_downsampled = train_data.downsample(
        downsample_valid,
        seed=VALID_SEED,
    )

    valid_code_path = os.path.join(output_dir, "valid-code.txt")
    valid_nl_path = os.path.join(output_dir, "valid-nl.txt")
    encode_dataset(
        valid_downsampled,
        valid_code_path,
        valid_nl_path,
        encoder_path,
        target_len,
    )

    # Generate new train data for each seed in our experiments
    if seeds is None:
        # if no seeds provided, we will create data at the root of the
        # experiment folder
        seeds = [None]

    for train_seed in seeds:
        if train_seed is None:
            train_seed = 0  # random
            seed_dir = output_dir
        else:
            seed_dir = os.path.join(output_dir, "seed-{}".format(train_seed))
            os.makedirs(seed_dir, exist_ok=True)

        train_downsampled = train_data.downsample(
            downsample_train,
            seed=train_seed,
        )

        train_code_path = os.path.join(seed_dir, "train-code.txt")
        train_nl_path = os.path.join(seed_dir, "train-nl.txt")
        encode_dataset(
            train_downsampled,
            train_code_path,
            train_nl_path,
            encoder_path,
            target_len,
        )


def encode_dataset(ds, code_path, nl_path, encoder_path, target_len):
    ds.to_text(code_path, nl_path)
    paths = [code_path, nl_path]
    for input_path in paths:
        output_path = os.path.splitext(input_path)[0] + ".npy"
        print(
            "Encoding {} w/ {} to {}".format(
                input_path,
                encoder_path,
                output_path,
            )
        )
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

    experiments = paper_experiments(output_dir)
    # DEBUG
    experiments = [experiments[0]]

    for exp in experiments:
        print("Generating experiment: {}".format(exp["output_dir"]))
        if os.path.exists(exp["output_dir"]) and not force:
            print(
                "Skipping {}, output folder exists".format(exp["output_dir"])
            )
            print("Use --force if re-run is desired")
            continue

        generate_experiment_folder(
            train_data,
            test_data,
            code_pipeline=exp["code"],
            nl_pipeline=exp["nl"],
            code_test_pipeline=exp["code_test"],
            nl_test_pipeline=exp["nl_test"],
            min_vocab_count=exp["min_count"],
            output_dir=exp["output_dir"],
            downsample_train=exp["downsample_train"],
            downsample_valid=exp["downsample_valid"],
            seeds=exp.get("seeds", seed),
        )


def run_experiments(base_dir, model, test_setting, force=False):
    experiment_folders = [
        os.path.join(base_dir, p) for p in os.listdir(base_dir)
    ]
    for experiment_root in experiment_folders:
        test_code_paths = {}
        test_nl_paths = {}
        if test_setting == "conala":
            test_code_paths['conala'] = os.path.join(
                experiment_root, "test-code.npy"
            )
            test_nl_paths['conala'] = os.path.join(
                experiment_root, "test-nl.npy"
            )
        elif test_setting == "github":
            test_code_paths['github'] = os.path.join(
                experiment_root, "test-code-github.npy"
            )
            test_nl_paths['github'] = os.path.join(
                experiment_root, "test-nl-github.npy"
            )
        else:
            test_code_paths['conala'] = os.path.join(
                experiment_root, "test-code.npy"
            )
            test_nl_paths['conala'] = os.path.join(
                experiment_root, "test-nl.npy"
            )
            test_code_paths['github'] = os.path.join(
                experiment_root, "test-code-github.npy"
            )
            test_nl_paths['github'] = os.path.join(
                experiment_root, "test-nl-github.npy"
            )

        valid_code_path = os.path.join(experiment_root, "valid-code.npy")
        valid_nl_path = os.path.join(experiment_root, "valid-nl.npy")
        embeddings_path = os.path.join(experiment_root, "embeddings.vec")
        encoder_path = os.path.join(experiment_root, "encoder.pkl")
        # Models to run
        models = []
        if model == "lstm":
            models.append('lstm')
        elif model == 'dan':
            models.append('dan')
        else:
            models.append('lstm')
            models.append('dan')

        experiment_subfolders = glob.glob(experiment_root + "/seed*")
        if len(experiment_subfolders) == 0:
            print(
                "No seed folders used, data must be at: {}".
                format(experiment_root)
            )
            experiment_subfolders = [experiment_root]

        for seed_folder in experiment_subfolders:
            code_path = os.path.join(seed_folder, "train-code.npy")
            nl_path = os.path.join(seed_folder, "train-nl.npy")
            for test_option in test_code_paths.keys():
                for model_option in models:
                    exp_folder = os.path.join(
                        seed_folder, model_option + '-' + test_option
                    )
                    utils.create_dir(exp_folder)

                    run_single_experiment(
                        exp_folder,
                        code_path,
                        nl_path,
                        valid_code_path,
                        valid_nl_path,
                        test_code_paths[test_option],
                        test_nl_paths[test_option],
                        embeddings_path,
                        encoder_path,
                        model_option,
                        force=force,
                    )


def run_single_experiment(
        folder,
        code_path,
        nl_path,
        valid_code_path,
        valid_nl_path,
        test_code_path,
        test_nl_path,
        embeddings_path,
        encoder_path,
        model_option,
        force=False,
):
    eval_results_path = os.path.join(folder, "results.json")
    if os.path.exists(eval_results_path) and not force:
        print("Skipping {}, results.json exists".format(folder))
        print("Use --force if re-run is desired")
        return

    train(
        code_path,
        nl_path,
        embeddings_path,
        encoder_path,
        model_option,
        print_every=1000,
        save_every=50,
        num_epochs=100,
        output_folder=folder,
        valid_code_path=valid_code_path,
        valid_docstrings_path=valid_nl_path,
    )
    # load the best model based on validation loss
    model_paths = glob.glob(os.path.join(folder, "models", "*best.pth"))
    assert len(model_paths) == 1, "Should only have 1 best model"
    model = load_model(model_paths[0])

    test_code, test_queries = load_evaluation_data(
        test_code_path,
        test_nl_path,
    )

    eval_results = evaluate_with_known_answer(
        model,
        test_code,
        test_queries,
    )
    with open(eval_results_path, "w") as fout:
        json.dump(eval_results, fout)


def initial_experiments(output_dir):
    """Experiments used at the start to figure out what made sense to try"""
    # pipelines
    simplest_pipeline = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    # Experiments
    experiments = []

    # Simplest process (tokenize and lower case)
    exp0 = dict(EMPTY_EXPERIMENT)
    exp0["code"] = simplest_pipeline
    exp0["nl"] = simplest_pipeline
    exp0["code_test"] = simplest_pipeline
    exp0["nl_test"] = simplest_pipeline
    exp0["output_dir"] = os.path.join(output_dir, "exp0")
    experiments.append(exp0)

    # Split code characters but only method name
    exp1 = dict(EMPTY_EXPERIMENT)
    exp1["code"] = preprocess.sequence(
        preprocess.extract_qualified_def_name,
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    exp1["nl"] = simplest_pipeline
    exp1["code_test"] = simplest_pipeline
    exp1["nl_test"] = simplest_pipeline
    exp1["output_dir"] = os.path.join(output_dir, "exp1")
    experiments.append(exp1)

    # Split code characters, method name and calls
    exp2 = dict(EMPTY_EXPERIMENT)
    exp2["code"] = preprocess.sequence(
        preprocess.plus(
            preprocess.extract_qualified_def_name,
            preprocess.extract_call_tokens,
        ),
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    exp2["nl"] = simplest_pipeline
    exp2["code_test"] = simplest_pipeline
    exp2["nl_test"] = simplest_pipeline
    exp2["output_dir"] = os.path.join(output_dir, "exp2")
    experiments.append(exp2)

    # Remove stop words and stem
    exp3 = dict(EMPTY_EXPERIMENT)
    exp3["code"] = preprocess.sequence(
        preprocess.plus(
            preprocess.extract_qualified_def_name,
            preprocess.extract_call_tokens,
        ),
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
        preprocess.stem_english_words,
    )
    exp3["nl"] = simplest_pipeline
    exp3["code_test"] = TEST_PIPELINE
    exp3["nl_test"] = simplest_pipeline
    exp3["output_dir"] = os.path.join(output_dir, "exp3")
    experiments.append(exp3)

    # Removal of stopwords/stemming but for all tokens in code
    exp4 = dict(EMPTY_EXPERIMENT)
    exp4["code"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
        preprocess.stem_english_words,
    )
    exp4["nl"] = simplest_pipeline
    exp4["code_test"] = TEST_PIPELINE
    exp4["nl_test"] = simplest_pipeline
    exp4["output_dir"] = os.path.join(output_dir, "exp4")
    experiments.append(exp4)

    ##### Experiment on NL #####
    # Best sequence for code (need update)
    best_sequence_for_code = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
        preprocess.stem_english_words,
    )

    # NL: remove stopwords and stemming
    exp5 = dict(EMPTY_EXPERIMENT)
    exp5["code"] = best_sequence_for_code
    exp5["nl"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
        preprocess.stem_english_words,
    )
    exp5["code_test"] = best_sequence_for_code
    exp5["nl_test"] = TEST_PIPELINE
    exp5["output_dir"] = os.path.join(output_dir, "exp5")
    experiments.append(exp5)

    # NL: take description, remove param etc.
    exp6 = dict(EMPTY_EXPERIMENT)
    exp6["code"] = best_sequence_for_code
    exp6["nl"] = preprocess.sequence(
        preprocess.remove_params_and_returns,
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
        preprocess.stem_english_words,
    )
    exp6["code_test"] = best_sequence_for_code
    exp6["nl_test"] = TEST_PIPELINE
    exp6["output_dir"] = os.path.join(output_dir, "exp6")
    experiments.append(exp6)

    # NL: take first sentence in docstring
    exp7 = dict(EMPTY_EXPERIMENT)
    exp7["code"] = best_sequence_for_code
    exp7["nl"] = preprocess.sequence(
        preprocess.remove_params_and_returns,
        preprocess.take_first_sentence,
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
        preprocess.stem_english_words,
    )
    exp7["code_test"] = best_sequence_for_code
    exp7["nl_test"] = TEST_PIPELINE
    exp7["output_dir"] = os.path.join(output_dir, "exp7")
    experiments.append(exp7)
    return experiments


def paper_experiments(output_dir):
    """After discussion, the experiments we are running for paper"""
    experiments = []
    # base NL used for code experiments
    base_nl_pipeline = preprocess.sequence(
        preprocess.remove_params_and_returns,
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    # base code used for NL experiments
    base_code_pipeline = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    # Base experiment configurations
    code = dict(EMPTY_EXPERIMENT)
    code["nl"] = base_nl_pipeline
    code["nl_test"] = code["nl"]

    nl = dict(EMPTY_EXPERIMENT)
    nl["code"] = base_code_pipeline
    nl["code_test"] = nl["code"]

    # Code experiments
    code1 = dict(code)
    code1["code"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    code1["code_test"] = code1["code"]
    code1["output_dir"] = os.path.join(output_dir, "code-1")
    experiments.append(code1)

    code2 = dict(code)
    code2["code"] = preprocess.sequence(
        preprocess.extract_def_name_and_call_tokens_resilient,
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    code2["code_test"] = code2["code"]
    code2["output_dir"] = os.path.join(output_dir, "code-2")
    experiments.append(code2)

    code3 = dict(code)
    code3["code"] = preprocess.sequence(
        code2["code"],
        preprocess.remove_english_stopwords,
    )
    code3["code_test"] = code3["code"]
    code3["output_dir"] = os.path.join(output_dir, "code-3")
    experiments.append(code3)

    code4 = dict(code)
    code4["code"] = preprocess.sequence(
        code3["code"],
        preprocess.stem_english_words,
    )
    code4["code_test"] = code4["code"]
    code4["output_dir"] = os.path.join(output_dir, "code-4")
    experiments.append(code4)

    # NL experiments
    nl1 = dict(nl)
    nl1["nl"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    nl1["nl_test"] = nl1["nl"]
    nl1["output_dir"] = os.path.join(output_dir, "nl-1")
    experiments.append(nl1)

    nl2 = dict(nl)
    nl2["nl"] = preprocess.sequence(
        preprocess.remove_params_and_returns,
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    nl2["nl_test"] = nl2["nl"]
    nl2["output_dir"] = os.path.join(output_dir, "nl-2")
    experiments.append(nl2)

    nl3 = dict(nl)
    nl3["nl"] = preprocess.sequence(
        nl2["nl"],
        preprocess.remove_english_stopwords,
    )
    nl3["nl_test"] = nl3["nl"]
    nl3["output_dir"] = os.path.join(output_dir, "nl-3")
    experiments.append(nl3)

    nl4 = dict(nl)
    nl4["nl"] = preprocess.sequence(
        nl3["nl"],
        preprocess.stem_english_words,
    )
    nl4["nl_test"] = nl4["nl"]
    nl4["output_dir"] = os.path.join(output_dir, "nl-4")
    experiments.append(nl4)

    nl5 = dict(nl)
    nl5["nl"] = preprocess.sequence(
        preprocess.remove_params_and_returns,
        preprocess.take_first_sentence,
        preprocess.split_on_code_characters,
        preprocess.lower_case,
    )
    nl5["nl_test"] = nl5["nl"]
    nl5["output_dir"] = os.path.join(output_dir, "nl-5")
    experiments.append(nl5)

    return experiments


def get_args():
    parser = argparse.ArgumentParser(
        description="Setup and run preprocessing experiments"
    )
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
        help="Root directory with experiment subfolders generated"
    )
    run_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force rerun of subdirectories with results (else skips)",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        help="Model you want to run experiment with, can be lstm, dan, both",
        default='both',
    )
    run_parser.add_argument(
        "--test",
        type=str,
        help="Test setting you want to run on. Can be github, conala, both",
        default='conala',
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
        run_experiments(args.data, args.model, args.test, force=args.force)
    else:
        raise ValueError("Unknown action: {}".format(args.which))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

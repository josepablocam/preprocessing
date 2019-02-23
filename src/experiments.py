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

# for partial, determined by interpolating between 10k subsample 100 epochs
# and 10 epochs for full
NUM_EPOCHS = {
    "full": 10,
    "partial-10": 100,
    "partial-50": 96,
    "partial-100": 91,
    "partial-250": 78,
    "partial-500": 55,
    "partial-750": 32,
}

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
    "seeds": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}


def produce_embeddings(train_data,
                       min_vocab_count,
                       dim,
                       output_dir,
                       force=False):
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
    if os.path.exists(embeddings_path) and not force:
        print(
            "Skipping build embeddings, exist at: {}".format(embeddings_path))
    else:
        run_fasttext(combined_path, min_vocab_count, dim, embeddings_raw_path)
    return embeddings_path


def generate_experiment_folder(
        train_data,
        test_data_dict,
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
        force=False,
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
            VALID_SEED)
        np.random.seed(seeds[0])

    train_data.reset()
    for ds in test_data_dict.values():
        ds.reset()

    print("Transforming training data")
    train_data.apply_pipeline(code_pipeline, which="code")
    train_data.apply_pipeline(nl_pipeline, which="nl")

    print("Transforming test data")
    for ds in test_data_dict.values():
        ds.apply_pipeline(code_test_pipeline, which="code")
        ds.apply_pipeline(nl_test_pipeline, which="nl")

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
    for name, ds in test_data_dict.items():
        test_code_path = os.path.join(output_dir,
                                      "test-code-{}.txt".format(name))
        test_nl_path = os.path.join(output_dir, "test-nl-{}.txt".format(name))
        encode_dataset(
            ds,
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

        if downsample_train is not None:
            train_downsampled = train_data.downsample(
                downsample_train,
                seed=train_seed,
            )
        else:
            train_downsampled = train_data

        train_code_path = os.path.join(seed_dir, "train-code.txt")
        train_nl_path = os.path.join(seed_dir, "train-nl.txt")
        if os.path.exists(train_code_path) and not force:
            print(
                "Data already generated, skipping output: {}".format(seed_dir))
        else:
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


def filter_experiment_subset(experiments, subset):
    """Filter out experiment folders based on name (not path)"""
    if subset is None:
        return experiments
    clean_experiments = []
    subset = [os.path.basename(s) for s in subset]
    for entry in experiments:
        if isinstance(entry, str):
            # for experiment folder name when running
            folder = entry
        elif isinstance(entry, dict) and "output_dir" in entry:
            # for experiment configurations when generating data
            folder = entry["output_dir"]
        else:
            raise ValueError("Unhandled entry type: {}".format(entry))

        if os.path.basename(folder) not in subset:
            print("Ignoring {}, not in {}".format(folder, subset))
        else:
            clean_experiments.append(entry)
    # sort experiments based on the subset
    clean_experiments = sorted(
        clean_experiments,
        key=lambda elem: subset.index(os.path.basename(elem)),
    )
    return list(clean_experiments)


def generate_experiments(
        train_data_path,
        test_data_names,
        test_data_paths,
        output_dir,
        train_downsample=None,
        seed=None,
        subset=None,
        force=False,
):
    """
    Generates subfolders for experiments
    """
    with open(train_data_path, "rb") as fin:
        train_data = pickle.load(fin)

    test_data = {}
    for name, path in zip(test_data_names, test_data_paths):
        with open(path, "rb") as fin:
            test_data[name] = pickle.load(fin)

    utils.create_dir(output_dir)

    experiments = paper_experiments(output_dir)
    experiments = filter_experiment_subset(experiments, subset)

    for exp in experiments:
        print("Generating experiment: {}".format(exp["output_dir"]))
        # TODO: we'll do forcing further down, at the seed level
        # if os.path.exists(exp["output_dir"]) and not force:
        #     print(
        #         "Skipping {}, output folder exists".format(exp["output_dir"])
        #     )
        #     print("Use --force if re-run is desired")
        #     continue

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
            force=force,
        )


def get_test_data_paths(folder):
    paths = {
        "conala": {
            "nl": os.path.join(folder, "test-nl-conala.npy"),
            "code": os.path.join(folder, "test-code-conala.npy"),
        },
        "github": {
            "nl": os.path.join(folder, "test-nl-github.npy"),
            "code": os.path.join(folder, "test-code-github.npy"),
        },
    }
    return paths


def run_experiments(
        base_dir,
        model_option,
        test_option,
        tune=False,
        subset=None,
        force=False,
):
    # Models to run
    models = []
    if model_option == "lstm":
        models.append("lstm")
    elif model_option == "dan":
        models.append("dan")
    elif model_option == 'dannew':
        models.append("dan2")
        models.append("dan3")
    elif model_option == "all":
        models.append("lstm")
        models.append("dan")
    else:
        raise Exception("Uknown model option: {}".format(model_option))

    experiment_folders = [
        os.path.join(base_dir, p) for p in os.listdir(base_dir)
    ]
    experiment_folders = filter_experiment_subset(
        experiment_folders,
        subset,
    )
    for experiment_root in experiment_folders:
        # shared across seeds
        valid_code_path = os.path.join(experiment_root, "valid-code.npy")
        valid_nl_path = os.path.join(experiment_root, "valid-nl.npy")
        embeddings_path = os.path.join(experiment_root, "embeddings.vec")
        encoder_path = os.path.join(experiment_root, "encoder.pkl")
        test_paths = get_test_data_paths(experiment_root)

        # path to test depends on the experiment folder
        test_paths = {
            k: v
            for k, v in test_paths.items()
            if (k == test_option) or (test_option == "all")
        }

        if len(test_paths) == 0:
            raise Exception("Unknown test option: {}".format(test_option))

        # each seed produces a new subfolder, where we train using
        # a new sample of the training data
        experiment_subfolders = glob.glob(experiment_root + "/seed*")
        if len(experiment_subfolders) == 0:
            print("No seed folders used, data must be at: {}".format(
                experiment_root))
            experiment_subfolders = [experiment_root]

        for seed_folder in experiment_subfolders:
            code_path = os.path.join(seed_folder, "train-code.npy")
            nl_path = os.path.join(seed_folder, "train-nl.npy")
            for model_option in models:
                # each model gets own folder where it is evaluated
                # on chosen test datasets
                if tune:
                    exp_folder = os.path.join(seed_folder, model_option+'-tune')
                else:
                    exp_folder = os.path.join(seed_folder, model_option)
                utils.create_dir(exp_folder)
                num_epochs = NUM_EPOCHS.get(
                    os.path.basename(experiment_root),
                    100,
                )
                run_single_experiment(
                    exp_folder,
                    code_path,
                    nl_path,
                    valid_code_path,
                    valid_nl_path,
                    test_paths,
                    embeddings_path,
                    encoder_path,
                    model_option,
                    tune=tune,
                    force=force,
                    num_epochs=num_epochs,
                    tune=tune,
                )


def get_trained_model_path(folder):
    model_paths = glob.glob(os.path.join(folder, "models", "*best.pth"))
    if len(model_paths) > 1:
        raise Exception("Can only have 1 best model from validation")
    elif len(model_paths) == 0:
        return None
    else:
        return model_paths[0]


def run_single_experiment(
        folder,
        code_path,
        nl_path,
        valid_code_path,
        valid_nl_path,
        test_paths_dict,
        embeddings_path,
        encoder_path,
        model_option,
        tune=False,
        force=False,
        num_epochs=100,
        tune=False,
):

    if get_trained_model_path(folder) is not None and not force:
        print("Skipping training in {}, model exists".format(folder))
        print("Use --force if re-training is desired")
    else:
        fixed_embeddings = not tune
        train(
            code_path,
            nl_path,
            embeddings_path,
            encoder_path,
            model_option,
            fixed_embeddings=not tune,
            print_every=1000,
            save_every=100,
            num_epochs=num_epochs,
            output_folder=folder,
            fixed_embeddings=fixed_embeddings,
            valid_code_path=valid_code_path,
            valid_docstrings_path=valid_nl_path,
        )

    model = load_model(get_trained_model_path(folder))

    results = []
    for dataset_name, test_paths in test_paths_dict.items():
        print("Evaluating {} on {}".format(model_option, dataset_name))
        test_code, test_queries = load_evaluation_data(
            test_paths["code"],
            test_paths["nl"],
        )
        dataset_results = evaluate_with_known_answer(
            model,
            test_code,
            test_queries,
        )
        dataset_results["dataset"] = dataset_name
        dataset_results["model"] = model_option
        results.append(dataset_results)

    results_path = os.path.join(folder, "results.json")
    with open(results_path, "w") as fout:
        json.dump(results, fout)


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

    size = dict(EMPTY_EXPERIMENT)
    size["code"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
    )
    size["code_test"] = size["code"]
    size["nl"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
    )
    size["nl_test"] = size["nl"]

    full_base = dict(EMPTY_EXPERIMENT)
    full_base["code"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
    )
    full_base["code_test"] = full_base["code"]
    full_base["nl"] = preprocess.sequence(
        preprocess.split_on_code_characters,
        preprocess.lower_case,
        preprocess.remove_english_stopwords,
    )
    full_base["nl_test"] = full_base["nl"]
    full_base["min_count"] = 10
    full_base["downsample_train"] = None
    full_base["downsample_valid"] = 500
    full_base["seeds"] = [10]

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

    code5 = dict(code)
    code5["code"] = preprocess.sequence(
        code1["code"],
        preprocess.remove_english_stopwords,
    )
    code5["code_test"] = code5["code"]
    code5["output_dir"] = os.path.join(output_dir, "code-5")
    experiments.append(code5)

    code6 = dict(code)
    code6["code"] = preprocess.sequence(
        code5["code"],
        preprocess.stem_english_words,
    )
    code6["code_test"] = code6["code"]
    code6["output_dir"] = os.path.join(output_dir, "code-6")
    experiments.append(code6)

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

    nl6 = dict(nl)
    nl6["nl"] = preprocess.sequence(
        nl1["nl"],
        preprocess.remove_english_stopwords,
    )
    nl6["nl_test"] = nl6["nl"]
    nl6["output_dir"] = os.path.join(output_dir, "nl-6")
    experiments.append(nl6)

    nl7 = dict(nl)
    nl7["nl"] = preprocess.sequence(
        nl6["nl"],
        preprocess.stem_english_words,
    )
    nl7["nl_test"] = nl7["nl"]
    nl7["output_dir"] = os.path.join(output_dir, "nl-7")
    experiments.append(nl7)

    # vocab size experiments
    # Note that min_count=5 we already have
    # some large min frequency experiments to observe performance degradation
    vocab_sizes = [1, 10, 15, 20, 100, 1000, 10000]
    for ix, vocab_size in enumerate(vocab_sizes, start=1):
        size_config = dict(size)
        size_config["min_count"] = vocab_size
        size_config["output_dir"] = os.path.join(output_dir,
                                                 "size-{}".format(vocab_size))
        experiments.append(size_config)

    full = dict(full_base)
    full["output_dir"] = os.path.join(output_dir, "full")
    experiments.append(full)

    # increasing amounts of data for training
    partial_sizes = [10, 50, 100, 250, 500, 750]
    num_seeds = [10, 9, 9, 7, 5, 3]

    for downsample_size, num_seed in zip(partial_sizes, num_seeds):
        partial_config = dict(full_base)
        partial_config["downsample_train"] = int(downsample_size * 1e3)
        partial_config["output_dir"] = os.path.join(
            output_dir,
            "partial-{}".format(downsample_size),
        )
        partial_config["seeds"] = list(range(10, (num_seed + 1) * 10, 10))
        experiments.append(partial_config)

    return experiments


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
        "--test_names",
        type=str,
        nargs="+",
        help="Names for test data sets (used as suffix in generated data)",
    )
    gen_parser.add_argument(
        "--test_paths",
        type=str,
        nargs="+",
        help="Paths to test datasets in canonical input form",
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
    gen_parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        help="Subset of experiment folders to generate (name, not full path)",
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
    run_parser.add_argument(
        "--model",
        type=str,
        help="Model you want to run experiment with, can be lstm, dan, all",
        default="all",
        choices=["lstm", "dan", "dannew", "all"],
    )
    run_parser.add_argument(
        "--test",
        type=str,
        help="Test setting you want to run on. Can be github, conala, all",
        default="all",
        choices=["github", "conala", "all"],
    )
    run_parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        help="Subset of experiment folders to run (name, not full path)",
    )
    run_parser.add_argument(
        "--tune",
        action="store_true",
        help="Train models with tune-able embeddings (i.e. not fixed)",
    )
    run_parser.set_defaults(which="run")
    return parser.parse_args()


def main():
    args = get_args()
    if args.which == "generate":
        generate_experiments(
            args.train,
            args.test_names,
            args.test_paths,
            args.output,
            train_downsample=args.downsample,
            seed=args.seed,
            subset=args.subset,
            force=args.force,
        )
    elif args.which == "run":
        run_experiments(
            args.data,
            args.model,
            args.test,
            tune=args.tune,
            subset=args.subset,
            force=args.force,
            tune=args.tune,
        )
    else:
        raise ValueError("Unknown action: {}".format(args.which))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

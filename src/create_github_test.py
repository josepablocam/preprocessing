#!/usr/bin/env python3
import argparse
import copy
import json
import pickle

import numpy as np

from .github import load
from .preprocess import CanonicalInput, remove_params_and_returns


def sample_test_data(input_path, seed, test_n, output_path):
    orig_data = load(input_path)
    # sample a subset of it
    np.random.seed(seed)
    np.random.shuffle(orig_data)
    sampled = orig_data[:test_n]

    # wrap in usual class for convenience
    sampled_ds = CanonicalInput(sampled)
    sampled_ds.apply_pipeline(remove_params_and_returns, which="nl")
    sampled_json = copy.deepcopy(sampled_ds.transformed)
    # add back in the original natural language
    sampled_ds.reset()
    for ix, d in enumerate(sampled_json):
        d["original_nl"] = sampled_ds.corpus[ix]["nl"]

    with open(output_path, "w") as fout:
        json.dump(sampled_json, fout, indent=2)


def get_args():
    parser = argparse.ArgumentParser(
        description="Sample test observations from github data")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to test json data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to dump json of test observations",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="RNG seed to sample",
        default=42,
    )
    parser.add_argument(
        "-n",
        "--test_n",
        type=int,
        help="Number of observations to sample as test data",
        default=1000,
    )
    return parser.parse_args()


def main():
    args = get_args()
    sample_test_data(args.input, args.seed, args.test_n, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

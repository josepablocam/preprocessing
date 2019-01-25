import argparse
import json

import numpy as np
import torch

from sklearn.metrics.pairwise import cosine_similarity

from .code_search_model import UnsupervisedModel


def location_of_correct(sim_mat):
    # sort code snippets based similarity for each query
    sorted_ixs = np.argsort(np.negative(sim_mat), axis=1)
    # the known answers are the "diagonal" values
    # ie. query 0 is paired with code 0 etc
    ground_truth = np.arange(0, sim_mat.shape[0]).reshape(-1, 1)
    # True for correct entries
    flag_for_correct = sorted_ixs == ground_truth
    return np.where(flag_for_correct)[1]


def get_mrr(locs):
    # start at index zero so offset by 1
    return np.mean(1.0 / (1 + locs))


def get_fraction_correct_at(locs, cutoff):
    return np.mean(locs < cutoff)


def load_evaluation_data(code_path, queries_path, num_obs=None):
    code = np.load(code_path)
    queries = np.load(queries_path)
    code = torch.tensor(code, dtype=torch.long)
    queries = torch.tensor(queries, dtype=torch.long)
    n = code.shape[0]
    if num_obs is not None and n > num_obs:
        print("Sampling {} obs for evaluation".format(num_obs))
        randixs = np.random.choice(
            np.arange(0, n), size=num_obs, replace=False
        )
        code = code[randixs]
        queries = queries[randixs]
    return code, queries


def normalize(mat):
    denoms = np.sqrt((mat**2).sum(axis=1).reshape(-1, 1))
    return mat / denoms


def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")
    model = model.eval()
    model = model.cpu()
    return model


def evaluate_with_known_answer(
        model,
        code,
        queries,
):
    assert code.shape[0] == queries.shape[0], "Code/queries must be aligned"

    with torch.no_grad():
        code_emb = model.embed_code(code).numpy()
        queries_emb = model.embed_nl(queries).numpy()

    # compute similarity matrix and from that the position of correct answer
    sim_mat = cosine_similarity(queries_emb, code_emb)
    ans_locs = location_of_correct(sim_mat)

    summary = {}
    mr = np.mean(ans_locs + 1)
    mrr = get_mrr(ans_locs)
    summary["mrr"] = mrr

    cutoffs = [1, 5, 10]
    fracs = []

    for c in cutoffs:
        frac = get_fraction_correct_at(ans_locs, c)
        fracs.append(frac)
    print("Num obs: {}".format(code.shape[0]))
    print("Mean Rank: {}".format(mr))
    print("MRR: {}".format(mrr))

    for c, f in zip(cutoffs, fracs):
        print("Fraction Correct@{}: {}".format(c, f))
        summary["success@{}".format(c)] = f
    return summary


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="Path to trained model",
    )
    parser.add_argument(
        "-c",
        "--code_path",
        type=str,
        help="Path to saved code dataset (must be aligned with queries)",
    )
    parser.add_argument(
        "-q",
        "--queries_path",
        type=str,
        help="Path to saved queries dataset (must be aligned with code)",
    )
    parser.add_argument(
        "-n",
        "--num_obs",
        type=int,
        help="Sample number of observations for test",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--rng_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    np.random.seed(args.rng_seed)
    code, queries = load_evaluation_data(
        args.code_path,
        args.queries_path,
        num_obs=args.num_obs,
    )

    model = load_model(args.model_path)

    res = evaluate_with_known_answer(
        model,
        code,
        queries,
    )
    if args.output is not None:
        with open(args.output, "w") as fout:
            json.dump(res, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

import argparse
from collections import defaultdict
import glob
import os
import json
import sys

import pandas as pd

from . import utils

RESULTS_FILE = "results.json"
STATS_FILE = "stats-dannew.json"


def get_args():
    parser = argparse.ArgumentParser(
        description="Get the result statistics of the preprocessing experiments"
    )
    subparsers = parser.add_subparsers(help="Actions")
    compute_parser = subparsers.add_parser("compute")
    compute_parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Root directory of experiments",
    )
    compute_parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        help="Subset of exp folders to compute stats",
    )
    compute_parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        help="Models to compute stats on",
    )
    compute_parser.set_defaults(which="compute")

    table_parser = subparsers.add_parser("latex")
    table_parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Root directory of experiments with stats.json computed already",
    )
    table_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save down latex of tables",
    )
    table_parser.add_argument(
        "-s",
        "--subset",
        type=str,
        nargs="+",
        help="Subset of experiment folders to use",
    )
    table_parser.set_defaults(which="latex")
    return parser.parse_args()


def get_experiment_folders(root_folder):
    return [os.path.join(root_folder, p) for p in os.listdir(root_folder) if not p=='full']


def compute_stats(root_folder, args):
    experiment_folders = get_experiment_folders(root_folder)
    if args.subset is not None:
        filtered = []
        for exp_folder in experiment_folders:
            if os.path.basename(exp_folder) in args.subset:
                filtered.append(exp_folder)
        experiment_folders = filtered
    for setting_folder in experiment_folders:
        experiment_subfolders = glob.glob(setting_folder + "/seed*")
        n = len(experiment_subfolders)  # Number of seeds
        results = {}
        models = ['lstm', 'dan']
        if args.model is not None:
            models = args.model
        tests = ['conala', 'github']
        metrics = ['mrr', 'success@1', 'success@5', 'success@10']
        for model in models:
            results[model] = {}
            for test in tests:
                results[model][test] = {}
                for metric in metrics:
                    results[model][test][metric] = []

        for seed_folder in experiment_subfolders:
            # Read results
            for model in models:
                model_path = os.path.join(seed_folder, model, RESULTS_FILE)
                if not os.path.exists(model_path):
                    print("Skipping {}, doesnt exist".format(model_path))
                    continue
                with open(model_path, 'r') as f:
                    data = json.load(f)
                    for data_entry in data:
                        for metric in metrics:
                            results[data_entry['model']][
                                data_entry['dataset']][metric].append(
                                    data_entry[metric])

        # Compute confidence interval
        confidence = 0.95
        results['confidence'] = confidence
        for model in models:
            for test in tests:
                for metric in metrics:
                    results[model][test][
                        metric] = utils.mean_confidence_interval(
                            results[model][test][metric],
                            confidence=confidence)

        # Store results under setting root folder
        results_path = os.path.join(setting_folder, STATS_FILE)
        with open(results_path, "w") as fout:
            json.dump(results, fout)
        print('Finish compute for ' + setting_folder)


def build_tables(all_stats):
    table_data = defaultdict(lambda: [])
    for folder_name, model_stats in all_stats.items():
        for model_name, dataset_stats in model_stats.items():
            if not isinstance(dataset_stats, dict):
                print("Skipping field: {}".format(dataset_stats))
                continue
            for dataset, values in dataset_stats.items():
                row = {
                    value_name: "{:.2f} ({:.2f})".format(v, ci)
                    for value_name, (v, ci) in values.items()
                }
                row["pipeline"] = folder_name
                table_data[(model_name, dataset)].append(row)
    dfs = {k: pd.DataFrame(vs) for k, vs in table_data.items()}
    # sort by pipeline name and order of columns
    cols = ["pipeline", "mrr", "success@1", "success@5", "success@10"]
    dfs = {k: df.sort_values("pipeline")[cols] for k, df in dfs.items()}
    return dfs


def to_latex(dfs):
    tables = []
    for (model, dataset), df in dfs.items():
        table_str = """
        \\begin{{table}}
        \\caption{{Results for model {} on dataset {}}}
        {}
        \\end{{table}}
        """.format(model, dataset, df.to_latex(index=False))
        tables.append(table_str)

    document = """
    \\documentclass{{article}}
    \\usepackage{{booktabs}}
    \\begin{{document}}
    {}
    \\end{{document}}\n
    """.format("\n".join(tables))
    return document


def build_latex_doc(root_folder, args):
    output = args.output
    experiment_folders = get_experiment_folders(root_folder)
    if args.subset is not None:
        filtered = []
        for exp_folder in experiment_folders:
            if os.path.basename(exp_folder) in args.subset:
                filtered.append(exp_folder)
        experiment_folders = filtered
    data = {}
    for folder in experiment_folders:
        if folder is not None and subset is not None and os.path.basename(folder) not in subset:
            print("Skipping {}, not in {}".format(folder, subset))
            continue

        stats_path = os.path.join(folder, STATS_FILE)
        with open(stats_path, "r") as fin:
            stats = json.load(fin)
            data[os.path.basename(folder)] = stats
    dfs = build_tables(data)
    doc = to_latex(dfs)
    with open(output, "w") as fout:
        fout.write(doc)


def main():
    args = get_args()
    # For each exp settings
    root_folder = args.data
    if args.which == "compute":
        compute_stats(root_folder, args)
    elif args.which == "latex":
        build_latex_doc(root_folder, args)
    else:
        raise Exception("Unknown action")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()

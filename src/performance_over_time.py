import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

from .evaluation import (
    load_model,
    load_evaluation_data,
    evaluate_with_known_answer,
)


def collect_evaluation_stats(model_run_log, code, queries, label):
    df_models = pd.read_csv(
        model_run_log, header=None, names=["model_path", "time"]
    )
    is_sim_model = df_models.model_path.str.contains("sim_model")
    df_models = df_models[is_sim_model]

    results = []
    for _, row in tqdm.tqdm(df_models.iterrows()):
        model_path = row.model_path
        epoch = int(model_path.split("_")[-1][:-4])
        time = row.time
        model = load_model(model_path)
        res = evaluate_with_known_answer(model, code, queries)
        res.update({'model': model_path, 'epoch': epoch, 'time': time})
        results.append(res)

    df_results = pd.DataFrame(results)
    df_results["label"] = label
    return df_results


def run(model_run_logs, labels, code, queries):
    if labels is None:
        labels = [os.path.dirname(l) for l in model_run_logs]

    dfs = []
    for model_run_log, label in zip(model_run_logs, labels):
        df = collect_evaluation_stats(model_run_log, code, queries, label)
        dfs.append(df)
    combined = pd.concat(dfs)
    stats = ["mrr", "success@1", "success@5", "success@10"]
    time_dim = ["epoch", "time"]
    for s in stats:
        for t in time_dim:
            sns.lineplot(data=combined, hue="label", x=t, y=s)
            plt.show(block=True)

def run_with_number(model_run_logs, labels, code, queries):
    if labels is None:
        labels = [os.path.dirname(l) for l in model_run_logs]

    dfs = []
    for model_run_log, label in zip(model_run_logs, labels):
        df = collect_evaluation_stats(model_run_log, code, queries, label)
        dfs.append(df)
    combined = pd.concat(dfs)
    stats = ["mrr", "success@10"]
    time_dim = ["epoch"]

    # Output best stats
    for label in labels:
        print("Best mrr for model "+label+":")
        print(combined.loc[combined["label"]==label, "mrr"].max())
        print(combined.loc[combined["label"]==label, "mrr"].idxmax())
        print("Best success at 10 for model "+label+":")
        print(combined.loc[combined["label"]==label, "success@10"].max())
        print(combined.loc[combined["label"]==label, "mrr"].idxmax())
        
    for s in stats:
        for t in time_dim:
            sns.lineplot(data=combined, hue="label", x=t, y=s)
            plt.show(block=True)

# Training loss plot (tensorboard)
# Validation stats plot
# Validation best number output

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "-m",
        "--model_run_logs",
        type=str,
        nargs="+",
        help="Paths to model timestamps.csv",
    )
    parser.add_argument(
        "-l",
        "--model_labels",
        type=str,
        nargs="+",
        help="Labels for models in plots",
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

    run_with_number(
        args.model_run_logs,
        args.model_labels,
        code,
        queries,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
